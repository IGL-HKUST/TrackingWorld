# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


@dataclass(eq=False)
class DeltaData:
    """
    Dataclass for storing video tracks data.
    """

    video: torch.Tensor = None  # B, S, C, H, W
    trajectory: torch.Tensor = None  # B, S, N, 3
    visibility: torch.Tensor = None  # B, S, N
    # optional data
    depth_init: torch.Tensor = None
    trajectory_d: torch.Tensor = None  # B, S, N
    dense_trajectory: torch.Tensor = None  # B, S, N
    dense_trajectory_d: torch.Tensor = None  # B, S, N
    dense_valid: torch.Tensor = None  # B, S, N
    videodepth: torch.Tensor = None
    videodepth_pred: torch.Tensor = None
    videodisp: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None  # B, S, N
    segmentation: Optional[torch.Tensor] = None  # B, S, 1, H, W
    seq_name: Optional[str] = None
    dataset_name: Optional[str] = None
    query_points: Optional[torch.Tensor] = None  # TapVID evaluation format
    videodepth: Optional[torch.Tensor] = None  # B, S, C, H, W
    intrs: Optional[torch.Tensor] = torch.eye(3).unsqueeze(0)  # B, S, C, H, W
    extrs: Optional[torch.Tensor] = torch.eye(3).unsqueeze(0)
    trajectory3d: Optional[torch.Tensor] = None  # B, S, N, 2
    dino_features: Optional[torch.Tensor] = None
    flow: Optional[torch.Tensor] = None  # B, H, W, 2
    flow_depth: Optional[torch.Tensor] = None  # B, H, W, 2
    flow_alpha: Optional[torch.Tensor] = None  # B, S, N, 2
    flow_valid: Optional[torch.Tensor] = None  # B, S, N

    rev_flow: Optional[torch.Tensor] = None  # B, H, W, 2
    rev_flow_depth: Optional[torch.Tensor] = None  # B, H, W, 2
    rev_flow_alpha: Optional[torch.Tensor] = None  # B, S, N, 2

    dense_rev_trajectory: torch.Tensor = None  # B, S, N
    dense_rev_trajectory_d: torch.Tensor = None  # B, S, N

    cond_flow: Optional[torch.Tensor] = None  # B, H, W, 2
    cond_flow_depth: Optional[torch.Tensor] = None  # B, H, W, 2
    cond_flow_alpha: Optional[torch.Tensor] = None  # B, S, N, 2

    cond_trajectory: torch.Tensor = None  # B, S, N
    cond_trajectory_d: torch.Tensor = None  # B, S, N

    traj_grid: Optional[torch.Tensor] = None  # B, H, W, 2

    data_type: Optional[torch.Tensor] = torch.zeros((1))

    depth_min: Optional[torch.Tensor] = None  # B, H, W, 2
    depth_max: Optional[torch.Tensor] = None  # B, H, W, 2

    dense_query_frame: Optional[torch.Tensor] = None  # B, H, W, 2


def collate_fn(batch):
    """
    Collate function for video tracks data.
    """

    collated_data = DeltaData()

    fields = dataclasses.fields(DeltaData)
    for f in fields:

        if hasattr(batch[0], f.name) and getattr(batch[0], f.name) is not None:
            # print("get", f.name, getattr(batch[0], f.name))
            list_data = [getattr(b, f.name) for b in batch]
            if isinstance(list_data[0], torch.Tensor):
                list_data = torch.stack(list_data, dim=0)
            setattr(collated_data, f.name, list_data)

    return collated_data


def collate_fn_train(batch):
    """
    Collate function for video tracks data during training.
    """
    gotit = [gotit for _, gotit in batch]

    collated_data = DeltaData()

    collated_data.video = torch.stack([b.video for b, _ in batch], dim=0)

    fields = dataclasses.fields(DeltaData)
    for f in fields:
        if f.name in ["video"]:
            continue

        if hasattr(batch[0][0], f.name) and getattr(batch[0][0], f.name) is not None:
            # print("get", f.name)
            list_data = [getattr(b, f.name) for b, _ in batch]
            if isinstance(list_data[0], torch.Tensor):
                list_data = torch.stack(list_data, dim=0)
            # d = torch.stack([getattr(b, f.name) for b, _ in batch], dim=0)
            setattr(collated_data, f.name, list_data)

    return collated_data, gotit


def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input.

    Returns:
        t_cuda: `t` moved to a cuda device, if supported.
    """
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj):
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj


def aug_depth(
    depth,
    grid: tuple = (8, 8),
    scale: tuple = (0.7, 1.3),
    shift: tuple = (-0.1, 0.1),
    gn_kernel: tuple = (7, 7),
    gn_sigma: tuple = (2.0, 2.0),
    mask_depth: torch.Tensor = None,
):
    """
    Augment depth for training.
    Include:
        - low resolution scale and shift augmentation
        - gaussian blurring with random kernel size

    Args:
        depth: 1 T H W tensor
        grid: resolution for scale and shift augmentation 16 * 16 by default

    """

    B, T, H, W = depth.shape

    if mask_depth is None:
        msk = depth != 0
    else:
        msk = mask_depth
    # generate the scale and shift map
    H_s, W_s = grid
    scale_map = torch.rand(B, T, H_s, W_s, device=depth.device) * (scale[1] - scale[0]) + scale[0]
    shift_map = torch.rand(B, T, H_s, W_s, device=depth.device) * (shift[1] - shift[0]) + shift[0]

    # scale and shift the depth map
    scale_map = F.interpolate(scale_map, (H, W), mode="bilinear", align_corners=True)
    shift_map = F.interpolate(shift_map, (H, W), mode="bilinear", align_corners=True)

    # local scale and shift the depth
    depth[msk] = (depth[msk] * scale_map[msk]) + shift_map[msk] * (depth[msk].mean())

    # gaussian blur
    depth = TF.gaussian_blur(depth, kernel_size=gn_kernel, sigma=gn_sigma)
    depth[~msk] = 0.01

    return depth


def add_noise_depth(depth, gn_sigma: float = 0.3, mask_depth: torch.Tensor = None):
    """
    Augment depth for training.
    Include:
        - low resolution scale and shift augmentation
        - gaussian blurring with random kernel size

    Args:
        depth: 1 T H W tensor
        grid: resolution for scale and shift augmentation 16 * 16 by default

    """

    B, T, H, W = depth.shape

    if mask_depth is None:
        msk = depth != 0
    else:
        msk = mask_depth

    noise1 = torch.rand((B, T, 3, 4), device=depth.device) * gn_sigma
    noise2 = torch.rand((B, T, 48, 64), device=depth.device) * gn_sigma / 3
    noise3 = torch.rand((B, T, 96, 128), device=depth.device) * gn_sigma / 9

    # print(noise1.shape, noise2.shape, noise3.shape)
    # print(noise1.max(), noise2.max(), noise3.max())

    noise1 = F.interpolate(noise1, (H, W), mode="bilinear", align_corners=True)
    noise2 = F.interpolate(noise2, (H, W), mode="bilinear", align_corners=True)
    noise3 = F.interpolate(noise1, (H, W), mode="bilinear", align_corners=True)

    depth[msk] = depth[msk] + noise1[msk] + noise2[msk] + noise3[msk]
    depth[~msk] = 0.01

    # scale and shift the depth map
    # scale_map = F.interpolate(scale_map, (H, W),
    #                           mode='bilinear', align_corners=True)
    # shift_map = F.interpolate(shift_map, (H, W),
    #                           mode='bilinear', align_corners=True)

    # # local scale and shift the depth
    # depth[msk] = (depth[msk] * scale_map[msk])+ shift_map[msk]*(depth[msk].mean())

    # # gaussian blur
    # depth = TF.gaussian_blur(depth, kernel_size=gn_kernel, sigma=gn_sigma)
    # depth[~msk] = 0.01

    return depth
