# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# import pytorch3d.ops as torch3d
from einops import einsum, rearrange, repeat


EPS = 1e-6


def smart_cat(tensor1, tensor2, dim):
    if tensor1 is None:
        return tensor2
    return torch.cat([tensor1, tensor2], dim=dim)


# def get_grid(height, width, shape=None, dtype="torch", device="cpu", align_corners=True, normalize=True):
#     H, W = height, width
#     S = shape if shape else []
#     if align_corners:
#         x = torch.linspace(0, 1, W, device=device)
#         y = torch.linspace(0, 1, H, device=device)
#         if not normalize:
#             x = x * (W - 1)
#             y = y * (H - 1)
#     else:
#         x = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device)
#         y = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device)
#         if not normalize:
#             x = x * W
#             y = y * H
#     x_view, y_view, exp = [1 for _ in S] + [1, -1], [1 for _ in S] + [-1, 1], S + [H, W]
#     x = x.view(*x_view).expand(*exp)
#     y = y.view(*y_view).expand(*exp)
#     grid = torch.stack([x, y], dim=-1)
#     if dtype == "numpy":
#         grid = grid.numpy()
#     return grid


def get_points_on_a_grid(
    size: Union[int, Tuple[int, ...]],
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    if isinstance(size, int):
        size = (size, size)

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size[0], device=device),
        torch.linspace(*range_x, size[1], device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


def reduce_masked_mean(input, mask, dim=None, keepdim=False):
    r"""Masked mean

    `reduce_masked_mean(x, mask)` computes the mean of a tensor :attr:`input`
    over a mask :attr:`mask`, returning

    .. math::
        \text{output} =
        \frac
        {\sum_{i=1}^N \text{input}_i \cdot \text{mask}_i}
        {\epsilon + \sum_{i=1}^N \text{mask}_i}

    where :math:`N` is the number of elements in :attr:`input` and
    :attr:`mask`, and :math:`\epsilon` is a small constant to avoid
    division by zero.

    `reduced_masked_mean(x, mask, dim)` computes the mean of a tensor
    :attr:`input` over a mask :attr:`mask` along a dimension :attr:`dim`.
    Optionally, the dimension can be kept in the output by setting
    :attr:`keepdim` to `True`. Tensor :attr:`mask` must be broadcastable to
    the same dimension as :attr:`input`.

    The interface is similar to `torch.mean()`.

    Args:
        inout (Tensor): input tensor.
        mask (Tensor): mask.
        dim (int, optional): Dimension to sum over. Defaults to None.
        keepdim (bool, optional): Keep the summed dimension. Defaults to False.

    Returns:
        Tensor: mean tensor.
    """

    mask = mask.expand_as(input)

    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (EPS + denom)
    return mean


def reduce_masked_median(x, mask, keep_batch=False):
    # x and mask are the same shape
    assert x.size() == mask.size()
    device = x.device

    B = list(x.shape)[0]
    x = x.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    if keep_batch:
        x = np.reshape(x, [B, -1])
        mask = np.reshape(mask, [B, -1])
        meds = np.zeros([B], np.float32)
        for b in list(range(B)):
            xb = x[b]
            mb = mask[b]
            if np.sum(mb) > 0:
                xb = xb[mb > 0]
                meds[b] = np.median(xb)
            else:
                meds[b] = np.nan
        meds = torch.from_numpy(meds).to(device)
        return meds.float()
    else:
        x = np.reshape(x, [-1])
        mask = np.reshape(mask, [-1])
        if np.sum(mask) > 0:
            x = x[mask > 0]
            med = np.median(x)
        else:
            med = np.nan
        med = np.array([med], np.float32)
        med = torch.from_numpy(med).to(device)
        return med.float()


def bilinear_sampler(input, coords, mode="bilinear", align_corners=True, padding_mode="border", return_mask=False):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        Tensor: sampled points.
    """

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        coords = coords * torch.tensor([2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device)
    else:
        coords = coords * torch.tensor([2 / size for size in reversed(sizes)], device=coords.device)

    coords -= 1

    out = F.grid_sample(input, coords, mode=mode, align_corners=align_corners, padding_mode=padding_mode)

    if not return_mask:
        return out

    if len(sizes) == 3:
        valid_mask = (
            (coords[..., 0] >= -1)
            & (coords[..., 0] < 1)
            & (coords[..., 1] >= -1)
            & (coords[..., 1] < 1)
            & (coords[..., 2] >= -1)
            & (coords[..., 2] < 1)
        )
    else:
        valid_mask = (coords[..., 0] >= -1) & (coords[..., 0] < 1) & (coords[..., 1] >= -1) & (coords[..., 1] < 1)

    return out, valid_mask


def bilinear_sampler_1d(input, coords, mode="bilinear", align_corners=True, padding_mode="border", return_mask=False):

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    # if align_corners:
    #     coords = coords * torch.tensor(
    #         [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device
    #     )
    # else:
    #     coords = coords * torch.tensor([2 / size for size in reversed(sizes)], device=coords.device)

    # coords -= 1

    coords_x = coords[..., 0]
    coords_y = coords[..., 1]

    coords_x = coords_x * 0
    coords_y = coords_y * 2 / sizes[0] - 1

    coords = torch.stack([coords_x, coords_y], dim=-1)

    out = F.grid_sample(input, coords, mode=mode, align_corners=align_corners, padding_mode=padding_mode)

    if not return_mask:
        return out

    if len(sizes) == 3:
        valid_mask = (
            (coords[..., 0] >= -1)
            & (coords[..., 0] < 1)
            & (coords[..., 1] >= -1)
            & (coords[..., 1] < 1)
            & (coords[..., 2] >= -1)
            & (coords[..., 2] < 1)
        )
    else:
        valid_mask = (coords[..., 0] >= -1) & (coords[..., 0] < 1) & (coords[..., 1] >= -1) & (coords[..., 1] < 1)

    return out, valid_mask


def sample_features3d(input, coords):
    r"""Sample 1d features

    `sample_features4d(input, coords)` samples the spatial features
    :attr:`input` represented by a 4D tensor :math:`(B, C, W)`.

    The field is sampled at coordinates :attr:`coords` using bilinear
    interpolation. :attr:`coords` is assumed to be of shape :math:`(B, R,
    3)`, where each sample has the format :math:`(x_i, y_i)`. This uses the
    same convention as :func:`bilinear_sampler` with `align_corners=True`.

    The output tensor has one feature per point, and has shape :math:`(B,
    R, C)`.

    Args:
        input (Tensor): spatial features.
        coords (Tensor): points.

    Returns:
        Tensor: sampled features.
    """

    B, _, _ = input.shape

    # B R 1 -> B R 1 2
    # coords = coords.unsqueeze(2)

    # B C R 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 1, 3).view(B, -1, feats.shape[1] * feats.shape[3])  # B C R 1 -> B R C


def sample_features4d(input, coords):
    r"""Sample spatial features

    `sample_features4d(input, coords)` samples the spatial features
    :attr:`input` represented by a 4D tensor :math:`(B, C, H, W)`.

    The field is sampled at coordinates :attr:`coords` using bilinear
    interpolation. :attr:`coords` is assumed to be of shape :math:`(B, R,
    3)`, where each sample has the format :math:`(x_i, y_i)`. This uses the
    same convention as :func:`bilinear_sampler` with `align_corners=True`.

    The output tensor has one feature per point, and has shape :math:`(B,
    R, C)`.

    Args:
        input (Tensor): spatial features.
        coords (Tensor): points.

    Returns:
        Tensor: sampled features.
    """

    B, _, _, _ = input.shape

    # B R 2 -> B R 1 2
    coords = coords.unsqueeze(2)

    # B C R 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 1, 3).view(B, -1, feats.shape[1] * feats.shape[3])  # B C R 1 -> B R C


def sample_features5d(input, coords, mode="bilinear"):
    r"""Sample spatio-temporal features

    `sample_features5d(input, coords)` works in the same way as
    :func:`sample_features4d` but for spatio-temporal features and points:
    :attr:`input` is a 5D tensor :math:`(B, T, C, H, W)`, :attr:`coords` is
    a :math:`(B, R1, R2, 3)` tensor of spatio-temporal point :math:`(t_i,
    x_i, y_i)`. The output tensor has shape :math:`(B, R1, R2, C)`.

    Args:
        input (Tensor): spatio-temporal features.
        coords (Tensor): spatio-temporal points.

    Returns:
        Tensor: sampled features.
    """

    B, T, _, _, _ = input.shape

    # B T C H W -> B C T H W
    input = input.permute(0, 2, 1, 3, 4)

    # B R1 R2 3 -> B R1 R2 1 3
    coords = coords.unsqueeze(3)

    # B C R1 R2 1
    feats = bilinear_sampler(input, coords, mode=mode)

    return feats.permute(0, 2, 3, 1, 4).view(
        B, feats.shape[2], feats.shape[3], feats.shape[1]
    )  # B C R1 R2 1 -> B R1 R2 C


def bilinear_sample2d(im, x, y, return_inbounds=False):
    # x and y are each B, N
    # output is B, C, N
    if len(im.shape) == 5:
        B, N, C, H, W = list(im.shape)
    else:
        B, C, H, W = list(im.shape)
    N = list(x.shape)[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)

    # inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<W_f+0.5).float()*(y<H_f+0.5).float()

    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int64, device=x.device) * dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # im is B x C x H x W
    # move C out to last dim
    if len(im.shape) == 5:
        im_flat = (im.permute(0, 3, 4, 1, 2)).reshape(B * H * W, N, C)
        i_y0_x0 = torch.diagonal(im_flat[idx_y0_x0.long()], dim1=1, dim2=2).permute(0, 2, 1)
        i_y0_x1 = torch.diagonal(im_flat[idx_y0_x1.long()], dim1=1, dim2=2).permute(0, 2, 1)
        i_y1_x0 = torch.diagonal(im_flat[idx_y1_x0.long()], dim1=1, dim2=2).permute(0, 2, 1)
        i_y1_x1 = torch.diagonal(im_flat[idx_y1_x1.long()], dim1=1, dim2=2).permute(0, 2, 1)
    else:
        im_flat = (im.permute(0, 2, 3, 1)).reshape(B * H * W, C)
        i_y0_x0 = im_flat[idx_y0_x0.long()]
        i_y0_x1 = im_flat[idx_y0_x1.long()]
        i_y1_x0 = im_flat[idx_y1_x0.long()]
        i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    if return_inbounds:
        x_valid = (x > -0.5).byte() & (x < float(W_f - 0.5)).byte()
        y_valid = (y > -0.5).byte() & (y < float(H_f - 0.5)).byte()
        inbounds = (x_valid & y_valid).float()
        inbounds = inbounds.reshape(
            B, N
        )  # something seems wrong here for B>1; i'm getting an error here (or downstream if i put -1)
        return output, inbounds

    return output  # B, C, N


def get_grid(height, width, shape=None, dtype="torch", device="cpu", align_corners=True, normalize=True):
    H, W = height, width
    S = shape if shape else []
    if align_corners:
        x = torch.linspace(0, 1, W, device=device)
        y = torch.linspace(0, 1, H, device=device)
        if not normalize:
            x = x * (W - 1)
            y = y * (H - 1)
    else:
        x = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device)
        y = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device)
        if not normalize:
            x = x * W
            y = y * H
    x_view, y_view, exp = [1 for _ in S] + [1, -1], [1 for _ in S] + [-1, 1], S + [H, W]
    x = x.view(*x_view).expand(*exp)
    y = y.view(*y_view).expand(*exp)
    grid = torch.stack([x, y], dim=-1)
    if dtype == "numpy":
        grid = grid.numpy()
    return grid


def dense_to_sparse_tracks_3d_in_3dspace(src_points, depth, x0, y0):
    H, W = depth.shape[1:]

    valid_mask = (
        (src_points[..., 0] > x0 + 2)
        & (src_points[..., 0] < x0 + W - 2)
        & (src_points[..., 1] > y0 + 2)
        & (src_points[..., 1] < y0 + H - 2)
    )
    src_points_ = src_points[valid_mask]  # N', 3

    intr = torch.tensor(
        [
            [W * 4, 0.0, W * 4 // 2],
            [0.0, W * 4, H * 4 // 2],
            [0.0, 0.0, 1.0],
        ]
    ).to(depth)

    # h, w = height, width
    # T = tracks.size(0)

    H, W = depth.shape[-2:]

    points_3d = torch.cat([src_points_[..., :2], torch.ones_like(src_points_[..., 0:1])], dim=-1)  # N, 3

    # points = torch.stack([x,y,torch.ones_like(x)], dim=-1)
    points_3d = torch.linalg.inv(intr[None, ...]) @ points_3d.reshape(-1, 3, 1)  # (TN) 3 1

    points_3d = points_3d.reshape(-1, 3)
    points_3d *= src_points_[..., 2:3]  # N, 3

    grid_xy = get_grid(H, W, device=depth.device, normalize=False)
    grid_xy = grid_xy.view(H * W, 2)
    grid_xy[:, 0] += x0
    grid_xy[:, 1] += y0

    grid_points = torch.cat(
        [
            grid_xy,
            torch.ones_like(grid_xy[..., 0:1]),
        ],
        dim=-1,
    )  # HW x 3

    grid_points_3d = torch.linalg.inv(intr[None, ...]) @ grid_points.reshape(-1, 3, 1)  # (TN) 3 1
    grid_points_3d = grid_points_3d.reshape(-1, 3)  # T N 3
    grid_points_3d *= depth.reshape(-1).unsqueeze(-1)  # (H W) 3

    _, idx, _ = torch3d.knn_points(points_3d[None, ...], grid_points_3d[None, ...], return_nn=False)
    idx = idx[0, :, 0]  # N

    # breakpoint()
    return idx, valid_mask

    # tracks = rearrange(tracks, "i s c h w -> i s c (h w) c")
    # vis = rearrange(vis, "s h w -> s (h w)")

    # tracks_sparse = tracks[:, :, :, idx, :] # n t c
    # vis_sparse = tracks[:, idx] # n t c

    # return tracks_sparse, vis_sparse, valid_mask


def depth_to_disparity(depth, depth_min, depth_max, filter_invalid=True):
    disparity = (1.0 / depth - 1.0 / depth_max) / (1.0 / depth_min - 1.0 / depth_max)

    if filter_invalid:
        disparity[disparity < 0] = 0
        disparity[disparity > 1] = 1

    return disparity


def disparity_to_depth(disp, depth_min, depth_max, filter_invalid=True):
    # disparity = (1.0 / depth - 1.0 / depth_max) / (1.0 / depth_min - 1.0 / depth_max)

    disp_scaled = disp * (1.0 / depth_min - 1.0 / depth_max) + 1.0 / depth_max

    if isinstance(disp_scaled, torch.Tensor):
        disp_scaled = torch.clamp(disp_scaled, 1.0 / depth_max, 1.0 / depth_min)
    elif isinstance(disp_scaled, np.array):
        disp_scaled = np.clip(disp_scaled, 1.0 / depth_max, 1.0 / depth_min)

    depth = 1.0 / disp_scaled

    return depth


def disparity_to_depth_scaleshift(depth, disp):

    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
        disp = disp.cpu().numpy()

    inv_metric_videodepth = 1.0 / np.clip(depth, 1e-6, 1e6)

    ms_metric_disp = inv_metric_videodepth - np.median(inv_metric_videodepth) + 1e-8
    ms_mono_disp = disp - np.median(disp) + 1e-8

    scale = np.median(ms_metric_disp / ms_mono_disp)
    shift = np.median(inv_metric_videodepth - scale * disp)

    aligned_videodisp = scale * disp + shift

    min_thre = min(1e-6, np.quantile(aligned_videodisp, 0.01))
    # set depth values that are too small to invalid (0)
    aligned_videodisp[aligned_videodisp < min_thre] = 0.01

    videodepth = 1.0 / aligned_videodisp

    return videodepth, (scale, shift)


def disparity_to_depth_with_scaleshift(disp, scale, shift):
    aligned_disp = scale * disp + shift

    depth = 1.0 / aligned_disp

    return depth


def convert_trajs_uvd_to_trajs_3d(trajs_uv, trajs_depth, vis, video, intr=None, query_frame=0):
    device = trajs_uv.device
    H, W = video.shape[-2:]

    if intr is None:
        intr = torch.tensor(
            [
                [W, 0.0, W // 2],
                [0.0, W, H // 2],
                [0.0, 0.0, 1.0],
            ]
        ).to(device)

    trajs_uv_homo = torch.cat([trajs_uv, torch.ones_like(trajs_uv[..., 0:1])], dim=-1)  # B T N 3

    xyz = einsum(trajs_uv_homo, torch.linalg.inv(intr), "b t n j, i j -> b t n i")
    xyz = xyz * trajs_depth

    query_rgb = video[:, query_frame]  # B 3 H W

    pred_tracks2dNm = trajs_uv[:, 0].clone()  #  B N 2
    pred_tracks2dNm[..., 0] = 2 * (pred_tracks2dNm[..., 0] / W - 0.5)
    pred_tracks2dNm[..., 1] = 2 * (pred_tracks2dNm[..., 1] / H - 0.5)
    color_interp = F.grid_sample(query_rgb, pred_tracks2dNm[:, :, None, :], align_corners=True)
    color_interp = rearrange(color_interp, "b c n 1 -> b n c")

    trajs_3d_dict = {
        "coords": xyz,
        "colors": color_interp,
        "vis": vis,
    }
    return trajs_3d_dict

def convert_trajs_uvd_to_trajs_3dv2(trajs_uv, trajs_depth, vis, video, intr=None, dense_frame_e=None):
    device = trajs_uv.device
    H, W = video.shape[-2:]

    if intr is None:
        intr = torch.tensor(
            [
                [W, 0.0, W // 2],
                [0.0, W, H // 2],
                [0.0, 0.0, 1.0],
            ]
        ).to(device)

    trajs_uv_homo = torch.cat([trajs_uv, torch.ones_like(trajs_uv[..., 0:1])], dim=-1)  # B T N 3

    xyz = einsum(trajs_uv_homo, torch.linalg.inv(intr), "b t n j, i j -> b t n i")
    xyz = xyz * trajs_depth

    color_interp_all = torch.zeros((dense_frame_e.shape[0], dense_frame_e.shape[1], 3)).cuda()
    for query_frame in dense_frame_e.unique():
      query_frame = query_frame.int()
      query_rgb = video[:, query_frame]  # B 3 H W

      pred_tracks2dNm = trajs_uv[:, query_frame, dense_frame_e[0]==query_frame].clone()  #  B N 2
      pred_tracks2dNm[..., 0] = 2 * (pred_tracks2dNm[..., 0] / W - 0.5)
      pred_tracks2dNm[..., 1] = 2 * (pred_tracks2dNm[..., 1] / H - 0.5)
      color_interp = F.grid_sample(query_rgb, pred_tracks2dNm[:, :, None, :], align_corners=True)
      #print((color_interp.sum(-1)==0).sum())
      color_interp = rearrange(color_interp, "b c n 1 -> b n c")
      color_interp_all[:, dense_frame_e[0]==query_frame] = color_interp
      #color_interp_list.append(color_interp)

    #color_interp = torch.cat(color_interp_list, dim=1)

    trajs_3d_dict = {
        "coords": xyz,
        "colors": color_interp_all,
        "dense_frame_e": dense_frame_e,
        "vis": vis,
    }
    return trajs_3d_dict
    