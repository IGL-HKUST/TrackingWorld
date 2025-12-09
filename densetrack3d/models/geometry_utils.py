from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from densetrack3d.models.model_utils import get_grid


EPS = 1e-6


def _reproject_2d3d(trajs_uvd, intrs):

    B, N = trajs_uvd.shape[:2]
    # intrs = sample.intrs
    fx, fy, cx, cy = intrs[:, 0, 0], intrs[:, 1, 1], intrs[:, 0, 2], intrs[:, 1, 2]

    trajs_3d = torch.zeros((B, N, 3), device=trajs_uvd.device)
    trajs_3d[..., 0] = trajs_uvd[..., 2] * (trajs_uvd[..., 0] - cx[..., None]) / fx[..., None]
    trajs_3d[..., 1] = trajs_uvd[..., 2] * (trajs_uvd[..., 1] - cy[..., None]) / fy[..., None]
    trajs_3d[..., 2] = trajs_uvd[..., 2]

    return trajs_3d


def _project_3d2d(trajs_3d, intrs):

    B, N = trajs_3d.shape[:2]
    # intrs = sample.intrs
    fx, fy, cx, cy = intrs[:, 0, 0], intrs[:, 1, 1], intrs[:, 0, 2], intrs[:, 1, 2]

    trajs_uvd = torch.zeros((B, N, 3), device=trajs_3d.device)
    trajs_uvd[..., 0] = trajs_3d[..., 0] * fx[..., None] / trajs_3d[..., 2] + cx[..., None]
    trajs_uvd[..., 1] = trajs_3d[..., 1] * fy[..., None] / trajs_3d[..., 2] + cy[..., None]
    trajs_uvd[..., 2] = trajs_3d[..., 2]

    return trajs_uvd


def reproject_2d3d(trajs_uvd, intrs):

    B, T, N = trajs_uvd.shape[:3]

    trajs_3d = _reproject_2d3d(trajs_uvd.reshape(-1, N, 3), intrs.reshape(-1, 3, 3))
    trajs_3d = rearrange(trajs_3d, "(B T) N C -> B T N C", T=T)

    return trajs_3d


def project_3d2d(trajs_3d, intrs):

    B, T, N = trajs_3d.shape[:3]

    trajs_uvd = _project_3d2d(trajs_3d.reshape(-1, N, 3), intrs.reshape(-1, 3, 3))
    trajs_uvd = rearrange(trajs_uvd, "(B T) N C -> B T N C", T=T)

    return trajs_uvd


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


def knn_mask(depths, x0y0, k=3):
    B, _, H, W = depths.shape

    depths_down = F.interpolate(depths, scale_factor=0.25, mode="nearest")
    H_down, W_down = depths_down.shape[-2:]
    x0, y0 = x0y0

    intr = torch.tensor(
        [
            [W, 0.0, W // 2],
            [0.0, W, H // 2],
            [0.0, 0.0, 1.0],
        ]
    ).to(depths)
    dense2d_grid = get_grid(H, W, normalize=False, device=depths.device)  # H W 2

    intr_down = torch.tensor(
        [
            [W_down, 0.0, W_down // 2],
            [0.0, W_down, H_down // 2],
            [0.0, 0.0, 1.0],
        ]
    ).to(depths)
    dense2d_grid_down = get_grid(H_down, W_down, normalize=False, device=depths.device)  # H W 2

    nn_inds_arr, nn_weights_arr = [], []
    for b in range(B):
        x0_, y0_ = x0[b], y0[b]

        dense2d_grid_ = dense2d_grid[y0_ * 4 : y0_ * 4 + H_down * 4, x0_ * 4 : x0_ * 4 + W_down * 4]
        dense2d_grid_ = dense2d_grid_.reshape(-1, 2)

        depth_ = depths[b, 0, y0_ * 4 : y0_ * 4 + H_down * 4, x0_ * 4 : x0_ * 4 + W_down * 4].reshape(-1, 1)

        uvd = torch.cat([dense2d_grid_, depth_], dim=-1)  # HW 3

        points_3d = _reproject_2d3d(uvd[None], intr[None])
        # points_3d = points_3d.squeeze(0) # HW 3

        dense2d_grid_down_ = dense2d_grid_down[y0_ : y0_ + H_down, x0_ : x0_ + W_down]
        dense2d_grid_down_ = dense2d_grid_down_.reshape(-1, 2)

        depth_down_ = depths_down[b, 0, y0_ : y0_ + H_down, x0_ : x0_ + W_down].reshape(-1, 1)

        uvd_down = torch.cat([dense2d_grid_down_, depth_down_], dim=-1)  # HW 3

        points_down_3d = _reproject_2d3d(uvd_down[None], intr_down[None])
        # points_down_3d = points_down_3d.squeeze(0) # HW 3

        nn_dist, nn_inds, _ = torch3d.knn_points(points_3d, points_down_3d, K=k)
        nn_dist, nn_inds = nn_dist.squeeze(0), nn_inds.squeeze(0)  # HW K, HW K

        # breakpoint()
        nn_valid_mask = nn_dist < 0.01  # HW K

        nn_weights = 1.0 / (nn_dist + 1e-6)  # HW K
        nn_weights[~nn_valid_mask] = -torch.finfo(nn_weights.dtype).max
        nn_weights = F.softmax(nn_weights, dim=-1)  # HW K

        # entry_novalid = (nn_valid_mask.sum(-1) == 0) # HW

        nn_inds_arr.append(nn_inds)
        nn_weights_arr.append(nn_weights)

    nn_inds = torch.stack(nn_inds_arr, dim=0)  # B HW K
    nn_weights = torch.stack(nn_weights_arr, dim=0)  # B HW K

    return nn_inds, nn_weights


def get_2d_from_3d_trajs(tracks, H, W):
    # video2d = video[0] # T C H W

    T, N = tracks.shape[:2]

    # H1, W1 = video[0].shape[-2:]
    intr = torch.tensor(
        [
            [W, 0.0, W // 2],
            [0.0, W, H // 2],
            [0.0, 0.0, 1.0],
        ]
    ).to(tracks)

    xyz = tracks[..., :3]

    uvd = intr[None, ...] @ xyz.reshape(-1, 3, 1)  # (TN) 3 1
    uvd = uvd.reshape(T, -1, 3)  # T N 3
    uvd[..., :2] /= uvd[..., 2:]

    # breakpoint()
    trajs_2d = torch.cat([uvd, tracks[..., 3:]], dim=-1)

    return trajs_2d


def least_square_align(depth, disp, query_frame=0, return_align_scalar=False, filter_invalid=True):

    if isinstance(depth, np.ndarray):
        is_numpy = True
        depth = torch.from_numpy(depth).float()
        disp = torch.from_numpy(disp).float()
    else:
        is_numpy = False

    if len(depth.shape) == 4:
        if depth.shape[1] == 1:
            depth = depth.squeeze(1)
        elif depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        else:
            raise ValueError("Invalid depth shape")

    if len(disp.shape) == 4:
        if disp.shape[1] == 1:
            disp = disp.squeeze(1)
        elif disp.shape[-1] == 1:
            disp = disp.squeeze(-1)
        else:
            raise ValueError("Invalid depth shape")

    T, H, W = disp.shape

    if depth.shape[1] != H or depth.shape[2] != W:
        depth = F.interpolate(depth.float().unsqueeze(1), (H, W), mode="nearest").squeeze(1)

    inv_depth = 1 / torch.clamp(depth, 1e-6, 1e6)

    # NOTE only align first frame
    x = disp[query_frame].clone().reshape(-1)
    y = inv_depth[query_frame].clone().reshape(-1)

    if filter_invalid:
        valid_mask = (depth[query_frame] > 1e-1) & (depth[query_frame] < 100)
        valid_mask = valid_mask.reshape(-1)
        x = x[valid_mask]
        y = y[valid_mask]

    A = torch.stack([x, torch.ones_like(x)], dim=-1)  # N, 2
    s, t = torch.linalg.lstsq(A, y, rcond=None)[0]

    aligned_disp = disp * s + t
    aligned_depth = 1 / torch.clamp(aligned_disp, 1e-6, 1e6)
    aligned_depth = aligned_depth.reshape(T, H, W)

    if is_numpy:
        aligned_depth = aligned_depth.numpy()
        s, t = s.numpy(), t.numpy()

    if return_align_scalar:
        return aligned_depth, s, t
    else:
        return aligned_depth
