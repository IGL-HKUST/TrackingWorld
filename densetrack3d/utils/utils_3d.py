from typing import Optional, Tuple, Union

import numpy as np
import pytorch3d.ops as torch3d
import torch
import torch.nn.functional as F
from einops import rearrange


def reproject(video, pred_tracks_3d, color=None):
    H, W = video.shape[-2:]
    T, N = pred_tracks_3d.shape[:2]
    intr = torch.tensor([[W, 0.0, W // 2], [0.0, W, H // 2], [0.0, 0.0, 1.0]]).to(pred_tracks_3d.device)
    xyztVis = pred_tracks_3d.clone()
    xyztVis[..., 2] = 1.0

    xyztVis = torch.linalg.inv(intr[None, ...]) @ xyztVis.reshape(-1, 3, 1)  # (TN) 3 1
    xyztVis = xyztVis.reshape(T, -1, 3)  # T N 3
    xyztVis[..., 2] *= pred_tracks_3d[..., 2]

    if color is None:
        pred_tracks2d = pred_tracks_3d[:, :, :2]
        pred_tracks2dNm = pred_tracks2d.clone()
        pred_tracks2dNm[..., 0] = 2 * (pred_tracks2dNm[..., 0] / W - 0.5)
        pred_tracks2dNm[..., 1] = 2 * (pred_tracks2dNm[..., 1] / H - 0.5)
        color_interp = F.grid_sample(video, pred_tracks2dNm[:, :, None, :], align_corners=True)
        color_interp = color_interp[:, :, :, 0].permute(0, 2, 1)
    else:
        color = torch.tensor(color).to(pred_tracks_3d.device)
        color_interp = color[None, None].repeat(T, N, 1)
    colored_pts = torch.cat([xyztVis, color_interp], dim=-1)

    colored_pts = colored_pts.cpu().numpy()

    return colored_pts
