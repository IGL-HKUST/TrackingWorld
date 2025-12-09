# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from densetrack3d.models.model_utils import bilinear_sampler, get_points_on_a_grid, convert_trajs_uvd_to_trajs_3d
from einops import einsum, rearrange, repeat


class DensePredictor2D(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.interp_shape = model.model_resolution
        self.model = model
        self.n_iters = 6

    @torch.inference_mode()
    def forward(
        self,
        video,  # (B, T, 3, H, W)
        grid_query_frame: int = 0,  # only for dense and regular grid tracks
        scale_input: bool = True,
        scale_to_origin: bool = True,
        use_efficient_global_attn: bool = True,
    ):
        B, T, C, H, W = video.shape
        device = video.device
        src_step = grid_query_frame

        ori_video = video.clone()

        if scale_input:
            video = F.interpolate(
                video.flatten(0, 1), tuple(self.interp_shape), mode="bilinear", align_corners=True
            ).reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])


        if use_efficient_global_attn:
            sparse_xy = get_points_on_a_grid((36, 48), video.shape[3:]).long().float()
            sparse_xy = torch.cat([src_step * torch.ones_like(sparse_xy[:, :, :1]), sparse_xy], dim=2).to(
                device
            )  # B, N, C
            sparse_queries = sparse_xy
        else:
            sparse_queries = None

        sparse_predictions, dense_predictions, _ = self.model(
            video=video[:, src_step:],
            sparse_queries=sparse_queries,
            iters=self.n_iters,
            use_efficient_global_attn=use_efficient_global_attn,
        )

        dense_traj_e, dense_vis_e = (
            dense_predictions["coords"],
            dense_predictions["vis"],
        )

        if "conf" not in dense_predictions.keys():
            dense_conf_e = dense_vis_e.clone()
        else:
            dense_conf_e = dense_predictions["conf"]

        if scale_to_origin:
            dense_traj_e[:, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
            dense_traj_e[:, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)

        dense_traj_e = rearrange(dense_traj_e, "b t c h w -> b t (h w) c")
        dense_vis_e = rearrange(dense_vis_e, "b t h w -> b t (h w)")
        dense_conf_e = rearrange(dense_conf_e, "b t h w -> b t (h w)")

        dense_vis_e = dense_vis_e > 0.8


        out = {
            "trajs_uv": dense_traj_e,
            "vis": dense_vis_e,
            "conf": dense_conf_e,
            "dense_reso": self.interp_shape,
        }

        return out
