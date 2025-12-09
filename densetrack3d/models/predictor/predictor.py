# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from densetrack3d.models.model_utils import bilinear_sample2d, get_points_on_a_grid, smart_cat, convert_trajs_uvd_to_trajs_3d


class Predictor3D(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.interp_shape = model.model_resolution
        self.model = model
        self.n_iters = 6
        self.support_grid_size = 6

    @torch.inference_mode()
    def forward(
        self,
        video,  # (B, T, 3, H, W)
        videodepth,
        queries: torch.Tensor = None,
        segm_mask: torch.Tensor = None,  # Segmentation mask of shape (B, 1, H, W)
        grid_size: int = 0,
        grid_query_frame: int = 0,  # only for dense and regular grid tracks
        backward_tracking: bool = False,
        predefined_intrs: torch.Tensor = None,
    ):
        if queries is None and grid_size == 0:
            out = self._compute_dense_tracks(
                video,
                videodepth,
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
                predefined_intrs=predefined_intrs,
            )
        else:
            out = self._compute_sparse_tracks(
                video,
                videodepth,
                queries,
                segm_mask,
                grid_size,
                add_support_grid=(grid_size == 0 or segm_mask is not None),
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
                predefined_intrs=predefined_intrs
            )

        return out

    def _compute_dense_tracks(
        self, 
        video, 
        videodepth, 
        grid_query_frame, 
        grid_size=80, 
        backward_tracking=False,
        predefined_intrs=None
    ):
        *_, H, W = video.shape
        grid_step = W // grid_size
        grid_width = W // grid_step
        grid_height = H // grid_step
        tracks = visibilities = None
        grid_pts = torch.zeros((1, grid_width * grid_height, 3)).to(video.device)
        grid_pts[0, :, 0] = grid_query_frame
        for offset in range(grid_step * grid_step):
            print(f"step {offset} / {grid_step * grid_step}")
            ox = offset % grid_step
            oy = offset // grid_step
            grid_pts[0, :, 1] = torch.arange(grid_width).repeat(grid_height) * grid_step + ox
            grid_pts[0, :, 2] = torch.arange(grid_height).repeat_interleave(grid_width) * grid_step + oy
            tracks_step, visibilities_step = self._compute_sparse_tracks(
                video=video,
                videodepth=videodepth,
                queries=grid_pts,
                backward_tracking=backward_tracking,
            )
            tracks = smart_cat(tracks, tracks_step, dim=2)
            visibilities = smart_cat(visibilities, visibilities_step, dim=2)

        return tracks, visibilities

    def _compute_sparse_tracks(
        self,
        video,
        videodepth,
        queries,
        segm_mask=None,
        grid_size=0,
        add_support_grid=False,
        grid_query_frame=0,
        backward_tracking=False,
        scale_to_origin=True,
        predefined_intrs=None
    ):
        B, T, C, H, W = video.shape

        ori_video = video.clone()

        video = F.interpolate(
            video.flatten(0, 1), tuple(self.interp_shape), mode="bilinear", align_corners=True
        ).reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
        videodepth = F.interpolate(videodepth.flatten(0, 1), tuple(self.interp_shape), mode="nearest").reshape(
            B, T, 1, self.interp_shape[0], self.interp_shape[1]
        )

        if queries is not None:
            B, N, D = queries.shape
            assert D == 3
            queries = queries.clone()
            queries[:, :, 1:] *= queries.new_tensor(
                [
                    (self.interp_shape[1] - 1) / (W - 1),
                    (self.interp_shape[0] - 1) / (H - 1),
                ]
            )

        elif (isinstance(grid_size, int) and grid_size > 0) or grid_size[0] > 0:
            grid_pts = get_points_on_a_grid(grid_size, self.interp_shape, device=video.device)
            if segm_mask is not None:
                segm_mask = F.interpolate(segm_mask, tuple(self.interp_shape), mode="nearest")
                point_mask = segm_mask[0, 0][
                    (grid_pts[0, :, 1]).round().long().cpu(),
                    (grid_pts[0, :, 0]).round().long().cpu(),
                ].bool()
                grid_pts = grid_pts[:, point_mask]

            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                dim=2,
            ).repeat(B, 1, 1)

        if add_support_grid:
            grid_pts = get_points_on_a_grid(self.support_grid_size, self.interp_shape, device=video.device)
            grid_pts = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)
            grid_pts = grid_pts.repeat(B, 1, 1)
            queries = torch.cat([queries, grid_pts], dim=1)

        # depths = videodepth
        # rgbds = torch.cat([video, depths], dim=2)
        # get the 3D queries # B, N, 3
        depth_interp = []
        for i in range(queries.shape[1]):
            depth_interp_i = bilinear_sample2d(
                videodepth[0, queries[:, i : i + 1, 0].long()], queries[:, i : i + 1, 1], queries[:, i : i + 1, 2]
            )
            depth_interp.append(depth_interp_i)

        depth_interp = torch.cat(depth_interp, dim=1)
        queries = smart_cat(queries, depth_interp, dim=-1)

        sparse_predictions, dense_predictions, _ = self.model(
            video=video, 
            videodepth=videodepth, 
            sparse_queries=queries, 
            iters=self.n_iters, 
            use_dense=False
        )

        traj_e, d_e, vis_e, conf_e = sparse_predictions["coords"], sparse_predictions["coord_depths"], sparse_predictions["vis"], sparse_predictions["conf"]

        if backward_tracking:
            traj_e, d_e, vis_e, conf_e = self._compute_backward_tracks(video, videodepth, queries, traj_e, d_e, vis_e, conf_e)
            if add_support_grid:
                queries[:, -self.support_grid_size**2 :, 0] = T - 1

        if add_support_grid:
            traj_e = traj_e[:, :, : -self.support_grid_size**2]
            d_e = d_e[:, :, : -self.support_grid_size**2]
            vis_e = vis_e[:, :, : -self.support_grid_size**2]
            conf_e = conf_e[:, :, : -self.support_grid_size**2]

        thr = 0.9
        vis_e = vis_e > thr

        # correct query-point predictions
        # see https://github.com/facebookresearch/co-tracker/issues/28

        # TODO: batchify
        for i in range(len(queries)):
            queries_t = queries[i, : traj_e.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))

            # overwrite the predictions with the query points
            traj_e[i, queries_t, arange] = queries[i, : traj_e.size(2), 1:3]
            d_e[i, queries_t, arange] = queries[i, : d_e.size(2), 3:4]
            vis_e[i, queries_t, arange] = True

        if scale_to_origin:
            traj_e[..., 0] *= (W - 1) / float(self.interp_shape[1] - 1)
            traj_e[..., 1] *= (H - 1) / float(self.interp_shape[0] - 1)

        if scale_to_origin:
            sparse_trajs_3d_dict = convert_trajs_uvd_to_trajs_3d(
                traj_e,
                d_e,
                vis_e,
                ori_video,
                query_frame=grid_query_frame,
                intr=predefined_intrs,
            )
        else:
            sparse_trajs_3d_dict = convert_trajs_uvd_to_trajs_3d(
                traj_e,
                d_e,
                vis_e,
                video, 
                query_frame=grid_query_frame, 
                intr=predefined_intrs
            )

        out = {
            "trajs_uv": traj_e, 
            "trajs_depth": d_e, 
            "vis": vis_e, 
            "conf": conf_e,
            "trajs_3d_dict": sparse_trajs_3d_dict,
        }

        return out

    def _compute_backward_tracks(self, video, videodepth, queries, traj_e, d_e, vis_e, conf_e):
        inv_video = video.flip(1).clone()
        inv_videodepth = videodepth.flip(1).clone()
        inv_queries = queries.clone()
        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

        sparse_predictions, dense_predictions, _ = self.model(
            video=inv_video,
            videodepth=inv_videodepth,
            sparse_queries=inv_queries,
            iters=self.n_iters,
            use_dense=False,
        )


        inv_trajs_e, inv_d_e, inv_vis_e, inv_conf_e = (
            sparse_predictions["coords"].flip(1),
            sparse_predictions["coord_depths"].flip(1),
            sparse_predictions["vis"].flip(1),
            sparse_predictions["conf"].flip(1),

        )

        arange = torch.arange(video.shape[1], device=queries.device)[None, :, None]

        mask = (arange < queries[:, None, :, 0]).unsqueeze(-1).repeat(1, 1, 1, 2)

        traj_e[mask] = inv_trajs_e[mask]
        d_e[mask[:, :, :, 0]] = inv_d_e[mask[:, :, :, 0]]
        vis_e[mask[:, :, :, 0]] = inv_vis_e[mask[:, :, :, 0]]
        conf_e[mask[:, :, :, 0]] = inv_conf_e[mask[:, :, :, 0]]

        return traj_e, d_e, vis_e, conf_e
