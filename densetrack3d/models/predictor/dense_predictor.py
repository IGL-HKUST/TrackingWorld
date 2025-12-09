# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional, Tuple, List, Dict, Union

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label
from einops import rearrange, repeat
from tqdm import tqdm

from densetrack3d.models.model_utils import (
    bilinear_sampler,
    get_points_on_a_grid,
    convert_trajs_uvd_to_trajs_3d,
    convert_trajs_uvd_to_trajs_3dv2,
    get_grid,
)

# --- Utility Functions ---

def remove_noise_from_mask(binary_mask: torch.Tensor, min_size: int = 10) -> torch.Tensor:
    """
    Remove small noise points from a binary mask using connected component analysis.
    """
    binary_mask = (binary_mask > 0).float()
    device = binary_mask.device

    if binary_mask.dim() == 3:
        cleaned_mask = torch.zeros_like(binary_mask)
        for c in range(binary_mask.shape[0]):
            cleaned_mask[c] = _clean_single_mask(binary_mask[c], min_size)
        return cleaned_mask.to(device)
    elif binary_mask.dim() == 2:
        return _clean_single_mask(binary_mask, min_size).to(device)
    else:
        raise ValueError("Input mask must be 2D [H, W] or 3D [C, H, W]")


def _clean_single_mask(mask: torch.Tensor, min_size: int) -> torch.Tensor:
    """Helper function to clean a single 2D mask using scipy."""
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    
    labeled_array, num_features = label(mask_np)
    if num_features == 0:
        return mask

    component_sizes = np.bincount(labeled_array.ravel())
    keep_components = component_sizes >= min_size
    keep_components[0] = 0  # Always ignore background
    
    keep_mask = keep_components[labeled_array]
    return torch.from_numpy(keep_mask.astype(np.float32)).to(mask.device)


def pad_to_T(tensor: torch.Tensor, T: int) -> torch.Tensor:
    """Pads the tensor along the time dimension (dim 1) to length T."""
    B, N, *dims = tensor.shape
    if N >= T:
        return tensor[:, :T, ...]
    
    # Calculate padding needed
    pad_size = T - N
    # Construct target shape for zero padding
    target_shape = [B, pad_size] + list(dims)
    zeros = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    
    return torch.cat([zeros, tensor], dim=1)


def generate_extra_queries(
    trajs_uv: torch.Tensor,
    trajs_vis: torch.Tensor,
    videodepth: torch.Tensor,
    img_shape: Tuple[int, int],
    frame_offset_idx: int = 0
) -> torch.Tensor:
    """
    Generate uncovered region coordinates for frame pairs to spawn new tracks.
    
    Args:
        trajs_uv: [B, 2, N, 2] Tracking points.
        trajs_vis: [B, 2, N] Visibility mask.
        videodepth: [B, T, 1, H, W] Depth maps.
        img_shape: (H, W) Target resolution.
        frame_offset_idx: Offset to add to frame indices.
    """
    H, W = img_shape
    num_pairs = trajs_uv.shape[0]
    device = trajs_uv.device
    all_extra_queries = []

    # Extract second frame data
    uv = trajs_uv[:, 1, :, :]  # [B, N, 2]
    vis = trajs_vis[:, 1, :]   # [B, N]

    for pair_idx in range(num_pairs):
        # 1. Filter valid visible points
        valid_mask_vis = vis[pair_idx] > 0.6
        valid_uv = torch.round(uv[pair_idx][valid_mask_vis]).long()

        # 2. Check bounds
        in_bounds = (valid_uv[:, 0] >= 0) & (valid_uv[:, 0] < W) & \
                    (valid_uv[:, 1] >= 0) & (valid_uv[:, 1] < H)
        valid_uv = valid_uv[in_bounds]

        # 3. Create occupancy mask
        mask = torch.ones(H * W, dtype=torch.bool, device=device)
        if valid_uv.shape[0] > 0:
            indices = valid_uv[:, 1] * W + valid_uv[:, 0]
            mask.index_fill_(0, indices, False)
        
        # 4. Clean noise
        mask_2d = mask.reshape(H, W)
        mask_cleaned = remove_noise_from_mask(mask_2d, min_size=40).flatten().bool()

        # 5. Extract uncovered coordinates
        uncovered_indices = torch.nonzero(mask_cleaned, as_tuple=False).squeeze(-1)
        uncovered_coords = torch.stack([uncovered_indices % W, uncovered_indices // W], dim=-1)

        # 6. Prepare query tensor: [Frame_Idx, X, Y, Depth]
        # Frame index
        frame_indices = torch.full((uncovered_coords.shape[0], 1), pair_idx + 1 + frame_offset_idx, 
                                   dtype=torch.long, device=device)
        queries_xy = torch.cat([frame_indices, uncovered_coords], dim=1) # [M, 3]

        # Sample depth
        queries_d = bilinear_sampler(
            videodepth[pair_idx : pair_idx + 1, 1], 
            rearrange(queries_xy[..., 1:3], "n c -> () () n c"), 
            mode="nearest"
        )
        queries_d = rearrange(queries_d, "b c m n -> b (m n) c")[0]
        
        # Combine [Frame, X, Y, Depth]
        all_extra_queries.append(torch.cat([queries_xy, queries_d], dim=-1))

    if all_extra_queries:
        return torch.cat(all_extra_queries, dim=0)
    return torch.empty((0, 3), dtype=torch.long, device=device)


# --- Main Class ---

class DensePredictor3D(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.interp_shape = model.model_resolution # e.g. (384, 512)
        self.n_iters = 6

    def _rescale_trajectories(self, traj: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Rescales coordinates from model resolution back to original resolution."""
        traj = traj.clone()
        traj[..., 0] *= (W - 1) / float(self.interp_shape[1] - 1)
        traj[..., 1] *= (H - 1) / float(self.interp_shape[0] - 1)
        return traj

    @torch.inference_mode()
    def forward_epef(
        self,
        video: torch.Tensor,
        videodepth: torch.Tensor,
        grid_query_frame: int = 0,
        scale_input: bool = True,
        scale_to_origin: bool = True,
        use_efficient_global_attn: bool = True,
        predefined_intrs: Optional[torch.Tensor] = None,
        use_cotracker: bool = False,
        clip_size: int = 2,
        save_dir: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        
        B, T, C, H, W = video.shape
        device = video.device
        src_step = grid_query_frame
        ori_video = video.clone()

        # 1. Preprocessing
        if scale_input:
            video = F.interpolate(
                video.flatten(0, 1), tuple(self.interp_shape), mode="bilinear", align_corners=True
            ).reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
            videodepth = F.interpolate(videodepth.flatten(0, 1), tuple(self.interp_shape), mode="nearest").reshape(
                B, T, 1, self.interp_shape[0], self.interp_shape[1]
            )

        # 2. Initial Queries
        sparse_queries_prior = None
        if use_efficient_global_attn:
            sparse_xy = get_points_on_a_grid((36, 48), video.shape[3:]).long().float()
            sparse_xy = torch.cat([src_step * torch.ones_like(sparse_xy[:, :, :1]), sparse_xy], dim=2).to(
                device
            ).repeat(B, 1, 1)  # B, N, C
            sparse_d = bilinear_sampler(
                videodepth[:, src_step], rearrange(sparse_xy[..., 1:3], "b n c -> b () n c"), mode="nearest"
            )
            sparse_d = rearrange(sparse_d, "b c m n -> b (m n) c")
            sparse_queries_prior = torch.cat([sparse_xy, sparse_d], dim=2)  #

        # 3. CoTracker (Optional)
        cotracker_results = None
        if use_cotracker:
            self.model.set_upsample_factor(8)
            sparse_xy_cotracker = get_grid(48, 64, normalize=False).reshape(-1, 2).unsqueeze(0) * 8
            sparse_xy_cotracker = torch.cat([src_step * torch.ones_like(sparse_xy_cotracker[:, :, :1]), sparse_xy_cotracker], dim=2).to(device).repeat(B, 1, 1)  
            self.model.register_cotracker()
            video_chunk = video[:, src_step:]
            self.model.cotracker(video_chunk=video_chunk, is_first_step=True, queries=sparse_xy_cotracker)
            for ind in range(0, video_chunk.shape[1] - self.model.cotracker.step, self.model.cotracker.step):
                pred_tracks, pred_visibility, pred_conf = self.model.cotracker(
                    video_chunk=video_chunk[:, ind : ind + self.model.cotracker.step * 2]
                )
            cotracker_results = (pred_tracks/4, pred_visibility, pred_conf)

        # 4. Base Forward Pass
        _, dense_preds, _, _, fmaps_pyramid = self.model.forward_epef(
            video=video[:, src_step:],
            videodepth=videodepth[:, src_step:],
            sparse_queries=sparse_queries_prior,
            iters=self.n_iters,
            use_efficient_global_attn=use_efficient_global_attn,
            use_cotracker=use_cotracker,
            cotracker_results=cotracker_results
        )

        # Initialize aggregators
        dense_traj_e = rearrange(dense_preds["coords"], "b t c h w -> b t (h w) c")
        dense_traj_d_e = rearrange(dense_preds["coord_depths"], "b t c h w -> b t (h w) c")
        dense_vis_e = rearrange(dense_preds["vis"], "b t h w -> b t (h w)")
        dense_conf_e = rearrange(dense_preds.get("conf", dense_preds["vis"]), "b t h w -> b t (h w)")
        dense_frame_e = torch.zeros_like(dense_vis_e)[:, 0] # Track source frame

        # 5. Sliding Window Extension
        self.model.set_upsample_factor(8)
        
        for idx in tqdm(range(clip_size, T - 1, clip_size), desc="EPEF Extension"):
            # Generate queries for new uncovered regions
            sparse_queries = generate_extra_queries(
                dense_traj_e[:, idx-1:], dense_vis_e[:, idx-1:], videodepth[:, idx-1:], self.interp_shape, -1
            )
            
            # Prepare priors for the new window
            if use_efficient_global_attn:
                # Re-generate grid for current step
                sparse_d = bilinear_sampler(
                videodepth[:, idx], rearrange(sparse_xy[..., 1:3], "b n c -> b () n c"), mode="nearest"
                )
                sparse_d = rearrange(sparse_d, "b c m n -> b (m n) c")
                sparse_queries_prior = torch.cat([sparse_xy, sparse_d], dim=2) 
                
                if use_cotracker:
                    video_chunk = video[:, idx:]
                    self.model.cotracker(video_chunk=video_chunk, is_first_step=True, queries=sparse_xy_cotracker)
                    for ind in range(0, video_chunk.shape[1] - self.model.cotracker.step, self.model.cotracker.step):
                        pred_tracks, pred_visibility, pred_conf = self.model.cotracker(
                            video_chunk=video_chunk[:, ind : ind + self.model.cotracker.step * 2]
                        )
                    cotracker_results = (pred_tracks / 4, pred_visibility, pred_conf)

            # Forward pass on window
            _, dense_preds_window, _, selected_indices, _ = self.model.forward_epef(
                video=video[:, idx:],
                videodepth=videodepth[:, idx:],
                sparse_queries=sparse_queries_prior,
                iters=self.n_iters,
                use_efficient_global_attn=use_efficient_global_attn,
                init_features=[f[:, idx:] for f in fmaps_pyramid],
                use_cotracker=use_cotracker,
                cotracker_results=cotracker_results
            )

            # Extract specific sparse tracks based on queries
            queries_idx = sparse_queries.long()
            qx, qy = queries_idx[:, 1], queries_idx[:, 2]
            
            # Sanity check
            if (qx >= self.interp_shape[1]).any() or (qy >= self.interp_shape[0]).any():
                raise ValueError("Generated queries out of bounds")

            # Extract predictions at query locations
            sparse_traj_e = dense_preds_window["coords"][:, :, :, qy, qx].transpose(-1, -2).clone()
            sparse_traj_d_e = dense_preds_window["coord_depths"][:, :, :, qy, qx].transpose(-1, -2).clone()
            sparse_vis_e = dense_preds_window["vis"][:, :, qy, qx].clone()
            sparse_conf_e = dense_preds_window.get("conf", dense_preds_window["vis"])[:, :, qy, qx].clone()
            
            # Initialize the start of the track with query position
            sparse_frame_e = torch.zeros_like(sparse_vis_e)[:, 0].clone()
            sparse_traj_e[0, sparse_frame_e[0].int(), torch.arange(sparse_vis_e.shape[-1], device=device)] = sparse_queries[:, 1:3]

            # Pad to full length T
            sparse_traj_e = pad_to_T(sparse_traj_e, T)
            sparse_traj_d_e = pad_to_T(sparse_traj_d_e, T)
            sparse_vis_e = pad_to_T(sparse_vis_e, T)
            sparse_conf_e = pad_to_T(sparse_conf_e, T)
            sparse_frame_e[:] = idx
            
            # Concatenate with main buffers
            # Note: The logic for sparse_frame_e concatenation in original code was slightly tricky (expanding dim 1).
            # Here we follow the pattern of appending new points to the list of tracks.
            dense_traj_e = torch.cat([dense_traj_e, sparse_traj_e], dim=2)
            dense_traj_d_e = torch.cat([dense_traj_d_e, sparse_traj_d_e], dim=2)
            dense_vis_e = torch.cat([dense_vis_e, sparse_vis_e], dim=2)
            dense_conf_e = torch.cat([dense_conf_e, sparse_conf_e], dim=2)
            dense_frame_e = torch.cat([dense_frame_e, sparse_frame_e], dim=1)

        # 6. Post-processing & 3D Conversion
        if scale_to_origin:
            dense_traj_e = self._rescale_trajectories(dense_traj_e, H, W)

        # Threshold visibility and force query point to be visible
        dense_vis_e = dense_vis_e > 0.6
        # Mark the start frame of each track as visible
        point_indices = torch.arange(dense_vis_e.shape[-1], device=device)
        frame_indices = dense_frame_e[0].long().clamp(max=T-1) # Safety clamp
        dense_vis_e[:, frame_indices, point_indices] = True

        # Convert to 3D
        convert_func = convert_trajs_uvd_to_trajs_3dv2
        dense_trajs_3d_dict = convert_func(
            dense_traj_e,
            dense_traj_d_e,
            dense_vis_e,
            ori_video if scale_to_origin else video,
            dense_frame_e=dense_frame_e,
            intr=predefined_intrs,
        )

        return {
            "trajs_uv": dense_traj_e,
            "trajs_depth": dense_traj_d_e,
            "vis": dense_vis_e,
            "trajs_3d_dict": dense_trajs_3d_dict,
            "conf": dense_conf_e,
            "dense_reso": self.interp_shape,
            "dense_frame_e": dense_frame_e
        }