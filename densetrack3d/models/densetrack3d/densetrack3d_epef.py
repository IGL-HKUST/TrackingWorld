
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float
from torch import Tensor

from densetrack3d.models.densetrack3d.blocks import BasicEncoder
from densetrack3d.models.densetrack3d.corr4d_blocks import Corr4DMLP
from densetrack3d.models.densetrack3d.update_transformer import EfficientUpdateFormer
from densetrack3d.models.densetrack3d.upsample_transformer import UpsampleTransformerAlibi
from densetrack3d.models.embeddings import get_1d_sincos_pos_embed_from_grid, get_2d_embedding, get_2d_sincos_pos_embed
from densetrack3d.models.model_utils import (
    bilinear_sampler,
    get_grid,
    sample_features4d,
    sample_features5d,
    smart_cat,
)


VideoType = Float[Tensor, "b t c h w"]

torch.manual_seed(0)

def sigmoid_inverse(y):
    y = torch.clamp(y, 1e-6, 1 - 1e-6)  # 避免除以 0
    return torch.log(y / (1 - y))

class DenseTrack3D(nn.Module):
    def __init__(
        self,
        window_len=8,
        stride=4,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        only_learnup=False,
        upsample_factor=4
    ):
        super().__init__()
        self.window_len_copy = window_len
        self.window_len = window_len
        self.stride = stride
        self.hidden_dim = 256
        self.latent_dim = 128
        self.upsample_factor = upsample_factor
        self.add_space_attn = add_space_attn
        self.fnet = BasicEncoder(
            input_dim=3, 
            output_dim=self.latent_dim,
            stride=self.stride,
        )
        self.num_virtual_tracks = num_virtual_tracks
        self.model_resolution = model_resolution
        self.input_dim = 1032
        self.updateformer = EfficientUpdateFormer(
            num_blocks=6,
            input_dim=self.input_dim,
            hidden_size=384,
            output_dim=self.latent_dim + 3,
            mlp_ratio=4.0,
            add_space_attn=add_space_attn,
            num_virtual_tracks=num_virtual_tracks,
            flash=False,
        )

        time_grid = torch.linspace(0, window_len - 1, window_len).reshape(1, window_len, 1)

        self.register_buffer("time_emb", get_1d_sincos_pos_embed_from_grid(self.input_dim, time_grid[0]))

        self.register_buffer(
            "pos_emb",
            get_2d_sincos_pos_embed(
                embed_dim=self.input_dim,
                grid_size=(
                    model_resolution[0] // stride,
                    model_resolution[1] // stride,
                ),
            ),
            persistent=False,
        )

        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.track_feat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.vis_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
        )

        self.conf_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
        )

        self.upsample_kernel_size = 5
        self.upsample_transformer = UpsampleTransformerAlibi(
            kernel_size=self.upsample_kernel_size, # kernel_size=3, # 
            stride=self.stride,
            latent_dim=self.latent_dim,
            num_attn_blocks=2,
            upsample_factor=self.upsample_factor,
        )

        self.initialize_up_weight()

        cmdtop_params = {
            "in_channel": 49,
            "out_channels": (64, 128, 128),
            "kernel_shapes": (3, 3, 2),
            "strides": (2, 2, 2),
        }

        self.cmdtop = nn.ModuleList([Corr4DMLP(**cmdtop_params) for _ in range(3)])

        self.radius_corr = 3
        dx = torch.linspace(-self.radius_corr, self.radius_corr, 2 * self.radius_corr + 1)
        dy = torch.linspace(-self.radius_corr, self.radius_corr, 2 * self.radius_corr + 1)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)
        delta = delta.view(2 * self.radius_corr + 1, 2 * self.radius_corr + 1, 2)
        self.register_buffer("delta_corr", delta)

        self.only_learnup = only_learnup

        if self.only_learnup:
            self.fixed_modules = [
                "fnet",
                "updateformer",
                "cmdtop",
                "norm",
                "track_feat_updater",
                "vis_predictor",
                "conf_predictor",
            ]

            for mod in self.fixed_modules:
                for p in getattr(self, mod).parameters():
                    p.requires_grad = False

        else:
            self.fixed_modules = []

    def set_upsample_factor(self, upsample_factor):
        self.upsample_factor = upsample_factor
        self.upsample_transformer.upsample_factor = upsample_factor
        
    def train(self, mode=True):
        super().train(mode)
        if self.only_learnup:
            for mod in self.fixed_modules:
                mod = getattr(self, mod)
                mod.eval()

    def get_latent_sim(self):
        latent = rearrange(self.updateformer.virual_tracks, "1 n 1 c -> n c")  # N C

        latent_norm = F.normalize(latent, p=2, dim=-1)  # [1xKnovelxC]
        latent_sim = einsum(latent_norm, latent_norm, "n c, m c -> n m")  # [KnovelxKnovel]

        return latent_sim

    def initialize_up_weight(self):
        def _basic_init(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                # torch.nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def register_cotracker(self):
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "..")       # densetrack3d/
        sys.path.append(os.path.abspath(project_root))
        from densetrack3d.models.cotracker.predictor import CoTrackerOnlinePredictor
        self.cotracker = CoTrackerOnlinePredictor(checkpoint='./checkpoints/scaled_online.pth').cuda()
        self.cotracker.eval()

        # self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").cuda()

    def upsample_with_mask(
        self, inp: Float[Tensor, "b c h_down w_down"], mask: Float[Tensor, "b 1 k h_up w_up"]
    ) -> Float[Tensor, "b c h_up w_up"]:
        """Upsample flow field [H/P, W/P, 2] -> [H, W, 2] using convex combination"""
        H, W = inp.shape[-2:]

        up_inp = F.unfold(
            inp, [self.upsample_kernel_size, self.upsample_kernel_size], padding=(self.upsample_kernel_size - 1) // 2
        )
        up_inp = rearrange(up_inp, "b c (h w) -> b c h w", h=H, w=W)
        up_inp = F.interpolate(up_inp, scale_factor=self.upsample_factor, mode="nearest")
        up_inp = rearrange(
            up_inp, "b (c i j) h w -> b c (i j) h w", i=self.upsample_kernel_size, j=self.upsample_kernel_size
        )

        up_inp = torch.sum(mask * up_inp, dim=2)
        return up_inp

    def get_single_corr_depth(
        self,
        depths: Float[Tensor, "b t 1 h w"],
        coords: Float[Tensor, "b t n c"],
        coord_depths: Float[Tensor, "b t n 1"],
        mode: str = "bilinear",
    ) -> Float[Tensor, "*"]:

        B, S, N = coords.shape[:3]
        local_queried_frames = torch.arange(0, S, device=coords.device).long()
        local_queried_frames = repeat(local_queried_frames, "s -> b s n 1", b=B, n=N)
        depth_queried_coords = torch.cat(
            [local_queried_frames, coords],
            dim=-1,
        )  # B S N 3

        sample_depth = sample_features5d(depths, depth_queried_coords, mode=mode)  # B, S, N, 1
        dcorrs = sample_depth / coord_depths

        return dcorrs

    def get_4dcorr_features(
        self,
        fmaps_pyramid: tuple[VideoType, ...],
        coords: Float[Tensor, "b t n c"],
        supp_track_feat: tuple[Float[Tensor, "*"], ...],
    ) -> Float[Tensor, "*"]:

        B, S, N = coords.shape[:3]
        r = self.radius_corr

        corrs_pyr = []
        for lvl, supp_track_feat_ in enumerate(supp_track_feat):
            centroid_lvl = coords.reshape(B * S, N, 1, 1, 2) / 2 ** (lvl - 1)
            coords_lvl = centroid_lvl + self.delta_corr[None, None]  # (B S) N (2r+1) (2r+1) 2

            _, _, C_, H_, W_ = fmaps_pyramid[lvl].shape
            sample_tgt_feat = bilinear_sampler(
                fmaps_pyramid[lvl].reshape(B * S, -1, H_, W_),
                coords_lvl.reshape(B * S, N, (2 * r + 1) * (2 * r + 1), 2),
                padding_mode="border",
            )

            sample_tgt_feat = sample_tgt_feat.view(B, S, -1, N, 2 * r + 1, 2 * r + 1)

            patches_input = einsum(sample_tgt_feat, supp_track_feat_, "b s c n h w, b n i j c -> b s n h w i j")
            patches_input = patches_input / torch.sqrt(torch.tensor(C_).float())
            patches_input = rearrange(patches_input, "b s n h w i j -> (b s n) h w i j")
            patches_emb = self.cmdtop[lvl](patches_input)
            patches = rearrange(patches_emb, "(b s n) c -> b s n c", b=B, s=S)

            corrs_pyr.append(patches)
        fcorrs = torch.cat(corrs_pyr, dim=-1)  # B S N C

        return fcorrs

    def get_track_feat(
        self,
        fmaps: VideoType,
        fmaps_pyramid: tuple[VideoType, ...],
        queried_frames: Float[Tensor, "b n"],
        queried_coords: Float[Tensor, "b n c"],
        num_levels: int = 3,
        radius: int = 3,
    ) -> tuple[Float[Tensor, "*"], tuple[Float[Tensor, "*"], ...]]:

        B = fmaps.shape[0]
        N = queried_coords.shape[1]

        sample_frames = queried_frames[:, None, :, None]
        sample_coords = torch.cat([sample_frames, queried_coords[:, None]], dim=-1)
        sample_track_feats = sample_features5d(fmaps, sample_coords)

        supp_track_feats_pyramid = []
        for lvl in range(num_levels):
            centroid_lvl = queried_coords.reshape(B * N, 1, 1, 2) / 2 ** (lvl - 1)
            coords_lvl = centroid_lvl + self.delta_corr[None]
            coords_lvl = coords_lvl.reshape(B, 1, N * (2 * radius + 1) * (2 * radius + 1), 2)

            sample_frames = queried_frames[:, None, :, None, None, None].expand(
                B, 1, N, 2 * radius + 1, 2 * radius + 1, 1
            )
            sample_frames = sample_frames.reshape(B, 1, N * (2 * radius + 1) * (2 * radius + 1), 1)
            sample_coords = torch.cat(
                [
                    sample_frames,
                    coords_lvl,
                ],
                dim=-1,
            )  # B 1 N' 3

            supp_track_feats = sample_features5d(
                fmaps_pyramid[lvl],
                sample_coords,
            )

            supp_track_feats = supp_track_feats.view(B, N, 2 * radius + 1, 2 * radius + 1, -1)
            supp_track_feats_pyramid.append(supp_track_feats)

        return sample_track_feats, supp_track_feats_pyramid

    def get_dense_track_feat(
        self,
        fmaps: VideoType,
        fmaps_pyramid: tuple[VideoType, ...],
        dense_coords: Float[Tensor, "b n c"],
        num_levels: int = 3,
        radius: int = 3,
    ) -> tuple[Float[Tensor, "*"], tuple[Float[Tensor, "*"], ...]]:

        B, N = dense_coords.shape[:2]

        sample_track_feats = sample_features4d(fmaps[:, 0], dense_coords)

        supp_track_feats_pyramid = []
        for lvl in range(num_levels):
            centroid_lvl = rearrange(dense_coords, "b n c -> (b n) () () c") / 2 ** (lvl - 1)

            coords_lvl = centroid_lvl + self.delta_corr[None]
            coords_lvl = rearrange(coords_lvl, "(b n) r1 r2 c -> b (n r1 r2) c", b=B, n=N)

            supp_track_feats = sample_features4d(fmaps_pyramid[lvl][:, 0], coords_lvl)
            supp_track_feats = rearrange(
                supp_track_feats, "b (n r1 r2) c -> b n r1 r2 c", n=N, r1=2 * radius + 1, r2=2 * radius + 1
            )
            supp_track_feats_pyramid.append(supp_track_feats)

        return sample_track_feats, supp_track_feats_pyramid

    def forward_window(
        self,
        fmaps_pyramid: tuple[VideoType, ...],
        depthmaps: Float[Tensor, "b t 1 h w"],
        coords: Float[Tensor, "b t n c"],
        coord_depths: Float[Tensor, "b t n 1"],
        vis: Float[Tensor, "b t n"],
        conf: Float[Tensor, "b t n"],
        track_feat: Float[Tensor, "b t n c"],
        supp_track_feat: tuple[Float[Tensor, "b t n c"], ...],
        track_mask: Float[Tensor, "b t n"],
        attention_mask: Bool[Tensor, "b t n"],
        iters: int = 4,
        use_efficient_global_attn: bool = False,
    ) -> tuple[
        Float[Tensor, "b t n c"],
        Float[Tensor, "b t n 1"],
        Float[Tensor, "b t n"],
        Float[Tensor, "b t n"],
        Float[Tensor, "b t n c"],
    ]:

        B, S, N = coords.shape[:3]

        track_mask_vis = torch.cat([track_mask, vis], dim=-1)  # b s n c

        # FIXME cut pos_emb in case input reso is smaller than default
        if self.input_reso[0] < self.model_resolution[0] or self.input_reso[1] < self.model_resolution[1]:
            pos_emb_crop = self.pos_emb[:, :, : self.input_reso[0] // self.stride, : self.input_reso[1] // self.stride]
            pos_emb = sample_features4d(pos_emb_crop.repeat(B, 1, 1, 1), coords[:, 0])
        else:
            assert self.input_reso[0] == self.model_resolution[0] and self.input_reso[1] == self.model_resolution[1]
            pos_emb = sample_features4d(self.pos_emb.repeat(B, 1, 1, 1), coords[:, 0])

        coord_preds = []
        coord_depth_preds = []

        # FIXME fix depth
        depthmaps[depthmaps < 1e-2] = 1e-2

        for iteration in range(iters):
            coords = coords.detach()  # B S N 3
            coord_depths = coord_depths.detach()
            coord_depths[coord_depths < 1e-2] = 1e-2

            # NOTE Prepare input to transformer
            fcorrs = self.get_4dcorr_features(fmaps_pyramid, coords, supp_track_feat)

            dcorrs = self.get_single_corr_depth(depthmaps, coords, coord_depths)

            # Get the 2D flow embeddings
            flows_2d = coords - coords[:, 0:1]
            flows_2d_emb = get_2d_embedding(flows_2d, 64, cat_coords=True)  # N S E
            flows_3d = coord_depths / coord_depths[:, 0:1]

            transformer_input = torch.cat(
                [
                    flows_2d_emb,
                    flows_2d,
                    flows_3d,
                    fcorrs,
                    dcorrs,
                    track_feat,
                    track_mask_vis,
                ],
                dim=-1,
            )
            x = transformer_input + pos_emb[:, None] + self.time_emb[:, :S, None]

            # NOTE Transformer part
            delta = self.updateformer(
                input_tensor=x,
                attn_mask=attention_mask,
                n_sparse=self.N_sparse,
                dH=self.dH,
                dW=self.dW,
                use_efficient_global_attn=use_efficient_global_attn,
            )

            delta_coords = delta[..., :2]
            ratio_coords_depths = torch.exp(0.1 * torch.clamp(delta[..., 2:3], -50, 50))
            delta_feat = self.track_feat_updater(self.norm(rearrange(delta[..., 3:], "b t n c -> (b t n) c")))
            delta_feat = rearrange(delta_feat, "(b t n) c -> b t n c", b=B, t=S)  ###########################

            # NOTE Update
            track_feat = track_feat + delta_feat
            coords = coords + delta_coords
            coord_depths = coord_depths * ratio_coords_depths
            ###########################

            coord_preds.append(coords.clone())
            coord_depth_preds.append(coord_depths.clone())

        vis_pred = self.vis_predictor(track_feat).squeeze(-1)  # b s n
        conf_pred = self.conf_predictor(track_feat).squeeze(-1)  # b s n

        return coord_preds, coord_depth_preds, vis_pred, conf_pred, track_feat

    def extract_features(self, video: VideoType) -> tuple[Float[Tensor, "*"], ...]:

        B, T = video.shape[:2]
        fmaps, higher_fmaps, lower_fmaps = self.fnet(
            rearrange(video, "b t c h w -> (b t) c h w"), return_intermediate=True
        )

        fmaps = rearrange(fmaps, "(b t) c h w -> b t c h w", b=B, t=T)
        higher_fmaps = rearrange(higher_fmaps, "(b t) c h w -> b t c h w", b=B, t=T)
        lower_fmaps = rearrange(lower_fmaps, "(b t) c h w -> b t c h w", b=B, t=T)

        return fmaps, higher_fmaps, lower_fmaps

    def filter_integer_coords(self, sparse_queried_coords, sparse_queried_frames):
        """
        Filter sparse_queried_coords for integer coordinates and select corresponding frames.
        
        Args:
            sparse_queried_coords (torch.Tensor): Shape [M, 3], coordinates with [y, x, ...].
            sparse_queried_frames (torch.Tensor): Shape [M], frame indices.
        
        Returns:
            tuple: (filtered_coords, filtered_frames)
                - filtered_coords: Shape [K, 2], integer coordinates [y, x].
                - filtered_frames: Shape [K], corresponding frame indices.
        """
        # Extract [y, x] coordinates
        coords = sparse_queried_coords[..., :2]  # Shape [M, 2]
        
        # Check if coordinates are integers (equal to their floor value)
        is_integer = torch.all(coords == coords.floor(), dim=-1)  # Shape [M]
        
        # Filter coordinates and frames
        filtered_coords = sparse_queried_coords[is_integer]  # Shape [K, 2], convert to long for integer coords
        filtered_frames = sparse_queried_frames[is_integer]  # Shape [K]
        
        return filtered_coords[None], filtered_frames[None]

    def filter_integer_coords_v2(self, sparse_queried_coords, sparse_queried_frames):
        """
        Filter sparse_queried_coords by keeping only the unique floored [y, x] coordinates.

        Args:
            sparse_queried_coords (torch.Tensor): Shape [M, 3], coordinates with [y, x, ...].
            sparse_queried_frames (torch.Tensor): Shape [M], frame indices.

        Returns:
            tuple: (filtered_coords, filtered_frames)
                - filtered_coords: Shape [K, 2], unique floored integer coordinates [y, x].
                - filtered_frames: Shape [K], one corresponding frame index for each unique coordinate.
        """
        # Extract and floor [y, x] coordinates
        coords = sparse_queried_coords[..., :2].floor().long()  # Shape [M, 2]

        # Get unique coordinates and indices
        unique_coords, inverse_indices = torch.unique(torch.cat([coords, sparse_queried_frames[..., None]], dim=-1), dim=1, return_inverse=True)

        # For each unique coordinate, select the first corresponding frame index
        selected_indices = torch.zeros(unique_coords.size(1), dtype=torch.long, device=coords.device)
        for i in range(unique_coords.size(1)):
            selected_indices[i] = torch.nonzero(inverse_indices == i, as_tuple=False)[0]

        # Use selected indices to get corresponding frames
        filtered_frames = sparse_queried_frames[:, selected_indices]
        filtered_coords = sparse_queried_coords[:, selected_indices]
        return filtered_coords, filtered_frames, selected_indices

    def prepare_sparse_queries(
        self,
        sparse_queries: Float[Tensor, "b n c"],
        fmaps: VideoType,
        fmaps_pyramid: tuple[VideoType, ...],
        second_stage=False
    ) -> tuple:

        B, S, _, fH, fW = fmaps.shape
        if second_stage:
          S = self.window_len

        device = fmaps.device

        self.N_sparse = N_sparse = sparse_queries.shape[1]
        sparse_queried_frames = sparse_queries[:, :, 0].long()

        # NOTE normalize queries
        sparse_queried_coords = sparse_queries[..., 1:4].clone()
        sparse_queried_coords[..., :2] = sparse_queried_coords[..., :2] / self.stride
        selected_indices = None
        if second_stage:
          if sparse_queried_coords.shape[1]!=36*48:
            
            sparse_queried_coords_extra, sparse_queried_frames_extra, selected_indices = self.filter_integer_coords_v2(sparse_queried_coords[:, 36*48:], sparse_queried_frames[:, 36*48:])
            sparse_queried_coords_extra = sparse_queried_coords_extra

            sparse_queried_coords = torch.cat([sparse_queried_coords[:, :36*48], sparse_queried_coords_extra], dim=1)
            sparse_queried_frames = torch.cat([sparse_queried_frames[:, :36*48], sparse_queried_frames_extra], dim=1)
            # sparse_queried_coords, sparse_queried_frames = self.filter_integer_coords_v2(sparse_queried_coords, sparse_queried_frames)
          self.N_sparse = N_sparse = sparse_queried_coords.shape[1]
        # We compute track features # FIXME only get 2d coord
        track_feat, supp_track_feats_pyramid = self.get_track_feat(
            fmaps,
            fmaps_pyramid,
            sparse_queried_frames,
            sparse_queried_coords[..., :2],
        )
        track_feat = repeat(track_feat, "b 1 n c -> b s n c", s=S)

        coords_init = sparse_queried_coords[..., :2].reshape(B, 1, N_sparse, 2).expand(B, S, N_sparse, 2).float()
        depths_init = sparse_queried_coords[..., 2:3].reshape(B, 1, N_sparse, 1).expand(B, S, N_sparse, 1).float()
        vis_init = torch.ones((B, S, N_sparse, 1), device=device).float() * 10
        conf_init = torch.ones((B, S, N_sparse, 1), device=device).float() * 10

        return (
            coords_init,
            depths_init,
            vis_init,
            conf_init,
            track_feat,
            supp_track_feats_pyramid,
            sparse_queried_frames,
            selected_indices
        )

    
    def prepare_dense_queries(
        self,
        coords_init: Float[Tensor, "b t n c"],
        coord_depths_init: Float[Tensor, "b t n 1"],
        vis_init: Float[Tensor, "b t n"],
        conf_init: Float[Tensor, "b t n"],
        track_feat: Float[Tensor, "b t n c"],
        supp_track_feats_pyramid: tuple[Float[Tensor, "*"], ...],
        fmaps: VideoType,
        fmaps_pyramid: tuple[VideoType, ...],
        depth_init_downsample: Float[Tensor, "b 1 h w"],
        is_train: bool = False,
        second_stage=False
    ) -> tuple:

        B, S, _, fH, fW = fmaps.shape
        if second_stage:
          S = self.window_len
        device = fmaps.device

        if is_train:
            if self.upsample_factor == 4:
                dH, dW = 15, 20
            else:
                dH, dW = 9, 12
            y0 = np.random.randint(0, fH * self.stride / self.upsample_factor - dH, size=B)
            x0 = np.random.randint(0, fW * self.stride / self.upsample_factor - dW, size=B)
        else:
            # dH, dW = self.model_resolution[0] // (self.stride), self.model_resolution[1] // (self.stride)
            dH, dW = self.input_reso[0] // (self.upsample_factor), self.input_reso[1] // (self.upsample_factor)
            x0, y0 = [0] * B, [0] * B
        self.dH, self.dW = dH, dW

        cropped_depths = torch.stack(
            [depth_init_downsample[b, :, y0_ : y0_ + dH, x0_ : x0_ + dW] for (b, y0_, x0_) in zip(range(B), y0, x0)],
            dim=0,
        )
        cropped_depths = rearrange(cropped_depths, "b c h w -> b (h w) c")

        dense_grid_2d = (
            get_grid(dH, dW, normalize=False, device=fmaps.device).reshape(-1, 2).unsqueeze(0).repeat(B, 1, 1)
        )  # B, (H, W) 2
        dense_grid_2d = dense_grid_2d * self.upsample_factor / self.stride 

        for b in range(B):
            dense_grid_2d[b, :, 0] += x0[b]
            dense_grid_2d[b, :, 1] += y0[b]

        dense_coords_init = repeat(dense_grid_2d, "b n c -> b s n c", s=S)
        dense_coord_depths_init = repeat(cropped_depths, "b n c -> b s n c", s=S)
        dense_vis_init = torch.ones((B, S, dH * dW, 1), device=device).float() * 10
        dense_conf_init = torch.ones((B, S, dH * dW, 1), device=device).float() * 10
        self.dense_grid_2d = dense_grid_2d
        dense_track_feat, dense_supp_track_feats_pyramid = self.get_dense_track_feat(
            fmaps,
            fmaps_pyramid,
            dense_grid_2d,
        )
        dense_track_feat = repeat(dense_track_feat, "b n c -> b s n c", s=S)

        dense_grid_2d_up = (
            get_grid(dH * self.upsample_factor, dW * self.upsample_factor, normalize=False, device=fmaps.device)
            .reshape(-1, 2)
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )
        for b in range(B):
            dense_grid_2d_up[b, :, 0] += x0[b] * self.upsample_factor
            dense_grid_2d_up[b, :, 1] += y0[b] * self.upsample_factor

        self.original_grid_low_reso = rearrange(dense_grid_2d.clone(), "b (h w) c -> b c h w", h=dH, w=dW)
        self.original_grid_high_reso = rearrange(
            dense_grid_2d_up.clone(), "b (h w) c -> b c h w", h=dH * self.upsample_factor, w=dW * self.upsample_factor
        )

        coords_init = smart_cat(
            coords_init, dense_coords_init, dim=2
        )  # torch.cat([coords_init, dense_coords_init], dim=2)
        coord_depths_init = smart_cat(
            coord_depths_init, dense_coord_depths_init, dim=2
        )  # torch.cat([coord_depths_init, dense_coord_depths_init], dim=2)
        vis_init = smart_cat(vis_init, dense_vis_init, dim=2)  # torch.cat([vis_init, dense_vis_init], dim=2)
        conf_init = smart_cat(conf_init, dense_conf_init, dim=2)  # torch.cat([vis_init, dense_vis_init], dim=2)
        track_feat = smart_cat(track_feat, dense_track_feat, dim=2)  # torch.cat([track_feat, dense_track_feat], dim=2)
        supp_track_feats_pyramid = [
            smart_cat(sf, dense_sf, dim=1)
            for sf, dense_sf in zip(supp_track_feats_pyramid, dense_supp_track_feats_pyramid)
        ]

        return (coords_init, coord_depths_init, vis_init, conf_init, track_feat, supp_track_feats_pyramid, (x0, y0))
    
    def forward(
        self,
        video: VideoType,
        videodepth: Float[Tensor, "b t 1 h w"],
        sparse_queries: Float[Tensor, "b n c"] = None,
        depth_init: Float[Tensor, "b 1 h w"] = None,
        iters: int = 4,
        is_train: bool = False,
        use_dense: bool = True,
        use_efficient_global_attn: bool = False,
        addtional_features=None
    ) -> tuple[dict | None, dict | None, tuple[dict | None, dict | None] | None]:
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3, H, W]): input videos.
            videodepth (FloatTensor[B, T, 1, H, W]): input videodepths.
            queries (FloatTensor[B, N, 3]): point queries.
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
            is_online (bool, optional): enables online mode. Defaults to False. Before enabling, call model.init_video_online_processing().
        """

        B, T, C, H, W = video.shape
        
        S = self.window_len
        device = video.device

        self.input_reso = (H, W)

        use_sparse = True if sparse_queries is not None else False

        assert use_sparse or use_dense, "At least one of use_sparse and use_dense must be True"
        self.use_dense = use_dense
        self.use_sparse = use_sparse

        assert S >= 2  # A tracker needs at least two frames to track something

        self.ori_T = ori_T = T
        self.Dz = Dz = W // self.stride

        step = S // 2  # How much the sliding window moves at every step
        video = 2 * (video / 255.0) - 1.0

        # Pad the video so that an integer number of sliding windows fit into it
        # TODO: we may drop this requirement because the transformer should not care
        # TODO: pad the features instead of the video
        if 1:#is_train:
            pad = 0
        else:
            pad = (S - T % S) % S  # We don't want to pad if T % S == 0

        if pad > 0:
            video = F.pad(video.reshape(B, 1, T, C * H * W), (0, 0, 0, pad), "replicate").reshape(B, -1, C, H, W)
            videodepth = F.pad(videodepth.reshape(B, 1, T, H * W), (0, 0, 0, pad), "replicate").reshape(B, -1, 1, H, W)

        if depth_init is None:
            depth_init = videodepth[:, 0].clone()

        fmaps, higher_fmaps, lower_fmaps = self.extract_features(video)
        fmaps_pyramid = [higher_fmaps, fmaps, lower_fmaps]
        fH, fW = fmaps.shape[-2:]

        videodepth_downsample = F.interpolate(videodepth.reshape(-1, 1, H, W), size=(fH, fW), mode="nearest").reshape(
            B, -1, 1, fH, fW
        )

        # depth_init_downsample = F.interpolate(depth_init, size=(fH, fW), mode="nearest")
        depth_init_downsample = F.interpolate(
            depth_init,
            size=(int(fH * self.stride/self.upsample_factor), int(fW * self.stride/self.upsample_factor)),
            mode='nearest'
        )

        self.dH, self.dW = 0, 0
        self.N_sparse = 0

        coords_init, depths_init, vis_init, conf_init, track_feat, supp_track_feats_pyramid = (
            None,
            None,
            None,
            None,
            None,
            [None] * 3,
        )
        if use_sparse:
            (
                coords_init,
                depths_init,
                vis_init,
                conf_init,
                track_feat,
                supp_track_feats_pyramid,
                sparse_queried_frames,
                selected_indices
            ) = self.prepare_sparse_queries(sparse_queries, fmaps, fmaps_pyramid)

            # We store our predictions here
            coords_predicted = torch.zeros((B, ori_T, self.N_sparse, 2), device=device)
            coords_depths_predicted = torch.zeros((B, ori_T, self.N_sparse, 1), device=device)
            vis_predicted = torch.zeros((B, ori_T, self.N_sparse), device=device)
            conf_predicted = torch.zeros((B, ori_T, self.N_sparse), device=device)
            all_coords_predictions, all_coord_depths_predictions, all_vis_predictions, all_conf_predictions = (
                [],
                [],
                [],
                [],
            )

        if use_dense:
            coords_init, depths_init, vis_init, conf_init, track_feat, supp_track_feats_pyramid, (x0, y0) = (
                self.prepare_dense_queries(
                    coords_init,
                    depths_init,
                    vis_init,
                    conf_init,
                    track_feat,
                    supp_track_feats_pyramid,
                    fmaps,
                    fmaps_pyramid,
                    depth_init_downsample,
                    is_train,
                )
            )

            dense_coords_up_predicted = torch.zeros(
                (B, ori_T, 2, self.dH * self.upsample_factor, self.dW * self.upsample_factor), device=device
            )
            dense_coord_depths_up_predicted = torch.zeros(
                (B, ori_T, 1, self.dH * self.upsample_factor, self.dW * self.upsample_factor), device=device
            )
            dense_vis_up_predicted = torch.zeros(
                (B, ori_T, self.dH * self.upsample_factor, self.dW * self.upsample_factor), device=device
            )
            dense_conf_up_predicted = torch.zeros(
                (B, ori_T, self.dH * self.upsample_factor, self.dW * self.upsample_factor), device=device
            )

            (
                all_dense_coords_predictions,
                all_dense_coord_depths_predictions,
                all_dense_vis_predictions,
                all_dense_conf_predictions,
            ) = ([], [], [], [])
            up_mask = None

        # We process ((num_windows - 1) * step + S) frames in total, so there are
        # (ceil((T - S) / step) + 1) windows
        num_windows = (T - S + step - 1) // step + 1
        # We process only the current video chunk in the online mode
        indices = range(0, step * num_windows, step)
        if len(indices) == 0:
            indices = [0]

        for ind in indices:
            # We copy over coords and vis for tracks that are queried
            # by the end of the previous window, which is ind + overlap
            if ind > 0:
                overlap = S - step

                copy_over = None

                if use_sparse:
                    copy_over = (sparse_queried_frames < ind + overlap)[:, None, :, None]  # B 1 N 1
                if use_dense:
                    copy_over = smart_cat(
                        copy_over, torch.ones((B, 1, self.dH * self.dW, 1), device=device), dim=2
                    ).bool()

                last_coords = coords[-1][:, -overlap:].clone()
                last_depths = coords_depths[-1][:, -overlap:].clone()
                last_vis = vis[:, -overlap:].clone()[..., None]
                last_conf = conf[:, -overlap:].clone()[..., None]

                coords_prev = torch.nn.functional.pad(
                    last_coords,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 2
                depths_prev = torch.nn.functional.pad(
                    last_depths,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 2
                vis_prev = torch.nn.functional.pad(
                    last_vis,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 1
                conf_prev = torch.nn.functional.pad(
                    last_conf,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 1

                coords_init = torch.where(copy_over.expand_as(coords_init), coords_prev, coords_init)
                depths_init = torch.where(copy_over.expand_as(depths_init), depths_prev, depths_init)
                vis_init = torch.where(copy_over.expand_as(vis_init), vis_prev, vis_init)
                conf_init = torch.where(copy_over.expand_as(conf_init), conf_prev, conf_init)

            # The attention mask is 1 for the spatio-temporal points within a track which is updated in the current window
            attention_mask, track_mask = None, None
            if use_sparse:
                attention_mask = (
                    (sparse_queried_frames < ind + T).reshape(B, 1, self.N_sparse).repeat(1, T, 1)
                )  # B S N

                # The track mask is 1 for the spatio-temporal points that actually need updating: only after begin queried, and not if contained in a previous window
                track_mask = (
                    sparse_queried_frames[:, None, :, None]
                    <= torch.arange(ind, ind + T, device=device)[None, :, None, None]
                ).contiguous()  # B S N 1
                if ind > 0:
                    track_mask[:, :overlap, :, :] = False

            if use_dense:
                track_mask = smart_cat(track_mask, torch.ones((B, T, self.dH * self.dW, 1), device=device), dim=2)
                attention_mask = smart_cat(
                    attention_mask, torch.ones((B, T, self.dH * self.dW), device=device), dim=2
                ).bool()

            coords, coords_depths, vis, conf, track_feat_updated = self.forward_window(
                fmaps_pyramid=[f[:, ind : ind + T] for f in fmaps_pyramid],
                depthmaps=videodepth_downsample[:, ind : ind + T],
                coords=coords_init,
                coord_depths=depths_init,
                vis=vis_init,
                conf=conf_init,
                track_feat=attention_mask.unsqueeze(-1).float() * track_feat,
                supp_track_feat=supp_track_feats_pyramid,
                track_mask=track_mask,
                attention_mask=attention_mask,
                iters=iters,
                use_efficient_global_attn=use_efficient_global_attn,
            )

            S_trimmed = min(T - ind, T)  # accounts for last window duration

            if use_sparse:
                coords_predicted[:, ind : ind + T] = coords[-1][:, :S_trimmed, : self.N_sparse] * self.stride
                coords_depths_predicted[:, ind : ind + T] = coords_depths[-1][:, :S_trimmed, : self.N_sparse]
                vis_predicted[:, ind : ind + T] = vis[:, :S_trimmed, : self.N_sparse]
                conf_predicted[:, ind : ind + T] = conf[:, :S_trimmed, : self.N_sparse]

                if is_train:
                    all_coords_predictions.append(
                        [(coord[:, :S_trimmed, : self.N_sparse] * self.stride) for coord in coords]
                    )
                    all_coord_depths_predictions.append(
                        [coords_d[:, :S_trimmed, : self.N_sparse] for coords_d in coords_depths]
                    )
                    all_vis_predictions.append(torch.sigmoid(vis[:, :S_trimmed, : self.N_sparse]))
                    all_conf_predictions.append(torch.sigmoid(conf[:, :S_trimmed, : self.N_sparse]))

            if use_dense:

                dense_coords_down_list = [
                    rearrange(coord_[:, :, self.N_sparse :], "b s (h w) c -> (b s) c h w", h=self.dH, w=self.dW)
                    for coord_ in coords
                ]
                dense_coords_depths_down_list = [
                    rearrange(coords_depth_[:, :, self.N_sparse :], "b s (h w) c -> (b s) c h w", h=self.dH, w=self.dW)
                    for coords_depth_ in coords_depths
                ]
                dense_vis_down = rearrange(
                    vis[:, :, self.N_sparse :], "b s (h w) -> (b s) 1 h w", h=self.dH, w=self.dW
                )
                dense_conf_down = rearrange(
                    conf[:, :, self.N_sparse :], "b s (h w) -> (b s) 1 h w", h=self.dH, w=self.dW
                )

                if up_mask is None:
                    flow_guidance = (rearrange(dense_coords_down_list[-1], "(b s) c h w -> b s c h w", b=B) - rearrange(self.original_grid_low_reso, "b c h w -> b 1 c h w")) / self.Dz
                    flow_guidance_extra = flow_guidance[:,-1:].repeat(1, S-T, 1, 1, 1)
                    flow_guidance = smart_cat(flow_guidance, flow_guidance_extra, dim=1)
                    #flow_guidance = rearrange(flow_guidance, "(b s) c h w -> b s c h w", b=B, s=S)
                    upsample_featmap = rearrange(
                        track_feat_updated[:, 0, self.N_sparse :], "b (h w) c -> b c h w", h=self.dH, w=self.dW
                    )
                    up_mask = []
                    for batch in range(B):
                      up_mask.append(self.upsample_transformer(
                          feat_map=upsample_featmap[batch:batch+1],
                          flow_map=flow_guidance[batch:batch+1],
                      ))
                    up_mask = torch.cat(up_mask, dim=0)
                    up_mask = repeat(up_mask, "b k h w -> b s k h w", s=T)
                    up_mask = rearrange(up_mask, "b s c h w -> (b s) 1 c h w")

                dense_coords_up_list, dense_coords_depths_up_list = [], []
                for pred_id_ in range(len(dense_coords_down_list)):
                    dense_coords_down = dense_coords_down_list[pred_id_]
                    dense_coords_depths_down = dense_coords_depths_down_list[pred_id_]

                    dense_coords_up = self.original_grid_high_reso[:, None] + rearrange(self.upsample_with_mask(
                        rearrange(rearrange(dense_coords_down, "(b s) c h w -> b s c h w", b=B) - rearrange(self.original_grid_low_reso, "b c h w -> b 1 c h w"), "b s c h w -> (b s) c h w") * self.stride,
                        up_mask,
                    ), "(b s) c h w -> b s c h w", b=B)
                    #dense_coords_up = rearrange(dense_coords_up, "(b s) c h w -> b s c h w", b=B, s=S)
                    dense_coords_up_list.append(dense_coords_up)

                    dense_coords_depths_up = self.upsample_with_mask(
                        dense_coords_depths_down,  # dense_coords_depths_down * (self.d_far-self.d_near) / self.Dz + self.d_near,
                        up_mask,
                    )

                    dense_coords_depths_up = rearrange(dense_coords_depths_up, "(b s) c h w -> b s c h w", b=B)

                    dense_coords_depths_up_list.append(dense_coords_depths_up)

                dense_vis_up = self.upsample_with_mask(dense_vis_down, up_mask)
                dense_vis_up = rearrange(dense_vis_up, "(b s) 1 h w -> b s h w", b=B)

                dense_conf_up = self.upsample_with_mask(dense_conf_down, up_mask)
                dense_conf_up = rearrange(dense_conf_up, "(b s) 1 h w -> b s h w", b=B)

                dense_coords_up_predicted[:, ind : ind + T] = dense_coords_up_list[-1][
                    :, :S_trimmed
                ]  # dense_coords_out[:, -1]
                dense_coord_depths_up_predicted[:, ind : ind + T] = dense_coords_depths_up_list[-1][
                    :, :S_trimmed
                ]  #  dense_coord_depths_out[:, -1]
                dense_vis_up_predicted[:, ind : ind + T] = dense_vis_up[:, :S_trimmed]
                dense_conf_up_predicted[:, ind : ind + T] = dense_conf_up[:, :S_trimmed]

                if is_train:
                    all_dense_coords_predictions.append(
                        [dense_coord[:, :S_trimmed] for dense_coord in dense_coords_up_list]
                    )  # B I T C H W
                    all_dense_coord_depths_predictions.append(
                        [dense_coord_depths[:, :S_trimmed] for dense_coord_depths in dense_coords_depths_up_list]
                    )  # B I T C H W
                    all_dense_vis_predictions.append(torch.sigmoid(dense_vis_up[:, :S_trimmed]))
                    all_dense_conf_predictions.append(torch.sigmoid(dense_conf_up[:, :S_trimmed]))

        sparse_predictions, dense_predictions = None, None

        if use_sparse:
            vis_predicted = torch.sigmoid(vis_predicted)
            conf_predicted = torch.sigmoid(conf_predicted)
            sparse_predictions = dict(
                coords=coords_predicted,
                coord_depths=coords_depths_predicted,
                vis=vis_predicted,
                conf=conf_predicted,
            )

        if use_dense:
            dense_vis_up_predicted = torch.sigmoid(dense_vis_up_predicted)
            dense_conf_up_predicted = torch.sigmoid(dense_conf_up_predicted)
            dense_predictions = dict(
                coords=dense_coords_up_predicted,
                coord_depths=dense_coord_depths_up_predicted,
                vis=dense_vis_up_predicted,
                conf=dense_conf_up_predicted,
            )

        if not is_train:
            return sparse_predictions, dense_predictions, None

        sparse_train_data_dict, dense_train_data_dict = None, None
        if use_sparse:
            mask = sparse_queried_frames[:, None] <= torch.arange(0, T, device=device)[None, :, None]
            sparse_train_data_dict = dict(
                coords=all_coords_predictions,
                coord_depths=all_coord_depths_predictions,
                vis=all_vis_predictions,
                conf=all_conf_predictions,
                mask=mask,
            )

        if use_dense:
            dense_train_data_dict = dict(
                coords=all_dense_coords_predictions,
                coord_depths=all_dense_coord_depths_predictions,
                vis=all_dense_vis_predictions,
                conf=all_dense_conf_predictions,
                x0y0=(x0, y0),
            )

        train_data = (sparse_train_data_dict, dense_train_data_dict)

        return sparse_predictions, dense_predictions, train_data

    def forward_epef(
        self,
        video: VideoType,
        videodepth: Float[Tensor, "b t 1 h w"],
        sparse_queries: Float[Tensor, "b n c"] = None,
        depth_init: Float[Tensor, "b 1 h w"] = None,
        iters: int = 4,
        is_train: bool = False,
        use_dense: bool = True,
        use_efficient_global_attn: bool = False,
        addtional_features=None,
        init_features=None,
        use_cotracker=False,
        cotracker_results=None
    ) -> tuple[dict | None, dict | None, tuple[dict | None, dict | None] | None]:
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3, H, W]): input videos.
            videodepth (FloatTensor[B, T, 1, H, W]): input videodepths.
            queries (FloatTensor[B, N, 3]): point queries.
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
            is_online (bool, optional): enables online mode. Defaults to False. Before enabling, call model.init_video_online_processing().
        """

        B, T, C, H, W = video.shape
        # if T < 16: 
        #   self.window_len = T

        S = self.window_len
        device = video.device

        self.input_reso = (H, W)

        use_sparse = True if sparse_queries is not None else False

        assert use_sparse or use_dense, "At least one of use_sparse and use_dense must be True"
        self.use_dense = use_dense
        self.use_sparse = use_sparse

        assert S >= 2  # A tracker needs at least two frames to track something

        self.ori_T = ori_T = T
        self.Dz = Dz = W // self.stride

        step = S // 2  # How much the sliding window moves at every step
        video = 2 * (video / 255.0) - 1.0

        # Pad the video so that an integer number of sliding windows fit into it
        # TODO: we may drop this requirement because the transformer should not care
        # TODO: pad the features instead of the video
        if is_train:
            pad = 0
        else:
            pad = (S - T % S) % S  # We don't want to pad if T % S == 0

        if pad > 0:
            video = F.pad(video.reshape(B, 1, T, C * H * W), (0, 0, 0, pad), "replicate").reshape(B, -1, C, H, W)
            videodepth = F.pad(videodepth.reshape(B, 1, T, H * W), (0, 0, 0, pad), "replicate").reshape(B, -1, 1, H, W)
        if use_cotracker:
            cotracker_tracks, cotracker_visibility, cotracker_conf = cotracker_results
            if pad > 0:
                cotracker_tracks = torch.nn.functional.pad(cotracker_tracks, (0, 0, 0, 0, 0, pad), "replicate")
                cotracker_visibility = torch.nn.functional.pad(cotracker_visibility, (0, 0, 0, pad), "replicate")
                cotracker_conf = torch.nn.functional.pad(cotracker_conf, (0, 0, 0, pad), "replicate")
        if depth_init is None:
            depth_init = videodepth[:, 0].clone()

        if init_features == None:
            fmaps, higher_fmaps, lower_fmaps = self.extract_features(video)
            fmaps_pyramid = [higher_fmaps, fmaps, lower_fmaps]
        else:
            fmaps_pyramid = [F.pad(f.reshape(B, 1, T, f.shape[2] * f.shape[3] * f.shape[4]), (0, 0, 0, pad), "replicate").reshape(B, -1, f.shape[2], f.shape[3], f.shape[4]) for f in init_features]
            fmaps, higher_fmaps, lower_fmaps = fmaps_pyramid[1], fmaps_pyramid[0], fmaps_pyramid[2]

        fH, fW = fmaps.shape[-2:]

        videodepth_downsample = F.interpolate(videodepth.reshape(-1, 1, H, W), size=(fH, fW), mode="nearest").reshape(
            B, -1, 1, fH, fW
        )

        # depth_init_downsample = F.interpolate(depth_init, size=(fH, fW), mode="nearest")
        depth_init_downsample = F.interpolate(
            depth_init,
            size=(int(fH * self.stride/self.upsample_factor), int(fW * self.stride/self.upsample_factor)),
            mode='nearest'
        )

        self.dH, self.dW = 0, 0
        self.N_sparse = 0

        coords_init, depths_init, vis_init, conf_init, track_feat, supp_track_feats_pyramid, selected_indices = (
            None,
            None,
            None,
            None,
            None,
            [None] * 3,
            None
        )
        if use_sparse:
            (
                coords_init,
                depths_init,
                vis_init,
                conf_init,
                track_feat,
                supp_track_feats_pyramid,
                sparse_queried_frames,
                selected_indices
            ) = self.prepare_sparse_queries(sparse_queries, fmaps, fmaps_pyramid, second_stage=True)

            # We store our predictions here
            coords_predicted = torch.zeros((B, ori_T, self.N_sparse, 2), device=device)
            coords_depths_predicted = torch.zeros((B, ori_T, self.N_sparse, 1), device=device)
            vis_predicted = torch.zeros((B, ori_T, self.N_sparse), device=device)
            conf_predicted = torch.zeros((B, ori_T, self.N_sparse), device=device)
            all_coords_predictions, all_coord_depths_predictions, all_vis_predictions, all_conf_predictions = (
                [],
                [],
                [],
                [],
            )

        if use_dense:
            coords_init, depths_init, vis_init, conf_init, track_feat, supp_track_feats_pyramid, (x0, y0) = (
                self.prepare_dense_queries(
                    coords_init,
                    depths_init,
                    vis_init,
                    conf_init,
                    track_feat,
                    supp_track_feats_pyramid,
                    fmaps,
                    fmaps_pyramid,
                    depth_init_downsample,
                    is_train,
                    second_stage=True
                )
            )

            dense_coords_up_predicted = torch.zeros(
                (B, ori_T, 2, self.dH * self.upsample_factor, self.dW * self.upsample_factor), device=device
            )
            dense_coord_depths_up_predicted = torch.zeros(
                (B, ori_T, 1, self.dH * self.upsample_factor, self.dW * self.upsample_factor), device=device
            )
            dense_vis_up_predicted = torch.zeros(
                (B, ori_T, self.dH * self.upsample_factor, self.dW * self.upsample_factor), device=device
            )
            dense_conf_up_predicted = torch.zeros(
                (B, ori_T, self.dH * self.upsample_factor, self.dW * self.upsample_factor), device=device
            )

            (
                all_dense_coords_predictions,
                all_dense_coord_depths_predictions,
                all_dense_vis_predictions,
                all_dense_conf_predictions,
            ) = ([], [], [], [])
            up_mask = None

        # We process ((num_windows - 1) * step + S) frames in total, so there are
        # (ceil((T - S) / step) + 1) windows
        num_windows = (T - S + step - 1) // step + 1
        # We process only the current video chunk in the online mode
        indices = range(0, step * num_windows, step)
        if len(indices) == 0:
            indices = [0]

        for ind in indices:
            # We copy over coords and vis for tracks that are queried
            # by the end of the previous window, which is ind + overlap
            if ind > 0:
                overlap = S - step

                copy_over = None

                if use_sparse:
                    copy_over = (sparse_queried_frames < ind + overlap)[:, None, :, None]  # B 1 N 1
                if use_dense:
                    copy_over = smart_cat(
                        copy_over, torch.ones((B, 1, self.dH * self.dW, 1), device=device), dim=2
                    ).bool()

                last_coords = coords[-1][:, -overlap:].clone()
                last_depths = coords_depths[-1][:, -overlap:].clone()
                last_vis = vis[:, -overlap:].clone()[..., None]
                last_conf = conf[:, -overlap:].clone()[..., None]

                coords_prev = torch.nn.functional.pad(
                    last_coords,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 2
                depths_prev = torch.nn.functional.pad(
                    last_depths,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 2
                vis_prev = torch.nn.functional.pad(
                    last_vis,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 1
                conf_prev = torch.nn.functional.pad(
                    last_conf,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 1

                coords_init = torch.where(copy_over.expand_as(coords_init), coords_prev, coords_init)
                depths_init = torch.where(copy_over.expand_as(depths_init), depths_prev, depths_init)
                vis_init = torch.where(copy_over.expand_as(vis_init), vis_prev, vis_init)
                conf_init = torch.where(copy_over.expand_as(conf_init), conf_prev, conf_init)

            # The attention mask is 1 for the spatio-temporal points within a track which is updated in the current window
            attention_mask, track_mask = None, None
            if use_sparse:
                attention_mask = (
                    (sparse_queried_frames < ind + S).reshape(B, 1, self.N_sparse).repeat(1, S, 1)
                )  # B S N

                # The track mask is 1 for the spatio-temporal points that actually need updating: only after begin queried, and not if contained in a previous window
                track_mask = (
                    sparse_queried_frames[:, None, :, None]
                    <= torch.arange(ind, ind + S, device=device)[None, :, None, None]
                ).contiguous()  # B S N 1
                if ind > 0:
                    track_mask[:, :overlap, :, :] = False

            if use_dense:
                track_mask = smart_cat(track_mask, torch.ones((B, S, self.dH * self.dW, 1), device=device), dim=2)
                attention_mask = smart_cat(
                    attention_mask, torch.ones((B, S, self.dH * self.dW), device=device), dim=2
                ).bool()

            torch.cuda.empty_cache()
            coords, coords_depths, vis, conf, track_feat_updated = self.forward_window(
                fmaps_pyramid=[f[:, ind : ind + S] for f in fmaps_pyramid],
                depthmaps=videodepth_downsample[:, ind : ind + S],
                coords=coords_init,
                coord_depths=depths_init,
                vis=vis_init,
                conf=conf_init,
                track_feat=attention_mask.unsqueeze(-1).float() * track_feat,
                supp_track_feat=supp_track_feats_pyramid,
                track_mask=track_mask,
                attention_mask=attention_mask,
                iters=iters,
                use_efficient_global_attn=use_efficient_global_attn,
            )

            S_trimmed = min(T - ind, S)  # accounts for last window duration

            if use_sparse:
                coords_predicted[:, ind : ind + S] = coords[-1][:, :S_trimmed, : self.N_sparse] * self.stride
                coords_depths_predicted[:, ind : ind + S] = coords_depths[-1][:, :S_trimmed, : self.N_sparse]
                vis_predicted[:, ind : ind + S] = vis[:, :S_trimmed, : self.N_sparse]
                conf_predicted[:, ind : ind + S] = conf[:, :S_trimmed, : self.N_sparse]

                if is_train:
                    all_coords_predictions.append(
                        [(coord[:, :S_trimmed, : self.N_sparse] * self.stride) for coord in coords]
                    )
                    all_coord_depths_predictions.append(
                        [coords_d[:, :S_trimmed, : self.N_sparse] for coords_d in coords_depths]
                    )
                    all_vis_predictions.append(torch.sigmoid(vis[:, :S_trimmed, : self.N_sparse]))
                    all_conf_predictions.append(torch.sigmoid(conf[:, :S_trimmed, : self.N_sparse]))

            if use_dense:

                dense_coords_down_list = [
                    rearrange(coord_[:, :, self.N_sparse :], "b s (h w) c -> (b s) c h w", h=self.dH, w=self.dW)
                    for coord_ in coords
                ]
                dense_coords_depths_down_list = [
                    rearrange(coords_depth_[:, :, self.N_sparse :], "b s (h w) c -> (b s) c h w", h=self.dH, w=self.dW)
                    for coords_depth_ in coords_depths
                ]
                dense_vis_down = rearrange(
                    vis[:, :, self.N_sparse :], "b s (h w) -> (b s) 1 h w", h=self.dH, w=self.dW
                )
                dense_conf_down = rearrange(
                    conf[:, :, self.N_sparse :], "b s (h w) -> (b s) 1 h w", h=self.dH, w=self.dW
                )

                if up_mask is None:
                    if use_cotracker:
                        flow_guidance = (rearrange(cotracker_tracks[:, ind : ind + S], "b s (h w) c -> (b s) c h w", b=B, s=S, c=2, w=dense_coords_down_list[-1].shape[-1]) - self.original_grid_low_reso) / self.Dz
                    else:
                        flow_guidance = (dense_coords_down_list[-1] - self.original_grid_low_reso) / self.Dz
                    flow_guidance = rearrange(flow_guidance, "(b s) c h w -> b s c h w", b=B, s=S)
                    # flow_guidance_extra = flow_guidance[:,-1:].repeat(1, 16-S, 1, 1, 1)
                    # flow_guidance = smart_cat(flow_guidance, flow_guidance_extra, dim=1)
                    upsample_featmap = rearrange(
                        track_feat_updated[:, 0, self.N_sparse :], "b (h w) c -> b c h w", h=self.dH, w=self.dW
                    )
                    torch.cuda.empty_cache()
                    up_mask = self.upsample_transformer(
                        feat_map=upsample_featmap,
                        flow_map=flow_guidance,
                    )
                    up_mask = repeat(up_mask, "b k h w -> b s k h w", s=S)
                    up_mask = rearrange(up_mask, "b s c h w -> (b s) 1 c h w")

                dense_coords_up_list, dense_coords_depths_up_list = [], []
                for pred_id_ in range(len(dense_coords_down_list)):
                    if use_cotracker:
                        dense_coords_down = rearrange(cotracker_tracks[:, ind : ind + S], "b s (h w) c -> (b s) c h w", b=B, s=S, c=2, w=dense_coords_down_list[-1].shape[-1])
                    else:
                        dense_coords_down = dense_coords_down_list[pred_id_]
                    dense_coords_depths_down = dense_coords_depths_down_list[pred_id_]

                    dense_coords_up = self.original_grid_high_reso + self.upsample_with_mask(
                        (dense_coords_down - self.original_grid_low_reso) * self.stride,
                        up_mask,
                    )
                    dense_coords_up = rearrange(dense_coords_up, "(b s) c h w -> b s c h w", b=B, s=S)
                    dense_coords_up_list.append(dense_coords_up)

                    dense_coords_depths_up = self.upsample_with_mask(
                        dense_coords_depths_down,  # dense_coords_depths_down * (self.d_far-self.d_near) / self.Dz + self.d_near,
                        up_mask,
                    )

                    dense_coords_depths_up = rearrange(dense_coords_depths_up, "(b s) c h w -> b s c h w", b=B, s=S)

                    dense_coords_depths_up_list.append(dense_coords_depths_up)
                if use_cotracker:
                    dense_vis_up = self.upsample_with_mask(rearrange(cotracker_visibility[:, ind : ind + S], "b s (c h w) -> (b s) c h w", b=B, s=S, c=1, w=dense_coords_down_list[-1].shape[-1]), up_mask)
                else:
                    dense_vis_up = self.upsample_with_mask(dense_vis_down, up_mask)
                dense_vis_up = rearrange(dense_vis_up, "(b s) 1 h w -> b s h w", b=B, s=S)

                if use_cotracker:
                    dense_conf_up = self.upsample_with_mask(rearrange(cotracker_conf[:, ind : ind + S], "b s (c h w) -> (b s) c h w", b=B, s=S, c=1, w=dense_coords_down_list[-1].shape[-1]), up_mask)
                else:
                    dense_conf_up = self.upsample_with_mask(dense_conf_down, up_mask)
                dense_conf_up = rearrange(dense_conf_up, "(b s) 1 h w -> b s h w", b=B, s=S)

                dense_coords_up_predicted[:, ind : ind + S] = dense_coords_up_list[-1][
                    :, :S_trimmed
                ]  # dense_coords_out[:, -1]
                dense_coord_depths_up_predicted[:, ind : ind + S] = dense_coords_depths_up_list[-1][
                    :, :S_trimmed
                ]  #  dense_coord_depths_out[:, -1]
                dense_vis_up_predicted[:, ind : ind + S] = dense_vis_up[:, :S_trimmed]
                dense_conf_up_predicted[:, ind : ind + S] = dense_conf_up[:, :S_trimmed]

                if is_train:
                    all_dense_coords_predictions.append(
                        [dense_coord[:, :S_trimmed] for dense_coord in dense_coords_up_list]
                    )  # B I T C H W
                    all_dense_coord_depths_predictions.append(
                        [dense_coord_depths[:, :S_trimmed] for dense_coord_depths in dense_coords_depths_up_list]
                    )  # B I T C H W
                    all_dense_vis_predictions.append(torch.sigmoid(dense_vis_up[:, :S_trimmed]))
                    all_dense_conf_predictions.append(torch.sigmoid(dense_conf_up[:, :S_trimmed]))

        sparse_predictions, dense_predictions = None, None
        self.window_len = self.window_len_copy
        if use_sparse:
            mask = sparse_queried_frames[:, None] <= torch.arange(0, T, device=device)[None, :, None]
            vis_predicted = torch.sigmoid(vis_predicted)
            conf_predicted = torch.sigmoid(conf_predicted)
            sparse_predictions = dict(
                coords=coords_predicted,
                coord_depths=coords_depths_predicted,
                vis=vis_predicted,
                conf=conf_predicted,
                mask=mask,
                sparse_queried_frames=sparse_queried_frames
            )

        if use_dense:
            if not use_cotracker:
                dense_vis_up_predicted = torch.sigmoid(dense_vis_up_predicted)
                dense_conf_up_predicted = torch.sigmoid(dense_conf_up_predicted)
            dense_predictions = dict(
                coords=dense_coords_up_predicted,
                coord_depths=dense_coord_depths_up_predicted,
                vis=dense_vis_up_predicted,
                conf=dense_conf_up_predicted,
            )

        if not is_train:
            return sparse_predictions, dense_predictions, None, selected_indices, [f[:, :-pad] if pad!=0 else f for f in fmaps_pyramid]

        sparse_train_data_dict, dense_train_data_dict = None, None
        if use_sparse:
            mask = sparse_queried_frames[:, None] <= torch.arange(0, T, device=device)[None, :, None]
            sparse_train_data_dict = dict(
                coords=all_coords_predictions,
                coord_depths=all_coord_depths_predictions,
                vis=all_vis_predictions,
                conf=all_conf_predictions,
                mask=mask,
                sparse_queried_frames=sparse_queried_frames,
            )

        if use_dense:
            dense_train_data_dict = dict(
                coords=all_dense_coords_predictions,
                coord_depths=all_dense_coord_depths_predictions,
                vis=all_dense_vis_predictions,
                conf=all_dense_conf_predictions,
                x0y0=(x0, y0),
            )

        train_data = (sparse_train_data_dict, dense_train_data_dict)
        
        return sparse_predictions, dense_predictions, train_data