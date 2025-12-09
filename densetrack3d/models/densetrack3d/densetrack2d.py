
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


class DenseTrack2D(nn.Module):
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
        self.window_len = window_len
        self.stride = stride
        self.hidden_dim = 256
        self.latent_dim = 128
        self.upsample_factor = upsample_factor
        self.add_space_attn = add_space_attn
        self.fnet = BasicEncoder(input_dim=3, output_dim=self.latent_dim)

        self.num_virtual_tracks = num_virtual_tracks
        self.model_resolution = model_resolution
        self.input_dim = 1032
        self.updateformer = EfficientUpdateFormer(
            num_blocks=6,
            input_dim=self.input_dim,
            hidden_size=384,
            output_dim=self.latent_dim + 2,
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
            upsample_factor=self.upsample_factor
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
            # self.grad_context = torch.no_grad
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

    def upsample_with_mask(
        self, inp: Float[Tensor, "b c h_down w_down"], mask: Float[Tensor, "b 1 k h_up w_up"]
    ) -> Float[Tensor, "b c h_up w_up"]:
        """Upsample flow field [H/P, W/P, 2] -> [H, W, 2] using convex combination"""
        H, W = inp.shape[-2:]

        up_inp = F.unfold(
            inp, [self.upsample_kernel_size, self.upsample_kernel_size], padding=(self.upsample_kernel_size - 1) // 2
        )
        up_inp = rearrange(up_inp, "b c (h w) -> b c h w", h=H, w=W)
        up_inp = F.interpolate(up_inp, scale_factor=self.stride, mode="nearest")
        up_inp = rearrange(
            up_inp, "b (c i j) h w -> b c (i j) h w", i=self.upsample_kernel_size, j=self.upsample_kernel_size
        )

        up_inp = torch.sum(mask * up_inp, dim=2)
        return up_inp


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
        coords: Float[Tensor, "b t n c"],
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

        for iteration in range(iters):
            coords = coords.detach()  # B S N 3

            # NOTE Prepare input to transformer
            fcorrs = self.get_4dcorr_features(fmaps_pyramid, coords, supp_track_feat)


            # Get the 2D flow embeddings
            flows_2d = coords - coords[:, 0:1]
            flows_2d_emb = get_2d_embedding(flows_2d, 64, cat_coords=True)  # N S E

            transformer_input = torch.cat(
                [
                    flows_2d_emb,
                    flows_2d.repeat(1,1,1,2),
                    fcorrs,
                    track_feat,
                    track_mask_vis,
                ],
                dim=-1,
            )
            x = transformer_input + pos_emb[:, None] + self.time_emb[:, :, None]

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
            delta_feat = self.track_feat_updater(self.norm(rearrange(delta[..., 2:], "b t n c -> (b t n) c")))
            delta_feat = rearrange(delta_feat, "(b t n) c -> b t n c", b=B, t=S)  ###########################

            # NOTE Update
            track_feat = track_feat + delta_feat
            coords = coords + delta_coords
            ###########################

            coord_preds.append(coords.clone())

        vis_pred = self.vis_predictor(track_feat).squeeze(-1)  # b s n
        conf_pred = self.conf_predictor(track_feat).squeeze(-1)  # b s n

        return coord_preds, vis_pred, conf_pred, track_feat

    def extract_features(self, video: VideoType) -> tuple[Float[Tensor, "*"], ...]:

        B, T = video.shape[:2]
        fmaps, higher_fmaps, lower_fmaps = self.fnet(
            rearrange(video, "b t c h w -> (b t) c h w"), return_intermediate=True
        )

        fmaps = rearrange(fmaps, "(b t) c h w -> b t c h w", b=B, t=T)
        higher_fmaps = rearrange(higher_fmaps, "(b t) c h w -> b t c h w", b=B, t=T)
        lower_fmaps = rearrange(lower_fmaps, "(b t) c h w -> b t c h w", b=B, t=T)

        return fmaps, higher_fmaps, lower_fmaps

    def prepare_sparse_queries(
        self,
        sparse_queries: Float[Tensor, "b n c"],
        fmaps: VideoType,
        fmaps_pyramid: tuple[VideoType, ...],
    ) -> tuple:

        S = self.window_len
        B, _, _, fH, fW = fmaps.shape
        device = fmaps.device

        self.N_sparse = N_sparse = sparse_queries.shape[1]
        sparse_queried_frames = sparse_queries[:, :, 0].long()

        # NOTE normalize queries
        sparse_queried_coords = sparse_queries[..., 1:4].clone()
        sparse_queried_coords[..., :2] = sparse_queried_coords[..., :2] / self.stride

        # We compute track features # FIXME only get 2d coord
        track_feat, supp_track_feats_pyramid = self.get_track_feat(
            fmaps,
            fmaps_pyramid,
            sparse_queried_frames,
            sparse_queried_coords[..., :2],
        )
        track_feat = repeat(track_feat, "b 1 n c -> b s n c", s=S)

        coords_init = sparse_queried_coords[..., :2].reshape(B, 1, N_sparse, 2).expand(B, S, N_sparse, 2).float()
        vis_init = torch.ones((B, S, N_sparse, 1), device=device).float() * 10
        conf_init = torch.ones((B, S, N_sparse, 1), device=device).float() * 10

        return (
            coords_init,
            vis_init,
            conf_init,
            track_feat,
            supp_track_feats_pyramid,
            sparse_queried_frames,
        )

    def prepare_dense_queries(
        self,
        coords_init: Float[Tensor, "b t n c"],
        vis_init: Float[Tensor, "b t n"],
        conf_init: Float[Tensor, "b t n"],
        track_feat: Float[Tensor, "b t n c"],
        supp_track_feats_pyramid: tuple[Float[Tensor, "*"], ...],
        fmaps: VideoType,
        fmaps_pyramid: tuple[VideoType, ...],
        # depth_init_downsample: Float[Tensor, "b 1 h w"],
        is_train: bool = False,
    ) -> tuple:

        S = self.window_len
        B, _, _, fH, fW = fmaps.shape
        device = fmaps.device

        if is_train:
            dH, dW = 15, 20
            y0 = np.random.randint(0, fH - dH, size=B)
            x0 = np.random.randint(0, fW - dW, size=B)
        else:
            # dH, dW = self.model_resolution[0] // (self.stride), self.model_resolution[1] // (self.stride)
            dH, dW = self.input_reso[0] // (self.stride), self.input_reso[1] // (self.stride)
            x0, y0 = [0] * B, [0] * B
        self.dH, self.dW = dH, dW

        dense_grid_2d = (
            get_grid(dH, dW, normalize=False, device=fmaps.device).reshape(-1, 2).unsqueeze(0).repeat(B, 1, 1)
        )  # B, (H, W) 2
        for b in range(B):
            dense_grid_2d[b, :, 0] += x0[b]
            dense_grid_2d[b, :, 1] += y0[b]

        dense_coords_init = repeat(dense_grid_2d, "b n c -> b s n c", s=S)
        dense_vis_init = torch.ones((B, S, dH * dW, 1), device=device).float() * 10
        dense_conf_init = torch.ones((B, S, dH * dW, 1), device=device).float() * 10

        dense_track_feat, dense_supp_track_feats_pyramid = self.get_dense_track_feat(
            fmaps,
            fmaps_pyramid,
            dense_grid_2d,
        )
        dense_track_feat = repeat(dense_track_feat, "b n c -> b s n c", s=S)

        dense_grid_2d_up = (
            get_grid(dH * self.stride, dW * self.stride, normalize=False, device=fmaps.device)
            .reshape(-1, 2)
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )
        for b in range(B):
            dense_grid_2d_up[b, :, 0] += x0[b] * self.stride
            dense_grid_2d_up[b, :, 1] += y0[b] * self.stride

        self.original_grid_low_reso = rearrange(dense_grid_2d.clone(), "b (h w) c -> b c h w", h=dH, w=dW)
        self.original_grid_high_reso = rearrange(
            dense_grid_2d_up.clone(), "b (h w) c -> b c h w", h=dH * self.stride, w=dW * self.stride
        )

        coords_init = smart_cat(
            coords_init, dense_coords_init, dim=2
        )  # torch.cat([coords_init, dense_coords_init], dim=2)
        vis_init = smart_cat(vis_init, dense_vis_init, dim=2)  # torch.cat([vis_init, dense_vis_init], dim=2)
        conf_init = smart_cat(conf_init, dense_conf_init, dim=2)  # torch.cat([vis_init, dense_vis_init], dim=2)
        track_feat = smart_cat(track_feat, dense_track_feat, dim=2)  # torch.cat([track_feat, dense_track_feat], dim=2)
        supp_track_feats_pyramid = [
            smart_cat(sf, dense_sf, dim=1)
            for sf, dense_sf in zip(supp_track_feats_pyramid, dense_supp_track_feats_pyramid)
        ]

        return (coords_init, vis_init, conf_init, track_feat, supp_track_feats_pyramid, (x0, y0))

    def forward(
        self,
        video: VideoType,
        sparse_queries: Float[Tensor, "b n c"] = None,
        iters: int = 4,
        is_train: bool = False,
        use_dense: bool = True,
        use_efficient_global_attn: bool = False,
    ) -> tuple[dict | None, dict | None, tuple[dict | None, dict | None] | None]:
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3, H, W]): input videos.
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
        if is_train:
            pad = 0
        else:
            pad = (S - T % S) % S  # We don't want to pad if T % S == 0

        if pad > 0:
            video = F.pad(video.reshape(B, 1, T, C * H * W), (0, 0, 0, pad), "replicate").reshape(B, -1, C, H, W)
            
        fmaps, higher_fmaps, lower_fmaps = self.extract_features(video)
        fmaps_pyramid = [higher_fmaps, fmaps, lower_fmaps]
        fH, fW = fmaps.shape[-2:]

        self.dH, self.dW = 0, 0
        self.N_sparse = 0

        coords_init, vis_init, conf_init, track_feat, supp_track_feats_pyramid = (
            None,
            None,
            None,
            None,
            [None] * 3,
        )
        if use_sparse:
            (
                coords_init,
                vis_init,
                conf_init,
                track_feat,
                supp_track_feats_pyramid,
                sparse_queried_frames,
            ) = self.prepare_sparse_queries(sparse_queries, fmaps, fmaps_pyramid)

            # We store our predictions here
            coords_predicted = torch.zeros((B, ori_T, self.N_sparse, 2), device=device)
            vis_predicted = torch.zeros((B, ori_T, self.N_sparse), device=device)
            conf_predicted = torch.zeros((B, ori_T, self.N_sparse), device=device)
            all_coords_predictions, all_vis_predictions, all_conf_predictions = (
                [],
                [],
                [],
            )

        if use_dense:
            coords_init, vis_init, conf_init, track_feat, supp_track_feats_pyramid, (x0, y0) = (
                self.prepare_dense_queries(
                    coords_init,
                    vis_init,
                    conf_init,
                    track_feat,
                    supp_track_feats_pyramid,
                    fmaps,
                    fmaps_pyramid,
                    is_train,
                )
            )

            dense_coords_up_predicted = torch.zeros(
                (B, ori_T, 2, self.dH * self.stride, self.dW * self.stride), device=device
            )
            dense_vis_up_predicted = torch.zeros(
                (B, ori_T, self.dH * self.stride, self.dW * self.stride), device=device
            )
            dense_conf_up_predicted = torch.zeros(
                (B, ori_T, self.dH * self.stride, self.dW * self.stride), device=device
            )

            (
                all_dense_coords_predictions,
                all_dense_vis_predictions,
                all_dense_conf_predictions,
            ) = ([], [], [])
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
                last_vis = vis[:, -overlap:].clone()[..., None]
                last_conf = conf[:, -overlap:].clone()[..., None]

                coords_prev = torch.nn.functional.pad(
                    last_coords,
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

            coords, vis, conf, track_feat_updated = self.forward_window(
                fmaps_pyramid=[f[:, ind : ind + S] for f in fmaps_pyramid],
                coords=coords_init,
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
                vis_predicted[:, ind : ind + S] = vis[:, :S_trimmed, : self.N_sparse]
                conf_predicted[:, ind : ind + S] = conf[:, :S_trimmed, : self.N_sparse]

                if is_train:
                    all_coords_predictions.append(
                        [(coord[:, :S_trimmed, : self.N_sparse] * self.stride) for coord in coords]
                    )
                    all_vis_predictions.append(torch.sigmoid(vis[:, :S_trimmed, : self.N_sparse]))
                    all_conf_predictions.append(torch.sigmoid(conf[:, :S_trimmed, : self.N_sparse]))

            if use_dense:

                dense_coords_down_list = [
                    rearrange(coord_[:, :, self.N_sparse :], "b s (h w) c -> (b s) c h w", h=self.dH, w=self.dW)
                    for coord_ in coords
                ]
                dense_vis_down = rearrange(
                    vis[:, :, self.N_sparse :], "b s (h w) -> (b s) 1 h w", h=self.dH, w=self.dW
                )
                dense_conf_down = rearrange(
                    conf[:, :, self.N_sparse :], "b s (h w) -> (b s) 1 h w", h=self.dH, w=self.dW
                )

                if up_mask is None:
                    flow_guidance = (dense_coords_down_list[-1] - self.original_grid_low_reso) / self.Dz
                    flow_guidance = rearrange(flow_guidance, "(b s) c h w -> b s c h w", b=B, s=S)
                    upsample_featmap = rearrange(
                        track_feat_updated[:, 0, self.N_sparse :], "b (h w) c -> b c h w", h=self.dH, w=self.dW
                    )

                    up_mask = self.upsample_transformer(
                        feat_map=upsample_featmap,
                        flow_map=flow_guidance,
                    )
                    up_mask = repeat(up_mask, "b k h w -> b s k h w", s=S)
                    up_mask = rearrange(up_mask, "b s c h w -> (b s) 1 c h w")

                dense_coords_up_list, dense_coords_depths_up_list = [], []
                for pred_id_ in range(len(dense_coords_down_list)):
                    dense_coords_down = dense_coords_down_list[pred_id_]

                    dense_coords_up = self.original_grid_high_reso + self.upsample_with_mask(
                        (dense_coords_down - self.original_grid_low_reso) * self.stride,
                        up_mask,
                    )
                    dense_coords_up = rearrange(dense_coords_up, "(b s) c h w -> b s c h w", b=B, s=S)
                    dense_coords_up_list.append(dense_coords_up)


                dense_vis_up = self.upsample_with_mask(dense_vis_down, up_mask)
                dense_vis_up = rearrange(dense_vis_up, "(b s) 1 h w -> b s h w", b=B, s=S)

                dense_conf_up = self.upsample_with_mask(dense_conf_down, up_mask)
                dense_conf_up = rearrange(dense_conf_up, "(b s) 1 h w -> b s h w", b=B, s=S)

                dense_coords_up_predicted[:, ind : ind + S] = dense_coords_up_list[-1][
                    :, :S_trimmed
                ]  # dense_coords_out[:, -1]
                dense_vis_up_predicted[:, ind : ind + S] = dense_vis_up[:, :S_trimmed]
                dense_conf_up_predicted[:, ind : ind + S] = dense_conf_up[:, :S_trimmed]

                if is_train:
                    all_dense_coords_predictions.append(
                        [dense_coord[:, :S_trimmed] for dense_coord in dense_coords_up_list]
                    )  # B I T C H W
                    all_dense_vis_predictions.append(torch.sigmoid(dense_vis_up[:, :S_trimmed]))
                    all_dense_conf_predictions.append(torch.sigmoid(dense_conf_up[:, :S_trimmed]))

        sparse_predictions, dense_predictions = None, None

        if use_sparse:
            vis_predicted = torch.sigmoid(vis_predicted)
            conf_predicted = torch.sigmoid(conf_predicted)
            sparse_predictions = dict(
                coords=coords_predicted,
                vis=vis_predicted,
                conf=conf_predicted,
            )

        if use_dense:
            dense_vis_up_predicted = torch.sigmoid(dense_vis_up_predicted)
            dense_conf_up_predicted = torch.sigmoid(dense_conf_up_predicted)
            dense_predictions = dict(
                coords=dense_coords_up_predicted,
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
                vis=all_vis_predictions,
                conf=all_conf_predictions,
                mask=mask,
            )

        if use_dense:
            dense_train_data_dict = dict(
                coords=all_dense_coords_predictions,
                vis=all_dense_vis_predictions,
                conf=all_dense_conf_predictions,
                x0y0=(x0, y0),
            )

        train_data = (sparse_train_data_dict, dense_train_data_dict)

        return sparse_predictions, dense_predictions, train_data
