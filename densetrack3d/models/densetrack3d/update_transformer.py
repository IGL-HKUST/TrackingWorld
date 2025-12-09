import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from densetrack3d.models.densetrack3d.blocks import Attention, AttnBlock, Mlp
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float, Int64, Shaped
from torch import Tensor, nn


class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        num_blocks=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        num_virtual_tracks=64,
        flash=False,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        # self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(torch.randn(1, num_virtual_tracks, 1, hidden_size))
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                    flash=flash,
                )
                for _ in range(num_blocks)
            ]
        )

        self.space_virtual_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                    flash=flash,
                )
                for _ in range(num_blocks)
            ]
        )
        self.space_point2virtual_blocks = nn.ModuleList(
            [
                CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio, flash=flash)
                for _ in range(num_blocks)
            ]
        )
        self.space_virtual2point_blocks = nn.ModuleList(
            [
                CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio, flash=flash)
                for _ in range(num_blocks)
            ]
        )

        self.space_local_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                    flash=flash,
                )
                for _ in range(num_blocks)
            ]
        )

        self.local_size = 6

        # assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def local_attention(self, point_tokens, dH, dW, j, B, T):
        # NOTE local attention
        local_size = self.local_size
        shift_size = local_size // 2

        orig_dH, orig_dW = dH, dW

        local_patches = rearrange(point_tokens, "b (h w) c -> b c h w", h=dH, w=dW)
        pad_h = local_size - local_patches.shape[-2] % local_size if local_patches.shape[-2] % local_size != 0 else 0
        pad_w = local_size - local_patches.shape[-1] % local_size if local_patches.shape[-1] % local_size != 0 else 0

        if pad_h > 0 or pad_w > 0:
            local_patches = F.pad(local_patches, (0, pad_w, 0, pad_h), "constant", 0)
            dH, dW = local_patches.shape[-2], local_patches.shape[-1]
        # if i % 2 == 1:
        #     use_shift = True
        # local_patches = torch.roll(local_patches, )
        # copy_local_patches = local_patches.clone()
        local_patches = F.unfold(local_patches, kernel_size=local_size, stride=local_size)  # (B T) C (H W)
        local_patches = rearrange(local_patches, "b (c p1 p2) l -> (b l) (p1 p2) c", p1=local_size, p2=local_size)

        attn_mask = local_patches.detach().abs().sum(-1) > 0  # B N

        # NOTE add embedding here
        # local_embed = self.space_local_emb.unsqueeze(0).repeat(local_patches.shape[0], 1, 1)
        # local_patches = local_patches + local_embed
        # breakpoint()
        local_patches = self.space_local_blocks[j](local_patches, mask=attn_mask)

        # breakpoint()
        local_patches = rearrange(
            local_patches,
            "(b h w) (p1 p2) c -> b c h p1 w p2",
            b=B * T,
            h=dH // local_size,
            w=dW // local_size,
            p1=local_size,
            p2=local_size,
        )

        # breakpoint()
        local_patches = local_patches.contiguous().view(B * T, -1, dH, dW)

        local_patches = local_patches[:, :, :orig_dH, :orig_dW]
        point_tokens = rearrange(local_patches, "b c h w -> b (h w) c")

        return point_tokens

    def forward(
        self,
        input_tensor: Float[Tensor, "b t n c_in"],
        attn_mask: Bool[Tensor, "b t n"] = None,
        n_sparse: int = 256,
        dH: int = 4,
        dW: int = 4,
        use_efficient_global_attn: bool = False,
    ) -> Float[Tensor, "b t n c_out"]:

        if use_efficient_global_attn:
            assert n_sparse > 0

        B, T, *_ = input_tensor.shape

        real_tokens = self.input_transform(input_tensor)
        virtual_tokens = repeat(self.virual_tracks, "1 n 1 c -> b n t c", b=B, t=T)
        virtual_tokens = rearrange(virtual_tokens, "b n t c -> b t n c")

        # self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([real_tokens, virtual_tokens], dim=2)
        N_total = tokens.shape[2]

        # j = 0

        tokens = rearrange(tokens, "b t n c -> (b n) t c")
        attn_mask = rearrange(attn_mask, "b t n -> (b t) n")

        for lvl in range(self.num_blocks):
            # tokens = rearrange(tokens, 'b t n c -> (b n) t c')
            tokens = self.time_blocks[lvl](tokens)

            tokens = rearrange(tokens, "(b n) t c -> (b t) n c", b=B, t=T)

            virtual_tokens = tokens[:, N_total - self.num_virtual_tracks :]
            real_tokens = tokens[:, : N_total - self.num_virtual_tracks]
            sparse_tokens = real_tokens[:, :n_sparse]
            dense_tokens = real_tokens[:, n_sparse:]

            # NOTE global attention

            if use_efficient_global_attn:
                sparse_mask = attn_mask[:, :n_sparse]
                virtual_tokens = self.space_virtual2point_blocks[lvl](virtual_tokens, sparse_tokens, mask=sparse_mask)
            else:
                virtual_tokens = self.space_virtual2point_blocks[lvl](virtual_tokens, real_tokens, mask=attn_mask)

            virtual_tokens = self.space_virtual_blocks[lvl](virtual_tokens)

            # NOTE local attention
            if dH > 0 and dW > 0:
                dense_tokens = self.local_attention(dense_tokens, dH, dW, lvl, B, T)
                real_tokens = torch.cat([sparse_tokens, dense_tokens], dim=1)

            real_tokens = self.space_point2virtual_blocks[lvl](real_tokens, virtual_tokens, mask=attn_mask)

            tokens = torch.cat([real_tokens, virtual_tokens], dim=1)

            if lvl == self.num_blocks - 1:  # NOTE last layer: no virtual tokens
                tokens = rearrange(tokens, "(b t) n c -> b t n c", b=B, t=T)
            else:
                tokens = rearrange(tokens, "(b t) n c -> (b n) t c", b=B, t=T)
            # tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)  # (B T) N C -> B N T C

        # tokens = rearrange(tokens, '(b n) t c -> b t n c',)

        real_tokens = tokens[:, :, : N_total - self.num_virtual_tracks]
        flow = self.flow_head(real_tokens)
        return flow


class CrossAttnBlock(nn.Module):
    def __init__(self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = Attention(
            hidden_size, context_dim=context_dim, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, context, mask=None):
        if mask is not None:
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(-1, self.cross_attn.heads, -1, context.shape[1])
            else:
                mask = mask[:, None, None].expand(-1, self.cross_attn.heads, x.shape[1], -1)

            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.cross_attn(self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x
