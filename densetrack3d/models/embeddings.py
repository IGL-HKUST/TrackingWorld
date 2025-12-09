# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple, Union

import torch
from einops import rearrange


def get_1d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    This function initializes a grid and generates a 2D positional embedding using sine and cosine functions.
    It is a wrapper of get_2d_sincos_pos_embed_from_grid.
    Args:
    - embed_dim: The embedding dimension.
    - grid_size: The grid size.
    Returns:
    - pos_embed: The generated 2D positional embedding.
    """

    grid = torch.arange(grid_size, dtype=torch.float)
    # grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    # grid = torch.stack(grid, dim=0)
    grid = grid.reshape([1, grid_size])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed.reshape(1, grid_size, -1).permute(0, 2, 1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    """
    This function initializes a grid and generates a 2D positional embedding using sine and cosine functions.
    It is a wrapper of get_2d_sincos_pos_embed_from_grid.
    Args:
    - embed_dim: The embedding dimension.
    - grid_size: The grid size.
    Returns:
    - pos_embed: The generated 2D positional embedding.
    """
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float)
    grid_w = torch.arange(grid_size_w, dtype=torch.float)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(0, 3, 1, 2)


def get_3d_sincos_pos_embed(embed_dim: int, grid_size: Union[int, Tuple[int, int, int]]) -> torch.Tensor:
    """
    This function initializes a grid and generates a 2D positional embedding using sine and cosine functions.
    It is a wrapper of get_2d_sincos_pos_embed_from_grid.
    Args:
    - embed_dim: The embedding dimension.
    - grid_size: The grid size.
    Returns:
    - pos_embed: The generated 2D positional embedding.
    """
    if isinstance(grid_size, tuple):
        grid_size_z, grid_size_h, grid_size_w = grid_size
    else:
        grid_size_z, grid_size_h = grid_size_w = grid_size

    grid_z = torch.arange(grid_size_z, dtype=torch.float)
    grid_h = torch.arange(grid_size_h, dtype=torch.float)
    grid_w = torch.arange(grid_size_w, dtype=torch.float)

    grid = torch.meshgrid(grid_z, grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([3, 1, grid_size_z, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(0, 3, 1, 2)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - grid: The grid to generate the embedding from.

    Returns:
    - emb: The generated 2D positional embedding.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=2)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()


def get_2d_embedding(xy: torch.Tensor, C: int, cat_coords: bool = True) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from given coordinates using sine and cosine functions.

    Args:
    - xy: The coordinates to generate the embedding from.
    - C: The size of the embedding.
    - cat_coords: A flag to indicate whether to concatenate the original coordinates to the embedding.

    Returns:
    - pe: The generated 2D positional embedding.
    """
    B, N, D = xy.shape
    assert D == 2

    x = xy[:, :, 0:1]
    y = xy[:, :, 1:2]
    div_term = (torch.arange(0, C, 2, device=xy.device, dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe = torch.cat([pe_x, pe_y], dim=2)  # (B, N, C*3)
    if cat_coords:
        pe = torch.cat([xy, pe], dim=2)  # (B, N, C*3+3)
    return pe


def get_2d_embedding(xy: torch.Tensor, C: int, cat_coords: bool = True) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from given coordinates using sine and cosine functions.

    Args:
    - xy: The coordinates to generate the embedding from.
    - C: The size of the embedding.
    - cat_coords: A flag to indicate whether to concatenate the original coordinates to the embedding.

    Returns:
    - pe: The generated 2D positional embedding.
    """
    ori_shape = xy.shape[:-1]
    D = xy.shape[-1]
    assert D == 2

    x = xy[..., 0:1].reshape(-1, 1)
    y = xy[..., 1:2].reshape(-1, 1)

    N = x.shape[0]

    div_term = (torch.arange(0, C, 2, device=xy.device, dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(N, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(N, C, device=xy.device, dtype=torch.float32)

    pe_x[:, 0::2] = torch.sin(x * div_term)
    pe_x[:, 1::2] = torch.cos(x * div_term)

    pe_y[:, 0::2] = torch.sin(y * div_term)
    pe_y[:, 1::2] = torch.cos(y * div_term)

    pe = torch.cat([pe_x, pe_y], dim=-1)  # (B, N, C*3)

    pe = pe.reshape(*ori_shape, -1)

    if cat_coords:
        pe = torch.cat([xy, pe], dim=-1)  # (B, N, C*3+3)
    return pe


def get_3d_embedding(xyz, C, cat_coords=True):
    B, N, D = xyz.shape
    assert D == 3

    x = xyz[:, :, 0:1]
    y = xyz[:, :, 1:2]
    z = xyz[:, :, 2:3]
    div_term = (torch.arange(0, C, 2, device=xyz.device, dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_z = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe_z[:, :, 0::2] = torch.sin(z * div_term)
    pe_z[:, :, 1::2] = torch.cos(z * div_term)

    pe = torch.cat([pe_x, pe_y, pe_z], dim=2)  # B, N, C*3
    if cat_coords:
        pe = torch.cat([pe, xyz], dim=2)  # B, N, C*3+3
    return pe


# def get_3d_embedding(xyz, C, cat_coords=True):
def get_3d_embedding_custom(xyz: torch.Tensor, C: int, scale=None, range_input=None, custom_C=None) -> torch.Tensor:
    B, N, D = xyz.shape
    assert D == 3

    # if range_input is None:
    #     range_input = 128
    # if scale is None:
    #     scale = 2 * math.pi

    # xyz = xyz * scale / range_input

    x = xyz[:, :, 0:1]
    y = xyz[:, :, 1:2]
    z = xyz[:, :, 2:3]

    assert C % 6 == 0

    if custom_C is not None:
        cx, cy, cz = custom_C[0], custom_C[1], custom_C[2]
    else:
        cx, cy, cz = C // 3, C // 3, C // 3

    pe_x = get_1d_sincos_pos_embed_from_grid(cx, x)  # 1, (B N) D
    pe_y = get_1d_sincos_pos_embed_from_grid(cy, y)
    pe_z = get_1d_sincos_pos_embed_from_grid(cz, z)

    emb = torch.cat([pe_x, pe_y, pe_z], dim=2)  # (1, H*W, D)
    emb = rearrange(emb, "1 (b n) d -> b n d", b=B, n=N)
    return emb

    # div_term = (
    #     torch.arange(0, C, 2, device=xyz.device, dtype=torch.float32) * (1000.0 / C)
    # ).reshape(1, 1, int(C / 2))

    # pe_x = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    # pe_y = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    # pe_z = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)

    # pe_x[:, :, 0::2] = torch.sin(x * div_term)
    # pe_x[:, :, 1::2] = torch.cos(x * div_term)

    # pe_y[:, :, 0::2] = torch.sin(y * div_term)
    # pe_y[:, :, 1::2] = torch.cos(y * div_term)

    # pe_z[:, :, 0::2] = torch.sin(z * div_term)
    # pe_z[:, :, 1::2] = torch.cos(z * div_term)

    # pe = torch.cat([pe_x, pe_y, pe_z], dim=2)  # B, N, C*3
    # if cat_coords:
    #     pe = torch.cat([pe, xyz], dim=2)  # B, N, C*3+3
    # return pe
