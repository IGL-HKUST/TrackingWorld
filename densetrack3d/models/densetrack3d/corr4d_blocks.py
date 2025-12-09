import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


class Conv2dSamePadding(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            # self.padding,
            0,
            self.dilation,
            self.groups,
        )


class Corr4DMLP(nn.Module):
    def __init__(
        self,
        in_channel: int = 49,
        out_channels: tuple = (64, 128, 128),
        kernel_shapes: tuple = (3, 3, 2),
        strides: tuple = (2, 2, 2),
    ):
        super().__init__()
        self.in_channels = [in_channel] + list(out_channels[:-1])
        self.out_channels = out_channels
        self.kernel_shapes = kernel_shapes
        self.strides = strides

        # self.linear_trans = nn.Sequential(
        #     nn.Conv2d(in_channel, 64, 1, 1, 0),
        #     nn.GroupNorm(64 // 16, 64),
        #     nn.ReLU()
        # )

        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2dSamePadding(
                        in_channels=self.in_channels[i],
                        out_channels=self.out_channels[i],
                        kernel_size=self.kernel_shapes[i],
                        stride=self.strides[i],
                    ),
                    # nn.Conv2d(
                    #     in_channels=self.in_channels[i],
                    #     out_channels=self.out_channels[i],
                    #     kernel_size=self.kernel_shapes[i],
                    #     stride=self.strides[i],
                    # ),
                    nn.GroupNorm(out_channels[i] // 16, out_channels[i]),
                    nn.ReLU(),
                )
                for i in range(len(out_channels))
            ]
        )

    def forward(self, x):
        """
        x: (b, h, w, i, j)
        """
        b = x.shape[0]

        out1 = rearrange(x, "b h w i j -> b (i j) h w")
        out2 = rearrange(x, "b h w i j -> b (h w) i j")

        out = torch.cat([out1, out2], dim=0)  # (2 * b) c h w

        # out = self.linear_trans(out)

        for i in range(len(self.out_channels)):
            out = self.conv[i](out)

        out = torch.mean(out, dim=(2, 3))  # (2 * b, out_channels[-1])
        out1, out2 = torch.split(out, b, dim=0)  # (b, out_channels[-1])
        out = torch.cat([out1, out2], dim=-1)  # (b, 2*out_channels[-1])

        return out
