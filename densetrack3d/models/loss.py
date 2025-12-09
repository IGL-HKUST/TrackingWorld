# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
import torch.nn.functional as F
from densetrack3d.models.model_utils import reduce_masked_mean
from einops import rearrange, repeat
from jaxtyping import Float, Int64
from torch import Tensor, nn


EPS = 1e-6


def huber_loss(x: Float[Tensor, "*"], y: Float[Tensor, "*"], delta: float = 1.0) -> Float[Tensor, "*"]:
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).float()
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


def track_loss(
    prediction: Float[Tensor, "b i s n c"],
    gt: Float[Tensor, "b s n c"],
    valid: Float[Tensor, "b s n"] = None,
    has_batch_dim: bool = True,
    is_dense: bool = False,
    use_huber_loss: bool = False,
    weight_offset: float = 0.8,
) -> Float[Tensor, ""]:

    if not has_batch_dim:
        prediction = prediction.unsqueeze(0)
        gt = gt.unsqueeze(0)
        if valid is not None:
            valid = valid.unsqueeze(0)

    if is_dense:
        prediction = rearrange(prediction, "b i s c h w -> b i s (h w) c")
        gt = rearrange(gt, "b s c h w -> b s (h w) c")
        if valid is not None:
            valid = rearrange(valid, "b s h w -> b s (h w)")

    I = prediction.shape[1]

    track_loss = 0
    for i in range(I):
        i_weight = weight_offset ** (I - i - 1)

        if use_huber_loss:
            i_loss = huber_loss(prediction[:, i], gt, delta=6.0)
        else:
            i_loss = (prediction[:, i] - gt).abs()  # S, N, 2

        i_loss = torch.mean(i_loss, dim=-1)  # S, N

        if valid is not None:
            track_loss += i_weight * reduce_masked_mean(i_loss, valid)
        else:
            track_loss += i_weight * i_loss.mean()

    return track_loss


def balanced_bce_loss(
    prediction: Float[Tensor, "b s n c"],
    gt: Float[Tensor, "b s n c"],
    valid: Float[Tensor, "b s n"] = None,
) -> Float[Tensor, ""]:

    pos = (gt > 0.95).float()
    neg = (gt < 0.05).float()

    label = pos * 2.0 - 1.0
    a = -label * prediction
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

    if valid is not None:
        pos = pos * valid
        neg = neg * valid

    pos_loss = reduce_masked_mean(loss, pos)
    neg_loss = reduce_masked_mean(loss, neg)

    balanced_loss = pos_loss + neg_loss

    return balanced_loss


def bce_loss(
    prediction: Float[Tensor, "b s n c"],
    gt: Float[Tensor, "b s n c"],
    valid: Float[Tensor, "b s n"] = None,
) -> Float[Tensor, ""]:

    if valid is None:
        loss = F.binary_cross_entropy(prediction, gt)
    else:
        loss = F.binary_cross_entropy(prediction, gt, reduction="none")
        loss = reduce_masked_mean(loss, valid)

    return loss


def confidence_loss(
    tracks: Float[Tensor, "b i s n c"],
    confidence: Float[Tensor, "b s n"],
    target_points: Float[Tensor, "b s n c"],
    visibility: Float[Tensor, "b s n"],
    valid: Float[Tensor, "b s n"] = None,
    expected_dist_thresh: float = 12.0,
    has_batch_dim: bool = True,
    is_dense: bool = False,
) -> Float[Tensor, ""]:
    """Loss for classifying if a point is within pixel threshold of its target."""
    # Points with an error larger than 12 pixels are likely to be useless; marking
    # them as occluded will actually improve Jaccard metrics and give
    # qualitatively better results.

    # if len(tracks.shape) == 5:
    #     B, I, S, N, C = tracks.shape
    # else:
    #     I, S, N, C = tracks.shape
    #     tracks = tracks.unsqueeze(0)
    #     confidence = confidence.unsqueeze(0)
    #     target_points = target_points.unsqueeze(0)
    #     visibility = visibility.unsqueeze(0)

    if not has_batch_dim:
        tracks = tracks.unsqueeze(0)
        confidence = confidence.unsqueeze(0)
        target_points = target_points.unsqueeze(0)
        visibility = visibility.unsqueeze(0)
        if valid is not None:
            valid = valid.unsqueeze(0)

    if is_dense:
        tracks = rearrange(tracks, "b i s c h w -> b i s (h w) c")
        target_points = rearrange(target_points, "b s c h w -> b s (h w) c")
        confidence = rearrange(confidence, "b s h w -> b s (h w)")
        visibility = rearrange(visibility, "b s h w -> b s (h w)")
        if valid is not None:
            valid = rearrange(valid, "b s h w -> b s (h w)")

    if not visibility.dtype == torch.bool:
        visibility = (visibility > 0.9).bool()

    err = torch.sum((tracks[:, -1].detach() - target_points) ** 2, dim=-1)
    conf_gt = (err <= expected_dist_thresh**2).float()
    logprob = F.binary_cross_entropy(confidence, conf_gt, reduction="none")
    logprob *= visibility.float()

    if valid is not None:
        logprob = reduce_masked_mean(logprob, valid)
    else:
        logprob = logprob.mean()

    return logprob
