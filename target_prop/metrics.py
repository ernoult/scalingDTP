from __future__ import annotations
from functools import singledispatch
from typing import Any
import torch
import numpy as np
from torch import nn, Tensor


@singledispatch
@torch.no_grad()
def compute_dist_angle(
    forward_module: Any, backward_module: Any
) -> tuple[float, float]:
    """
    Computes distance and angle between the feedforward and feedback weights
    """
    raise NotImplementedError((forward_module, backward_module))


@compute_dist_angle.register(Tensor)
@compute_dist_angle.register(nn.Parameter)
def _compute_dist_angle_between_weights(
    forward_weight: Tensor, feedback_weight: Tensor
) -> tuple[float, float]:
    F = forward_weight
    G = feedback_weight
    dist = torch.sqrt(((F - G) ** 2).sum() / (F ** 2).sum())

    F_flat = F.flatten(1)
    G_flat = G.flatten(1)
    cos_angle = ((F_flat * G_flat).sum(1)) / torch.sqrt(
        ((F_flat ** 2).sum(1)) * ((G_flat ** 2).sum(1))
    )
    angle_rad = torch.acos(cos_angle).mean()
    angle = torch.rad2deg(angle_rad)
    return dist.item(), angle.item()


@compute_dist_angle.register(nn.Linear)
def _compute_dist_angle_linear(
    forward_module: nn.Linear, backward_module: nn.Linear
) -> tuple[float, float]:
    """
    Computes angle and distance between feedforward and feedback weights of linear layers
    
    Returns angle (degrees) and distance (no real unit I guess) as floats.
    """
    F = forward_module.weight
    G = backward_module.weight.t()
    return compute_dist_angle(F, G)


@compute_dist_angle.register(nn.Conv2d)
def _compute_dist_angle_conv(
    forward_module: nn.Conv2d, backward_module: nn.ConvTranspose2d
) -> tuple[Tensor, Tensor]:
    """
    Computes distance and angle between feedforward and feedback convolutional kernels
    """
    F = forward_module.weight
    G = backward_module.weight
    return compute_dist_angle(F, G)


from target_prop.layers import ConvPoolBlock


@compute_dist_angle.register(ConvPoolBlock)
@compute_dist_angle.register(nn.Sequential)
def _(
    forward_module: ConvPoolBlock, backward_module: nn.Sequential
) -> tuple[Tensor, Tensor]:
    # NOTE: For now, assume that if we're passed a `Sequential`, it will have a
    # nn.Conv2d layer at key 'conv' and that the backward_module will have a
    # `nn.ConvTranspose2d` at key `conv`.

    # IDEA: Could instead start from the front of the forward_module and the back of the
    # backward_module and compute the angles between the corresponding layers?
    # angles = []
    # distances = []
    # for i in range(len(forward_module)):
    #     F_i = forward_module[i]
    #     G_i = backward_module[-1 - i]
    #     dist, angle = compute_dist_angle(F_i, G_i)

    conv2d = forward_module.conv
    convtranspose2d = backward_module.conv
    return compute_dist_angle(conv2d, convtranspose2d)
