from __future__ import annotations
from functools import singledispatch

import torch
import numpy as np
from torch import nn, Tensor


@singledispatch
def compute_dist_angle(
    forward_module: nn.Module, backward_module: nn.Module
) -> tuple[Tensor, Tensor]:
    """
    Computes angle and distance between the feedforward and feedback weights
    """
    raise NotImplementedError((forward_module, backward_module))


@compute_dist_angle.register(nn.Linear)
def _compute_dist_angle_linear(
    forward_module: nn.Linear, backward_module: nn.Linear
) -> tuple[Tensor, Tensor]:
    """
    Computes angle and distance between feedforward and feedback weights of linear layers
    """
    F = forward_module.weight
    G = backward_module.weight.t()

    dist = torch.sqrt(((F - G) ** 2).sum() / (F ** 2).sum())

    F_flat = torch.reshape(F, (F.size(0), -1))
    G_flat = torch.reshape(G, (G.size(0), -1))
    cos_angle = ((F_flat * G_flat).sum(1)) / torch.sqrt(
        ((F_flat ** 2).sum(1)) * ((G_flat ** 2).sum(1))
    )
    angle = (180.0 / np.pi) * (
        torch.acos(cos_angle).mean()
    )  # Note: removed a ".item()"
    return dist, angle


@compute_dist_angle.register(nn.Conv2d)
def _compute_dist_angle_conv(
    forward_module: nn.Conv2d, backward_module: nn.ConvTranspose2d
) -> tuple[Tensor, Tensor]:
    """
    Computes distance and angle between feedforward and feedback convolutional kernels
    """
    F = forward_module.weight
    G = backward_module.weight
    dist = torch.sqrt(((F - G) ** 2).sum() / (F ** 2).sum())

    F_flat = torch.reshape(F, (F.size(0), -1))
    G_flat = torch.reshape(G, (G.size(0), -1))
    cos_angle = ((F_flat * G_flat).sum(1)) / torch.sqrt(
        ((F_flat ** 2).sum(1)) * ((G_flat ** 2).sum(1))
    )
    angle = (180.0 / np.pi) * (torch.acos(cos_angle).mean())  # Note: Removed '.item()'

    return dist, angle


from target_prop.layers import ConvPoolBlock, ConvTransposePoolBlock, Sequential


@compute_dist_angle.register(ConvPoolBlock)
@compute_dist_angle.register(Sequential)
def _(
    forward_module: ConvPoolBlock, backward_module: ConvTransposePoolBlock
) -> tuple[Tensor, Tensor]:
    conv2d = forward_module.conv
    convtranspose2d = backward_module.conv
    return compute_dist_angle(conv2d, convtranspose2d)
