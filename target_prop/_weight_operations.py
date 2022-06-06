""" NOTE: This is currently unused.

TODO: Not 100% sure I understand what this does
"""
from functools import singledispatch

import torch
from torch import Tensor, nn


@singledispatch
def init_symetric_weights(forward_layer: nn.Module, backward_layer: nn.Module) -> None:
    if any(p.requires_grad for p in forward_layer.parameters()):
        raise NotImplementedError(forward_layer, backward_layer)


init_symmetric_weights = init_symetric_weights


@init_symetric_weights.register
def _weight_b_sym_sequential(forward_layer: nn.Sequential, backward_layer: nn.Sequential) -> None:
    for f_layer, b_layer in zip(forward_layer, backward_layer[::-1]):
        init_symetric_weights(f_layer, b_layer)


@init_symetric_weights.register(nn.Conv2d)
def _weight_b_sym_conv2d(forward_layer: nn.Conv2d, backward_layer: nn.ConvTranspose2d) -> None:
    assert forward_layer.weight.shape == backward_layer.weight.shape
    with torch.no_grad():
        # NOTE: I guess the transposition isn't needed here?
        backward_layer.weight.data = forward_layer.weight.data

    if forward_layer.bias is not None:
        assert backward_layer.bias is not None
        forward_layer.bias.data.zero_()
        backward_layer.bias.data.zero_()
    else:
        assert backward_layer.bias is None


@init_symetric_weights.register(nn.Linear)
def _weight_b_sym_linear(forward_layer: nn.Linear, backward_layer: nn.Linear) -> None:
    assert forward_layer.in_features == backward_layer.out_features
    assert forward_layer.out_features == backward_layer.in_features
    # TODO: Double check that this bias term initialization is ok.
    if forward_layer.bias is None:
        assert backward_layer.bias is None
    else:
        assert backward_layer.bias is not None
        forward_layer.bias.data.zero_()
        backward_layer.bias.data.zero_()

    with torch.no_grad():
        # NOTE: I guess the transposition isn't needed here?
        backward_layer.weight.data = forward_layer.weight.data.t()


from target_prop.networks.resnet import BasicBlock, InvertedBasicBlock


@init_symetric_weights.register(BasicBlock)
def _init_symmetric_weights_residual(forward_layer: BasicBlock, backward_layer: InvertedBasicBlock):
    init_symetric_weights(forward_layer.conv1, backward_layer.conv1),
    init_symetric_weights(forward_layer.bn1, backward_layer.bn1),
    init_symetric_weights(forward_layer.conv2, backward_layer.conv2),
    init_symetric_weights(forward_layer.bn2, backward_layer.bn2),
    if len(forward_layer.shortcut) > 0:  # Non-identity shortcut
        init_symetric_weights(forward_layer.shortcut.conv, backward_layer.shortcut.conv)
        init_symetric_weights(forward_layer.shortcut.bn, backward_layer.shortcut.bn)


@singledispatch
def weight_b_normalize(backward_layer: nn.Module, dx: Tensor, dy: Tensor, dr: Tensor) -> None:
    """TODO: I don't yet understand what this is supposed to do."""
    return
    # raise NotImplementedError(f"No idea what this means atm.")


@weight_b_normalize.register
def linear_weight_b_normalize(
    backward_layer: nn.Linear, dx: Tensor, dy: Tensor, dr: Tensor
) -> None:
    # dy = dy.view(dy.size(0), -1)
    # dx = dx.view(dx.size(0), -1)
    # dr = dr.view(dr.size(0), -1)

    factor = ((dy**2).sum(1)) / ((dx * dr).view(dx.size(0), -1).sum(1))
    factor = factor.mean()

    with torch.no_grad():
        backward_layer.weight.data = factor * backward_layer.weight.data


@weight_b_normalize.register
def conv_weight_b_normalize(
    backward_layer: nn.ConvTranspose2d, dx: Tensor, dy: Tensor, dr: Tensor
) -> None:
    # first technique: same normalization for all out fmaps

    dy = dy.view(dy.size(0), -1)
    dx = dx.view(dx.size(0), -1)
    dr = dr.view(dr.size(0), -1)

    factor = ((dy**2).sum(1)) / ((dx * dr).sum(1))
    factor = factor.mean()
    # factor = 0.5*factor

    with torch.no_grad():
        backward_layer.weight.data = factor * backward_layer.weight.data

    # second technique: fmaps-wise normalization
    """
    dy_square = ((dy.view(dy.size(0), dy.size(1), -1))**2).sum(-1)
    dx = dx.view(dx.size(0), dx.size(1), -1)
    dr = dr.view(dr.size(0), dr.size(1), -1)
    dxdr = (dx*dr).sum(-1)

    factor = torch.bmm(dy_square.unsqueeze(-1), dxdr.unsqueeze(-1).transpose(1,2)).mean(0)

    factor = factor.view(factor.size(0), factor.size(1), 1, 1)

    with torch.no_grad():
        self.b.weight.data = factor*self.b.weight.data
    """
