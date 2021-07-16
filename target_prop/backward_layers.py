from functools import singledispatch
from torch import nn
import torch
import copy
from typing import TypeVar, Tuple


@singledispatch
def get_backward_equivalent(
    layer: nn.Module, init_symetric_weights: bool = False
) -> nn.Module:
    """Returns the module to be used to compute the 'backward' version of `layer`.
    
    Parameters
    ----------
    layer : nn.Module
        Layer of the forward-pass.
        
    init_symetric_weights : bool, optional
        Wether to initialize the weights of the backward layer based on those of the
        forward layer.


    Returns
    -------
    nn.Module
        Layer to use at the same index as `layer` in the backward-pass model.

    Raises
    ------
    NotImplementedError
        When we don't know what type of layer to use for the backward pass of `layer`.
    """
    raise NotImplementedError(
        f"Don't know what the 'backward' equivalent of {layer} is!"
    )


@get_backward_equivalent.register
def backward_linear(layer: nn.Linear, init_symetric_weights: bool = False) -> nn.Linear:
    # assert layer.bias is None, "Not sure how to handle the bias term"
    backward = type(layer)(
        in_features=layer.out_features,
        out_features=layer.in_features,
        bias=layer.bias is not None,
    )
    if init_symetric_weights:
        with torch.no_grad():
            backward.weight.data = layer.weight.data.t()
    return backward


@get_backward_equivalent.register
def backward_conv(
    layer: nn.Conv2d, init_symetric_weights: bool = False
) -> nn.ConvTranspose2d:
    assert len(layer.kernel_size) == 2
    assert len(layer.stride) == 2
    assert len(layer.padding) == 2
    assert len(layer.output_padding) == 2
    k_h, k_w = layer.kernel_size
    s_h, s_w = layer.stride
    p_h, p_w = layer.padding
    d_h, d_w = layer.dilation
    op_h, op_w = layer.output_padding
    assert k_h == k_w, "only support square kernels for now"
    assert s_h == s_w, "only support square stride for now"
    assert p_h == p_w, "only support square padding for now"
    assert d_h == d_w, "only support square padding for now"
    assert op_h == op_w, "only support square output_padding for now"

    backward = nn.ConvTranspose2d(
        in_channels=layer.out_channels,
        out_channels=layer.in_channels,
        kernel_size=(k_h, k_w),
        # TODO: Not 100% sure about these values:
        stride=(s_h, s_w),
        dilation=d_h,
        padding=(p_h, p_w),
        # TODO: Get this value programmatically.
        output_padding=(s_h - 1, s_w - 1),   
        # output_padding=(op_h + 1, op_w + 1),  # Not sure this will always hold
    )

    if init_symetric_weights:
        # TODO: Not sure the `torch.no_grad` is necessary here:
        with torch.no_grad():
            # NOTE: the transposition isn't needed here (I think)
            backward.weight.data = layer.weight.data
    return backward


@get_backward_equivalent.register(nn.ReLU)
def backward_activation(
    activation_layer: nn.Module, init_symetric_weights: bool = False
) -> nn.Module:
    # TODO: Return an identity function?
    raise NotImplementedError(
        f"Activations aren't invertible by default!"
    )
    # NOTE: It may also be the case that this activation is misplaced (leading to the
    # mis-alignment of layers in contiguous blocks of a network)
    return nn.Identity()
    # return copy.deepcopy(activation_layer)
