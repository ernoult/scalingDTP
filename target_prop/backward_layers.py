from __future__ import annotations
from functools import singledispatch
from torch import nn
import torch
from typing import OrderedDict, TypeVar, Tuple, overload

from functools import singledispatch
from typing import (
    Any,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import torch
from torch import Tensor, nn


ModuleType = TypeVar("ModuleType", bound=nn.Module, covariant=True)


@runtime_checkable
class Invertible(Protocol):
    input_shape: Tuple[int, ...] = ()
    output_shape: Tuple[int, ...] = ()
    enforce_shapes: bool = False


@overload
def invert(layer: nn.Sequential) -> nn.Sequential:
    ...


@overload
def invert(layer: nn.Module) -> nn.Module:
    ...


@singledispatch
def invert(layer: Union[nn.Module, nn.Sequential]) -> Union[nn.Module, nn.Sequential]:
    """Returns the module to be used to compute the 'backward' version of `layer`.
    
    Parameters
    ----------
    layer : nn.Module
        Layer of the forward-pass.

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


def forward_pre_hook(module: Invertible, inputs: tuple[Tensor, ...]) -> None:
    if isinstance(inputs, tuple) and len(inputs) == 1:
        input = inputs[0]
        _check_input_shape(module, input)


def forward_hook(
    module: Invertible, _: Any, outputs: Tensor | tuple[Tensor, ...]
) -> None:
    if isinstance(outputs, tuple) and len(outputs) == 1:
        output = outputs[0]
        _check_output_shape(module, output)
    elif isinstance(outputs, Tensor):
        output = outputs
        _check_output_shape(module, output)


from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)

register_module_forward_pre_hook(forward_pre_hook)
register_module_forward_hook(forward_hook)


@singledispatch
def add_hooks(module: nn.Module) -> Union[nn.Module, Invertible]:
    # if forward_pre_hook not in module._forward_pre_hooks.values():
    #     module.register_forward_pre_hook(forward_pre_hook)
    # if forward_hook not in module._forward_hooks.values():
    #     module.register_forward_hook(forward_hook)
    return module


# @add_hooks.register
# def add_hooks_to_sequential(module: nn.Sequential) -> Union[nn.Sequential, Invertible]:
#     # if forward_pre_hook not in module._forward_pre_hooks.values():
#     #     module.register_forward_pre_hook(forward_pre_hook)
#     # if forward_hook not in module._forward_hooks.values():
#     #     module.register_forward_hook(forward_hook)
#     # return type(module)(
#     #     OrderedDict((key, add_hooks(value)) for key, value in module._modules.items())
#     # )
#     for layer in module:
#         _ = add_hooks(layer)
#     return module


def _check_input_shape(module: Invertible, x: Tensor) -> None:
    input_shape = tuple(x.shape[1:])
    if not hasattr(module, "enforce_shapes"):
        module.enforce_shapes = False
    if getattr(module, "input_shape", ()) is ():
        module.input_shape = input_shape
    elif module.enforce_shapes and input_shape != module.input_shape:
        raise RuntimeError(
            f"Layer {module} expected individual inputs to have shape {module.input_shape}, but "
            f"got {input_shape} "
        )


def _check_output_shape(module: Invertible, output: Tensor) -> None:
    output_shape = tuple(output.shape[1:])
    if not hasattr(module, "enforce_shapes"):
        module.enforce_shapes = False
    if getattr(module, "output_shape", ()) is ():
        module.output_shape = output_shape
    elif module.enforce_shapes and output_shape != module.output_shape:
        raise RuntimeError(
            f"Outputs of layer {module} have unexpected shape {output_shape} "
            f"(expected {module.output_shape})!"
        )


@invert.register
def invert_linear(layer: nn.Linear) -> nn.Linear:
    # assert layer.bias is None, "Not sure how to handle the bias term"
    backward = type(layer)(
        in_features=layer.out_features,
        out_features=layer.in_features,
        bias=layer.bias is not None,
    )
    return backward


@invert.register
def invert_conv(layer: nn.Conv2d) -> nn.ConvTranspose2d:
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
    return backward


@invert.register(nn.ReLU)
def invert_activation(activation_layer: nn.Module) -> nn.Module:
    # TODO: Return an identity function?
    raise NotImplementedError(f"Activations aren't invertible by default!")
    # NOTE: It may also be the case that this activation is misplaced (leading to the
    # mis-alignment of layers in contiguous blocks of a network)
    return nn.Identity()
    # return copy.deepcopy(activation_layer)


@invert.register(nn.ELU)
def _invert_elu(activation_layer: nn.ELU) -> nn.Module:
    return nn.ELU(alpha=activation_layer.alpha, inplace=False)


@invert.register(nn.AvgPool2d)
def _invert_avgpool(pooling_layer: nn.AvgPool2d) -> nn.Upsample:
    assert pooling_layer.kernel_size == 2, pooling_layer
    assert pooling_layer.stride == 2, pooling_layer
    return nn.Upsample(scale_factor=2)
