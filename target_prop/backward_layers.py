from __future__ import annotations

from functools import singledispatch
from typing import OrderedDict, Protocol, TypeVar, runtime_checkable

from torch import Tensor, nn

ModuleType = TypeVar("ModuleType", bound=nn.Module)


@runtime_checkable
class Invertible(Protocol):
    """A Module that is "easy to invert" since it has known input and output shapes.

    It's easier to mark modules as invertible in-place than to create new subclass for every single
    nn.Module class that we want to potentially use in the forward net.
    NOTE @lebrice: This is what I was doing before, using a kind of `InvertibleMixin`. Using a
    Protocol for this isn't 100% necessary. Just having fun with the new structural subtyping.
    """

    input_shape: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()
    enforce_shapes: bool = False


@singledispatch
def invert(layer: nn.Module | Invertible) -> nn.Module:
    """Returns the module to be used to compute the pseudoinverse of `layer`.

    NOTE: All concrete handlers below usually assume that a layer has been marked as 'invertible'.
    This is usually

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
    raise NotImplementedError(f"Don't know what the 'backward' equivalent of {layer} is!")


@invert.register
def invert_linear(layer: nn.Linear) -> nn.Linear:
    # NOTE: Not sure how to handle the bias term
    backward = type(layer)(
        in_features=layer.out_features,
        out_features=layer.in_features,
        bias=layer.bias is not None,
    )
    return backward


@invert.register(nn.Sequential)
def invert_sequential(module: nn.Sequential) -> nn.Sequential:
    """Returns a Module that can be used to compute or approximate the inverse
    operation of `self`.

    NOTE: In the case of Sequential, the order of the layers in the returned network
    is reversed compared to the input.
    """
    # assert module.input_shape and module.output_shape, "Use the net before inverting."
    # NOTE: Inverting a ResNet (which subclasses Sequential) doesn't try to create another ResNet!
    # It just returns a Sequential.
    return nn.Sequential(
        OrderedDict((name, invert(module)) for name, module in list(module._modules.items())[::-1]),
    )


@invert.register(nn.Identity)
def invert_identity(module: nn.Identity) -> nn.Identity:
    return nn.Identity()


@invert.register
def invert_conv(layer: nn.Conv2d) -> nn.ConvTranspose2d:
    assert len(layer.kernel_size) == 2
    assert len(layer.stride) == 2
    assert len(layer.padding) == 2
    assert len(layer.output_padding) == 2
    k_h, k_w = layer.kernel_size
    s_h, s_w = layer.stride
    assert not isinstance(layer.padding, str)
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
        bias=layer.bias is not None,
        # output_padding=(op_h + 1, op_w + 1),  # Not sure this will always hold
    )
    return backward


@invert.register(nn.ReLU)
def invert_relu(activation_layer: nn.Module) -> nn.Module:
    return nn.ReLU(inplace=False)


@invert.register(nn.ELU)
def _invert_elu(activation_layer: nn.ELU) -> nn.Module:
    return nn.ELU(alpha=activation_layer.alpha, inplace=False)


def check_shapes_hook(
    module: Invertible,
    inputs: Tensor | tuple[Tensor, ...],
    outputs: Tensor | tuple[Tensor, ...],
) -> None:
    """Hook that sets the `input_shape` and `output_shape` attributes on the layers if not present.

    Also, if the `enforce_shapes` attribute is set to `True` on `module`, and the shapes don't match
    with their respective attributes, this will raise an error.
    """
    if isinstance(inputs, tuple) and len(inputs) == 1:
        inputs = inputs[0]
    if isinstance(outputs, tuple) and len(outputs) == 1:
        outputs = outputs[0]

    if not isinstance(inputs, Tensor) or not isinstance(outputs, Tensor):
        # For now we can only add this hook on modules that take one tensor in and return one
        # tensor out.
        return

    # Don't consider the batch dimension.
    input_shape = tuple(inputs.shape[1:])
    output_shape = tuple(outputs.shape[1:])
    # Set the `input_shape`, `output_shape`, `enforce_shapes` attributes if not present:
    if not hasattr(module, "enforce_shapes"):
        module.enforce_shapes = False
    # NOTE: not using hasattr since some layers might have type annotations with empty tuple or smt.
    if getattr(module, "input_shape", ()) == ():
        module.input_shape = input_shape
    if getattr(module, "output_shape", ()) == ():
        module.output_shape = output_shape

    # NOTE: This isinstance check works with the `Invertible` procol since the attributes are there.
    assert isinstance(module, Invertible)

    if module.enforce_shapes:
        if input_shape != module.input_shape:
            raise RuntimeError(
                f"Layer {module} expected individual inputs to have shape {module.input_shape}, but "
                f"got {input_shape} "
            )
        if output_shape != module.output_shape:
            raise RuntimeError(
                f"Outputs of layer {module} have unexpected shape {output_shape} "
                f"(expected {module.output_shape})!"
            )


@singledispatch
def mark_as_invertible(module: ModuleType) -> ModuleType | Invertible:
    """Makes the module easier to "invert" by adding hooks that set the
    `input_shape` and `output_shape` attributes. Modifies the module in-place.
    """
    if check_shapes_hook not in module._forward_hooks.values():
        module.register_forward_hook(check_shapes_hook)
    return module


@mark_as_invertible.register(nn.Sequential)
def _mark_sequential_as_invertible(
    module: nn.Sequential,
) -> nn.Sequential | Invertible:
    if check_shapes_hook not in module._forward_hooks.values():
        module.register_forward_hook(check_shapes_hook)
    for layer in module:
        # NOTE:
        _ = mark_as_invertible(layer)
    return module
