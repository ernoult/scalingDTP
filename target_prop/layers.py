from __future__ import annotations

from collections import OrderedDict, deque
from typing import (
    Tuple,
    Type,
    TypeVar,
)

import torch
from torch import Tensor, nn
from torch.nn.modules.conv import _size_2_t
from torch.nn.modules.pooling import AdaptiveMaxPool2d, _size_any_t

from .backward_layers import invert, Invertible, add_hooks

ModuleType = TypeVar("ModuleType", bound=nn.Module, covariant=True)


def forward_each(module: nn.Sequential, xs: list[Tensor]) -> list[Tensor]:
    """Gets the outputs of every layer, given inputs for each layer `xs`.

    Parameters
    ----------
    x : List[Tensor]
        A list of tensors, one per layer, which will be used as the inputs for each
        forward layer.

    Returns
    -------
    List[Tensor]
        The outputs of each forward layer.
    """
    xs = list(xs) if not isinstance(xs, (list, tuple)) else xs
    assert len(xs) == len(module)
    return [layer(x_i) for layer, x_i in zip(module, xs)]


def forward_all(
    module: nn.Sequential, x: Tensor, allow_grads_between_layers: bool = False,
) -> list[Tensor]:
    """Gets the outputs of all forward layers for the given input. 
    
    Parameters
    ----------
    x : Tensor
        Input tensor.

    allow_grads_between_layers : bool, optional
        Wether to allow gradients to flow from one layer to the next.
        When `False` (default), outputs of each layer are detached before being
        fed to the next layer.

    Returns
    -------
    List[Tensor]
        The outputs of each forward layer.
    """
    activations: list[Tensor] = []
    for layer in module:
        x = layer(x if allow_grads_between_layers else x.detach())
        activations.append(x)
    return activations


@invert.register(nn.Sequential)
def invert_sequential(module: nn.Sequential) -> nn.Sequential:
    """ Returns a Module that can be used to compute or approximate the inverse
    operation of `self`.

    NOTE: In the case of Sequential, the order of the layers in the returned network
    is reversed compared to the input.
    """
    assert module.input_shape and module.output_shape, "Use the net before inverting."
    return type(module)(
        OrderedDict(
            (name, invert(module))
            for name, module in list(module._modules.items())[::-1]
        ),
    )


class Reshape(nn.Module, Invertible):
    def __init__(self, target_shape: Tuple[int, ...]):
        super().__init__()
        self.target_shape = tuple(target_shape)
        # add_hooks(self)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs.reshape([inputs.shape[0], *self.target_shape])
        if self.target_shape == (-1,):
            self.target_shape = outputs.shape[1:]
        return outputs

    def extra_repr(self) -> str:
        return f"({self.input_shape} -> {self.target_shape})"

    # def __repr__(self):
    #     return f"{type(self).__name__}({self.input_shape} -> {self.target_shape})"


@invert.register
def invert_reshape(module: Reshape) -> Reshape:
    assert module.input_shape and module.output_shape, "Use the net before inverting."
    return type(module)(target_shape=module.input_shape,)


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, Invertible):
    def __init__(self, output_size: Tuple[int, int] = None):
        super().__init__(output_size=output_size)
        # add_hooks(self)


@invert.register
def invert_avgpool2d(module: AdaptiveAvgPool2d) -> AdaptiveAvgPool2d:
    """ Returns a nn.AdaptiveAvgPool2d, which will actually upsample the input! """
    assert module.input_shape and module.output_shape, "Use the net before inverting."
    return type(module)(
        output_size=module.input_shape[-2:],  # type: ignore
    )


class MaxUnpool2d(nn.MaxUnpool2d, Invertible):
    # TODO: use a magic_bridge deque that is shared from the forward to the backward
    # net.

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t | None = None,
        padding: _size_2_t = 0,
        magic_bridge: deque[Tensor] = None,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.magic_bridge: deque[Tensor] | None = magic_bridge

    def forward(
        self, input: Tensor, indices: Tensor = None, output_size: list[int] = None
    ) -> Tensor:
        if indices is None:
            assert self.magic_bridge, "Need to pass indices or use a magic bridge!"
            # Only inspect rather than pop the item out, because of how the feedback
            # loss uses this backward layer twice.
            indices = self.magic_bridge[0]
        return super().forward(input=input, indices=indices, output_size=output_size)


@invert.register
def invert_maxunpool2d(module: AdaptiveMaxPool2d, init_symetric_weights: bool = False) -> "AdaptiveMaxPool2d":
    raise NotImplementedError("Never really need to invert a max Unpool layer.")
    assert module.input_shape and module.output_shape
    assert len(module.input_shape) > 2
    return AdaptiveMaxPool2d(
        output_size=module.input_shape[-2:],
    )


class MaxPool2d(nn.MaxPool2d, Invertible):
    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: _size_any_t = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=True,
            ceil_mode=ceil_mode,
        )
        self._return_indices = return_indices
        self.magic_bridge: deque[Tensor] = deque(maxlen=1)

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        out, indices = super().forward(input)
        # Push the indices onto the 'magic bridge':
        self.magic_bridge.append(indices)
        if self._return_indices:
            return out, indices
        return out


@invert.register
def invert_maxpool2d(module: MaxPool2d, init_symetric_weights: bool = False) -> MaxUnpool2d:
    return MaxUnpool2d(
        kernel_size=module.kernel_size,
        stride=None,  # todo: Not sure waht to do with this value here.
        padding=0,  # todo
        magic_bridge=module.magic_bridge,
    )


class BatchUnNormalize(nn.Module):
    """ TODO: Meant to be something like the 'inverse' of a batchnorm2d

    NOTE: No need to make this 'Invertible', because we don't really care about
    inverting this, we moreso would like to obtain this from 'inverting' batchnorm2d.
    """

    def __init__(self, num_features: int, dtype=torch.float32):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_features, dtype=dtype), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.scale)
        self.offset = nn.Parameter(torch.zeros(num_features, dtype=dtype), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        return input * self.scale + self.offset


@invert.register(nn.BatchNorm2d)
def invert_batchnorm(layer: nn.BatchNorm2d, init_symetric_weights: bool = False) -> BatchUnNormalize:
    # TODO: Is there a way to initialize symetric weights for BatchNorm?
    return BatchUnNormalize(num_features=layer.num_features, dtype=layer.weight.dtype)


class ConvPoolBlock(nn.Sequential, Invertible):
    """Convolutional block with max-pooling and an activation.

    NOTE: This isn't used anymore, Instead, I just use the `Sequential` class, where
    each passed layer is directly invertible.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: Type[nn.Module] = nn.ELU,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_type = activation_type
        self.conv: nn.Conv2d
        self.rho: nn.Module
        self.pool: nn.Module
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        rho = activation_type()
        pool = MaxPool2d(2)
        super().__init__(
            OrderedDict([("conv", conv), ("rho", rho), ("pool", pool)]),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Feedforward operator (x --> y = F(x))
        """
        return super().forward(x)
