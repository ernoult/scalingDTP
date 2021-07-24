from __future__ import annotations
from functools import singledispatch
import torch
from inspect import isclass
from torch import nn, Tensor
from torch.nn import functional as F
from collections import deque
from typing import (
    Any,
    NamedTuple,
    Iterable,
    Type,
    Callable,
    Sequence,
    ClassVar,
)
from collections import OrderedDict
from abc import ABC, abstractmethod
from torchvision.models.resnet import BasicBlock
from torch.nn.modules.conv import _size_2_t
from functools import wraps
import torch
from torch import Tensor, nn
from typing import Union, List, Iterable, Sequence, Tuple
from .backward_layers import get_backward_equivalent
from typing import Generic
from typing import TypeVar

ModuleType = TypeVar("ModuleType", bound=nn.Module)


class Invertible(nn.Module, Generic[ModuleType], ABC):
    """ ABC for a Module that is invertible, i.e. whose `invert` method will return
    another Module which can produce the "backward equivalent" of its forward pass.

    Modules that inherit from this are also made aware of their input and output shapes,
    provided they are used at least once with some example input.
    This is done in order to make it easier to produce the backward pass network.
    
    This type is Generic, and the type argument indicates the type of network returned
    by the `invert` method.
    """

    def __init__(
        self,
        *args,
        input_shape: Tuple[int, ...] = None,
        output_shape: Tuple[int, ...] = None,
        enforce_shapes: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)  # type: ignore
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.enforce_shapes = enforce_shapes
        if self.input_shape is None or self.enforce_shapes:
            # Register a forward hook to set/check the output shape.
            self.register_forward_pre_hook(type(self).forward_pre_hook)
        if self.output_shape is None or self.enforce_shapes:
            # Register a forward hook to set/check the output shape.
            self.register_forward_hook(type(self).forward_hook)

    @abstractmethod
    def invert(self) -> ModuleType:
        """ Returns a Module that can be used to compute or approximate the inverse
        operation of `self`.
        """
        raise NotImplementedError

    def __invert__(self) -> ModuleType:
        """ Overrides the bitwise invert operator `~`, so that you can do something
        cute like:
        
        ```
        backward_net = forward_net.invert()
        # also equivalent to this:
        backward_net = ~forward_net
        ```
        """
        raise NotImplementedError

    @staticmethod
    def forward_pre_hook(module: Invertible, inputs: tuple[Tensor, ...]) -> None:
        assert len(inputs) == 1
        if isinstance(inputs[0], tuple):
            # Inspect only the first tensor if needed.
            inputs = inputs[0]
        module._check_input_shape(inputs[0])

    @staticmethod
    def forward_hook(module: Invertible, _: Any, output: Tensor | tuple[Tensor, ...]) -> None:
        if isinstance(output, tuple):
            output = output[0]
        module._check_output_shape(output)

    def _check_input_shape(self, x: Tensor) -> None:
        input_shape = tuple(x.shape[1:])
        if self.input_shape is None:
            self.input_shape = input_shape
        elif self.enforce_shapes and input_shape != self.input_shape:
            raise RuntimeError(
                f"Layer {self} expected inputs to have shape {self.input_shape}, but "
                f"got {input_shape} "
            )

    def _check_output_shape(self, output: Tensor) -> None:
        output_shape = tuple(output.shape[1:])
        if self.output_shape is None:
            self.output_shape = output_shape  # type: ignore
        elif self.enforce_shapes and output_shape != self.output_shape:
            raise RuntimeError(
                f"Outputs of layer {self} have unexpected shape {output_shape} "
                f"(expected {self.output_shape})!"
            )


@get_backward_equivalent.register(Invertible)
def _(network: Invertible):
    # Register this, so that calling `get_backward_equivalent` will use the `invert`
    # method if its input is an `Invertible` subclass.

    # Otherwise, `get_backward_equivalent` will fallback to its handlers for built-in
    # modules like `nn.Conv2d`, `nn.ConvTranspose2d`, etc.
    return network.invert()


class Sequential(Invertible["Sequential"], nn.Sequential):
    def forward_each(self, xs: list[Tensor]) -> list[Tensor]:
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
        assert len(xs) == len(self)
        return [layer(x_i) for layer, x_i in zip(self, xs)]

    def forward_all(
        self, x: Tensor, allow_grads_between_layers: bool = False,
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
        for layer in self:
            x = layer(x if allow_grads_between_layers else x.detach())
            activations.append(x)
        return activations

    def invert(self) -> Sequential:
        """ Returns a Module that can be used to compute or approximate the inverse
        operation of `self`.

        NOTE: In the case of Sequential, the order of the layers in the returned network
        is reversed compared to the input.
        """
        assert self.input_shape and self.output_shape, "Use the net before inverting."
        return type(self)(
            OrderedDict(
                (name, get_backward_equivalent(module))
                for name, module in list(self._modules.items())[::-1]
            ),
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            enforce_shapes=self.enforce_shapes,
        )

    # NOTE: Not sure if we should instead use the 'reversed' function, because it might
    # be confused with the reverse-iterator.
    def __reversed__(self) -> Sequential:
        return self[::-1]


class Reshape(Invertible["Reshape"]):
    def __init__(
        self,
        target_shape: Tuple[int, ...] = None,
        output_shape: Tuple[int, ...] = None,
        input_shape: Tuple[int, ...] = None,
        enforce_shapes: bool = True,
    ):
        if target_shape is None:
            assert output_shape, "need one of target_shape or output_shape."
            target_shape = output_shape
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            enforce_shapes=enforce_shapes,
        )
        self.target_shape = tuple(target_shape)

    def forward(self, inputs):
        outputs = inputs.reshape([inputs.shape[0], *self.target_shape])
        if self.target_shape == (-1,):
            self.target_shape = outputs.shape[1:]
        return outputs

    def __repr__(self):
        return f"{type(self).__name__}({self.input_shape} -> {self.target_shape})"

    def invert(self) -> Reshape:
        assert self.input_shape and self.output_shape, "Use the net before inverting."
        return type(self)(
            # target_shape=self.target_shape,
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            enforce_shapes=self.enforce_shapes,
        )


class AdaptiveAvgPool2d(Invertible, nn.AdaptiveAvgPool2d):
    def __init__(
        self,
        output_size: Tuple[int, int] = None,
        input_shape: Tuple[int, ...] = None,
        output_shape: Tuple[int, ...] = None,
        enforce_shapes: bool = True,
    ):
        assert output_size or output_shape, "Need one of those"
        if not output_size and output_shape:
            output_size = output_shape[-2:]
        super().__init__(
            output_size=output_size,
            input_shape=input_shape,
            output_shape=output_shape,
            enforce_shapes=enforce_shapes,
        )

    def invert(self) -> AdaptiveAvgPool2d:
        """ Returns a nn.AdaptiveAvgPool2d, which will actually upsample the input! """
        assert self.input_shape and self.output_shape, "Use the net before inverting."
        return type(self)(
            output_size=self.input_shape[-2:],
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            enforce_shapes=self.enforce_shapes,
        )


class MaxUnpool2d(Invertible["AdaptiveMaxPool2d"], nn.MaxUnpool2d):
    # TODO: use a magic_bridge deque that is shared from the forward to the backward
    # net.

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t | None = None,
        padding: _size_2_t = 0,
        input_shape: tuple[int, ...] = None,
        output_shape: tuple[int, ...] = None,
        enforce_shapes: bool = True,
        magic_bridge: deque[Tensor] = None,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            input_shape=input_shape,
            output_shape=output_shape,
            enforce_shapes=enforce_shapes,
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

    def invert(self) -> "AdaptiveMaxPool2d":
        raise NotImplementedError("Never really need to invert a max Unpool layer.")
        assert self.input_shape and self.output_shape
        assert len(self.input_shape) > 2
        return AdaptiveMaxPool2d(
            output_size=self.input_shape[-2:],
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            enforce_shapes=self.enforce_shapes,
        )

from torch.nn.modules.pooling import _size_any_t
from typing import Optional


class MaxPool2d(Invertible[MaxUnpool2d], nn.MaxPool2d):
    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: Optional[_size_any_t] = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        input_shape: Tuple[int, ...] = None,
        output_shape: Tuple[int, ...] = None,
        enforce_shapes: bool = True,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=True,
            ceil_mode=ceil_mode,
            input_shape=input_shape,
            output_shape=output_shape,
            enforce_shapes=enforce_shapes,
        )
        self._return_indices = return_indices
        self.magic_bridge: deque[Tensor] = deque(maxlen=1)

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        out, indices = super().forward(input)
        self.magic_bridge.append(indices)
        if self._return_indices:
            return out, indices
        return out

    def invert(self) -> MaxUnpool2d:
        return MaxUnpool2d(
            kernel_size=self.kernel_size,
            stride=None,  # todo: Not sure waht to do with this value here.
            padding=0,  # todo
            magic_bridge=self.magic_bridge,
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            enforce_shapes=self.enforce_shapes,
        )


class AdaptiveMaxPool2d(Invertible[MaxUnpool2d], nn.AdaptiveMaxPool2d):
    # TODO: Adaptive max pool that shares the indices automagically with the backward
    # network through a shared 'magic bridge' deque.
    def __init__(
        self,
        output_size: Tuple[int, int] = None,
        return_indices: bool = False,
        magically_share_indices_with_inverse: bool = True,
        input_shape: Tuple[int, ...] = None,
        output_shape: Tuple[int, ...] = None,
        enforce_shapes: bool = True,
    ):
        assert output_size or output_shape, "Need one of those"
        if not output_size and output_shape:
            assert len(output_shape) > 2
            output_size = output_shape[-2:]  # type: ignore
        super().__init__(
            output_size=output_size,
            return_indices=True,
            input_shape=input_shape,
            output_shape=output_shape,
            enforce_shapes=enforce_shapes,
        )
        self._return_indices = return_indices
        self.magically_share_indices_with_inverse = magically_share_indices_with_inverse
        self.magic_bridge: deque[Tensor] = deque(maxlen=2)

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        out, indices = super().forward(input)
        if self.magically_share_indices_with_inverse:
            assert (
                len(self.magic_bridge) != 2
            ), "Why didn't the backward net pull the indices out?"
            self.magic_bridge.append(indices)
        if self._return_indices:
            return out, indices
        return out

    def invert(self) -> MaxUnpool2d:
        assert False, (self.kernel_size, self.stride, self.padding)
        return MaxUnpool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,  # todo
            padding=0,  # todo
            magic_bridge=self.magic_bridge,
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            enforce_shapes=self.enforce_shapes,
        )

    # def forward(self, input, indices: Tensor, output_size=None):
    #     return super().forward(input, indices, output_size=output_size)


class BatchUnNormalize(nn.Module):
    """ TODO: Meant to be something like the 'inverse' of a batchnorm2d

    NOTE: No need to make this 'Invertible', because we don't really care about
    inverting this, we moreso would like to obtain this from 'inverting' batchnorm2d.
    """

    def __init__(self, num_features: int, dtype=torch.float32):
        super().__init__()
        self.scale = nn.Parameter(
            torch.ones(num_features, dtype=dtype), requires_grad=True
        )
        torch.nn.init.xavier_uniform_(self.scale)
        self.offset = nn.Parameter(
            torch.zeros(num_features, dtype=dtype), requires_grad=True
        )

    def forward(self, input: Tensor) -> Tensor:
        return input * self.scale + self.offset


@get_backward_equivalent.register(nn.BatchNorm2d)
def _(layer: nn.BatchNorm2d) -> BatchUnNormalize:
    return BatchUnNormalize(num_features=layer.num_features, dtype=layer.weight.dtype)


class ConvPoolBlock(Sequential, Invertible["ConvTransposePoolBlock"]):
    """Convolutional block with max-pooling and an activation.

    TODO: Morph this into a Sequential subclass.
    """

    def __init__(
        self,
        in_channels: int = None,
        out_channels: int = None,
        activation_type: Type[nn.Module] = nn.ELU,
        input_shape: Tuple[int, ...] = None,
        output_shape: Tuple[int, ...] = None,
        enforce_shapes: bool = True,
    ):
        assert in_channels or input_shape
        assert out_channels or output_shape
        if in_channels is None:
            assert input_shape
            assert len(input_shape) == 3
            in_channels = input_shape[0]
        if out_channels is None:
            assert output_shape
            assert len(output_shape) == 3
            out_channels = output_shape[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_type = activation_type
        self.conv: nn.Conv2d
        self.rho: nn.Module
        self.pool: nn.Module

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        rho = activation_type()
        pool: nn.Module
        # pool = AdaptiveMaxPool2d()
        # self.pool = nn.MaxPool2d(2, stride=2, return_indices=False)
        if output_shape is not None:
            # If we already know the output shape we want, then we can use this!
            # TODO: Use this instead?
            # self.pool = AdaptiveMaxPool2d(
            #     output_shape=self.output_shape,
            #     enforce_shapes=self.enforce_shapes,
            #     return_indices=False,
            #     magically_share_indices_with_inverse=True,
            # )
            pool = AdaptiveAvgPool2d(
                output_shape=output_shape,
                # input_shape=self.input_shape,  # NOTE: Can't pass this, since its the input of the conv we'll get.
                enforce_shapes=enforce_shapes,
            )
        else:
            pool = nn.AvgPool2d(2)

        super().__init__(
            OrderedDict([("conv", conv), ("rho", rho), ("pool", pool),]),
            input_shape=input_shape,
            output_shape=output_shape,
            enforce_shapes=enforce_shapes,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Feedforward operator (x --> y = F(x))
        """
        return super().forward(x)
        y = self.conv(x)
        y = self.rho(y)
        y = self.pool(y)
        return y

    def invert(self) -> ConvTransposePoolBlock:
        assert self.input_shape and self.output_shape, "Use the net before inverting."
        return ConvTransposePoolBlock(
            in_channels=self.out_channels,
            out_channels=self.in_channels,
            activation_type=self.activation_type,
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            enforce_shapes=self.enforce_shapes,
        )


class ConvTransposePoolBlock(Sequential, Invertible[ConvPoolBlock]):
    """Convolutional transpose block with max-pooling and an activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: Type[nn.Module],
        input_shape: Tuple[int, ...] = None,
        output_shape: Tuple[int, ...] = None,
        enforce_shapes: bool = True,
    ):
        assert in_channels or input_shape
        assert out_channels or output_shape
        if in_channels is None:
            assert input_shape
            assert len(input_shape) == 3
            in_channels = input_shape[0]
        if out_channels is None:
            assert output_shape
            assert len(output_shape) == 3
            out_channels = output_shape[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_type = activation_type
        # super().__init__(
        #     input_shape=input_shape,
        #     output_shape=output_shape,
        #     enforce_shapes=enforce_shapes,
        # )
        # TODO: If using MaxUnpool, we'd have to add some kind of 'magic_get_indices'
        # function that gets passed as an constructor argument to this 'backward' block
        # by the forward block, and which would just retrieve the current max_indices
        # from the MaxPool2d layer.
        # self.unpool = nn.MaxUnpool2d(2, )
        self.unpool: nn.Module
        self.rho: nn.Module
        self.conv: nn.ConvTranspose2d
        # if self.output_shape is not None:
        # NOTE: Can't really
        # This is way cooler, but we can't use it, because we'd need to know the
        # input size of this convolution (which we don't!)
        # self.unpool = AdaptiveAvgPool2d(
        #     output_size=
        #     # output_shape=self.output_shape,  # Can't pass this
        #     input_shape=self.input_shape,
        #     enforce_shapes=self.enforce_shapes,
        # )
        # else:

        # NOTE: Need to pass the `scale_factor` kwargs to Upsample, passing a
        # positional value of `2` doesn't work.
        unpool = nn.Upsample(scale_factor=2, mode="nearest")
        rho = activation_type()
        conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        super().__init__(
            OrderedDict([("unpool", unpool), ("rho", rho), ("conv", conv),]),
            input_shape=input_shape,
            output_shape=output_shape,
            enforce_shapes=enforce_shapes,
        )

    def forward(self, y: Tensor, output_size: list[int] = None) -> Tensor:
        """
        Feedback operator (y --> r = G(y))
        """
        # NOTE: Could also let this 'float', but for now it's easier this way.
        if output_size is None:
            assert self.output_shape, "need either `output_size` or `self.output_shape`"
            assert len(self.output_shape) == 3
            output_size = self.output_shape[-2:]
        # Don't pass the output size when using nn.Upsample:
        # r = self.unpool(y, output_size=output_size)
        r = self.unpool(y)
        r = self.rho(r)
        r = self.conv(r, output_size=output_size)
        return r

    def invert(self) -> ConvPoolBlock:
        assert self.input_shape, "Use the net before inverting."
        assert self.output_shape, "Use the net before inverting."
        assert len(self.input_shape) == 3
        return ConvPoolBlock(
            in_channels=self.out_channels,
            out_channels=self.in_channels,
            activation_type=self.activation_type,
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            enforce_shapes=self.enforce_shapes,
        )
