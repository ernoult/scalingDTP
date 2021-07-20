from __future__ import annotations
from functools import singledispatch
import torch
from inspect import isclass
from torch import nn, Tensor
from torch.nn import functional as F
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
        module._check_input_shape(inputs[0])

    @staticmethod
    def forward_hook(module: Invertible, _: Any, output: Tensor) -> None:
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


class ConvPoolBlock(Sequential):
    """Convolutional block with max-pooling and an activation.
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
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            enforce_shapes=enforce_shapes,
        )
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.pool: nn.Module
        # self.pool = nn.MaxPool2d(2, stride=2, return_indices=False)
        if self.output_shape is not None:
            # If we already know the output shape we want, then we can use this!
            self.pool = AdaptiveAvgPool2d(
                output_shape=self.output_shape,
                # input_shape=self.input_shape,  # NOTE: Can't pass this, since its the input of the conv we'll get.
                enforce_shapes=self.enforce_shapes,
            )
        else:
            self.pool = nn.AvgPool2d(2)
        self.rho = activation_type()

    def forward(self, x: Tensor) -> Tensor:
        """
        Feedforward operator (x --> y = F(x))
        """
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


class ConvTransposePoolBlock(Invertible):
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
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            enforce_shapes=enforce_shapes,
        )
        # TODO: If using MaxUnpool, we'd have to add some kind of 'magic_get_indices'
        # function that gets passed as an constructor argument to this 'backward' block
        # by the forward block, and which would just retrieve the current max_indices
        # from the MaxPool2d layer.
        # self.unpool = nn.MaxUnpool2d(2, )
        self.unpool: nn.Module
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
        self.unpool = nn.Upsample(scale_factor=2, mode="nearest")
        self.rho = activation_type()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
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
