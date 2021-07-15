from functools import singledispatch
import torch
from inspect import isclass
from torch import nn, Tensor
from torch.nn import functional as F
from typing import (
    Optional,
    NamedTuple,
    Iterable,
    Type,
    Callable,
    List,
    Sequence,
    Tuple,
    Union,
    ClassVar,
)
from abc import ABC, abstractmethod
from torchvision.models.resnet import BasicBlock
from torch.nn.modules.conv import _size_2_t
from functools import wraps
import torch
from torch import Tensor, nn
from typing import Union, List, Iterable, Sequence, Tuple


class Sequential(nn.Sequential, Sequence[nn.Module]):
    def forward_each(self, xs: List[Tensor]) -> List[Tensor]:
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
        assert len(xs) == len(self), (len(xs), len(self))
        return [layer(x_i) for layer, x_i in zip(self, xs)]

    def forward_all(
        self, x: Tensor, allow_grads_between_layers: bool = False,
    ) -> List[Tensor]:
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
        return forward_accumulate(
            self, x, allow_grads_between_layers=allow_grads_between_layers,
        )


# @torch.jit.script
def forward_accumulate(
    model: nn.Sequential, x: Tensor, allow_grads_between_layers: bool = False,
) -> List[Tensor]:
    """ IDEA: Gather all the forward activations into a list. """
    activations: List[Tensor] = []
    for layer in model:
        x = layer(x if allow_grads_between_layers else x.detach())
        activations.append(x)
    return activations


class Conv2dActivation(nn.Conv2d):
    activation: Callable[[Tensor], Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        activation: Callable[[Tensor], Tensor] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        if activation is None:
            if not hasattr(type(self), "activation"):
                raise RuntimeError(
                    "Need to either pass an activation as an argument to the "
                    "constructor, or have a callable `activation` class attribute."
                )
            activation = type(self).activation
        self._activation = activation
        assert callable(self._activation)

    def forward(self, input: Tensor) -> Tensor:
        return self._activation(super().forward(input))


class Conv2dReLU(Conv2dActivation):
    activation = F.relu


class Conv2dELU(Conv2dActivation):
    activation = F.elu


class ConvTranspose2dActivation(nn.ConvTranspose2d):
    activation: ClassVar[Callable[[Tensor], Tensor]]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
        activation: Callable[[Tensor], Tensor] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        if activation is None:
            if not hasattr(type(self), "activation"):
                raise RuntimeError(
                    "Need to either pass an activation as an argument to the "
                    "constructor, or have a callable `activation` class attribute."
                )
            activation = type(self).activation
        self._activation = activation
        assert callable(self._activation)

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        out: Tensor = super().forward(input, output_size=output_size)
        return self._activation(out)


class ConvTranspose2dReLU(ConvTranspose2dActivation):
    activation = F.relu


class ConvTranspose2dELU(ConvTranspose2dActivation):
    activation = F.elu


class Reshape(nn.Module):
    def __init__(
        self, target_shape: Tuple[int, ...], source_shape: Tuple[int, ...] = None
    ):
        self.target_shape = tuple(target_shape)
        self.source_shape = tuple(source_shape) if source_shape else ()
        super().__init__()

    def forward(self, inputs):
        if self.source_shape:
            if inputs.shape[1:] != self.source_shape:
                raise RuntimeError(
                    f"Inputs have unexpected shape {inputs.shape[1:]}, expected "
                    f"{self.source_shape}."
                )
        else:
            self.source_shape = inputs.shape[1:]
        outputs = inputs.reshape([inputs.shape[0], *self.target_shape])
        if self.target_shape == (-1,):
            self.target_shape = outputs.shape[1:]
        return outputs

    def __repr__(self):
        return f"{type(self).__name__}({self.source_shape} -> {self.target_shape})"


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(
        self, output_size: Tuple[int, ...], input_shape: Tuple[int, ...] = None
    ):
        super().__init__(output_size=output_size)
        self.input_shape: Tuple[int, ...] = input_shape or ()
        self.output_shape: Tuple[int, ...] = ()

    def forward(self, input):
        if self.input_shape == ():
            assert len(input.shape[1:]) == 3
            input_shape = input.shape[1:]
            self.input_shape = (input_shape[0], input_shape[1], input_shape[2])
        elif input.shape[1:] != self.input_shape:
            raise RuntimeError(
                f"Inputs have unexpected shape {input.shape[1:]}, expected "
                f"{self.input_shape}."
            )
        out = super().forward(input)
        if not self.output_shape:
            self.output_shape = out.shape[1:]
        elif out.shape[1:] != self.output_shape:
            raise RuntimeError(
                f"Outputs have unexpected shape {out.shape[1:]}, expected "
                f"{self.output_shape}."
            )
        return out


class BatchUnNormalize(nn.Module):
    """ TODO: Implement the 'opposite' of batchnorm2d """

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


from typing import overload
from collections import OrderedDict


class ConvPoolBlock(nn.Module):
    """Convolutional block with max-pooling and an activation.
    """

    def __init__(
        self, in_channels: int, out_channels: int, activation_type: Type[nn.Module],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_type = activation_type
        # NOTE: Could also subclass Sequential and do this?
        # super().__init__(OrderedDict(
        #     conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        #     pool = nn.MaxPool2d(2, stride=2, return_indices=True),
        #     rho = activation_type(),
        # ))
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # self.pool = nn.MaxPool2d(2, stride=2, return_indices=False)
        self.pool = nn.AvgPool2d(2)
        self.rho = activation_type()
        self.input_shape: Optional[Tuple[int, int, int]] = None
        self.output_shape: Optional[Tuple[int, int, int]] = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Feedforward operator (x --> y = F(x))
        """
        input_shape = tuple(x.shape[1:])
        if self.input_shape is None:
            assert len(input_shape) == 3
            self.input_shape = input_shape  # type: ignore
        elif input_shape != self.input_shape:
            # NOTE: This doesn't usually make sense for conv layers, which can be
            # applied to differently-shaped images, however here it might be useful
            # because we want the backward nets to produce a fixed output size.
            raise RuntimeError(
                f"Inputs have unexpected shape {input_shape}, expected "
                f"{self.input_shape}."
            )
        y = self.conv(x)
        y = self.rho(y)
        y = self.pool(y)
        output_shape = tuple(y.shape[1:])
        if self.output_shape is None:
            assert len(output_shape) == 3
            self.output_shape = output_shape  # type: ignore
        elif output_shape != self.output_shape:
            raise RuntimeError(
                f"Outputs have unexpected shape {output_shape}, expected "
                f"{self.output_shape}."
            )
        return y

    def extra_repr(self) -> str:
        return super().extra_repr() + (
            f"(input_shape={self.input_shape}, output_shape={self.output_shape})"
        )


class ConvTransposePoolBlock(nn.Module):
    """Convolutional transpose block with max-pooling and an activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: Type[nn.Module],
        input_shape: Tuple[int, int, int] = None,
        output_shape: Tuple[int, int, int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_type = activation_type
        self.output_shape = output_shape
        self.input_shape = input_shape
        
        # self.unpool = nn.MaxUnpool2d(2, )
        # NOTE: Need to pass the `scale_factor` kwargs to Upsample, passing a positional
        # arg of `2` doesn't work.
        self.unpool = nn.Upsample(scale_factor=2, mode="nearest")
        self.rho = activation_type()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, y: Tensor) -> Tensor:
        """
        Feedback operator (y --> r = G(y))
        """
        input_shape = tuple(y.shape[1:])
        if self.input_shape is None:
            assert len(input_shape) == 3
            self.input_shape = input_shape  # type: ignore
        elif input_shape != self.input_shape:
            # NOTE: This doesn't usually make sense for conv layers, which can be
            # applied to differently-shaped images, however here it might be useful
            # because we want the backward nets to produce a fixed output size.
            raise RuntimeError(
                f"Inputs have unexpected shape {input_shape}, expected "
                f"{self.input_shape}."
            )
        assert self.output_shape, "need to know output size in advance for now."
        output_size = self.output_shape[-2:]
        # Don't pass the output size when using nn.Upsample:
        # r = self.unpool(y, output_size=output_size)
        r = self.unpool(y)
        r = self.rho(r)
        r = self.conv(r, output_size=output_size)

        output_shape = tuple(r.shape[1:])
        if self.output_shape is None:
            assert len(output_shape) == 3
            self.output_shape = output_shape  # type: ignore
        elif output_shape != self.output_shape:
            raise RuntimeError(
                f"Outputs have unexpected shape {output_shape}, expected "
                f"{self.output_shape}."
            )

        return r

    def extra_repr(self) -> str:
        return super().extra_repr() + (
            f"(input_shape={self.input_shape}, output_shape={self.output_shape})"
        )