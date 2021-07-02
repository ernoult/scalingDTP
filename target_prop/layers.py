from functools import singledispatch
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import (
    Optional,
    NamedTuple,
    Iterable,
    Callable,
    List,
    Sequence,
    Tuple,
    Union,
)
from abc import ABC, abstractmethod
from torchvision.models import resnet18
from torch.nn import Sequential
from .fused_layers import TargetPropModule


class TargetPropSequentialV2(TargetPropModule):
    def __init__(
        self,
        forward_layers: Union[nn.Sequential, Sequence[nn.Module]],
        backward_layers: Union[nn.Sequential, Sequence[nn.Module]],
    ):
        super().__init__()
        self.forward_net = nn.Sequential(*forward_layers)
        self.backward_net = nn.Sequential(*backward_layers)

    def ff(self, x: Tensor) -> Tensor:
        return self.forward_net(x)

    def bb(self, x: Tensor, y: Tensor) -> Tensor:
        return self.backward_net(x)

    def weight_b_sym(self):
        for i, (forward_layer, backward_layer) in enumerate(
            zip(self.forward_net, self.backward_net)
        ):
            weight_b_sym(forward_layer, backward_layer)
            # self.layers[i].weight_b_sym()

    def forward_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.forward_net.parameters()

    def backward_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.backward_net.parameters()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.forward_net(x)

    def forward_all(self, x: Tensor, allow_grads_between_layers: bool = False) -> List[Tensor]:
        if allow_grads_between_layers:
            return forward_accumulate(self.forward_net, x)
        else:
            return layerwise_independant_forward_accumulate(self.forward_net, x)

    def backward_all(self, y: Tensor, allow_grads_between_layers: bool = False) -> List[Tensor]:
        if allow_grads_between_layers:
            return forward_accumulate(self.backward_net, y)
        else:
            return layerwise_independant_forward_accumulate(self.backward_net, y)

    def layerwise_forward_backward(self, x: Union[Tensor, List[Tensor]]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Return the y's and the r's of each the layers, where `r` is the 'reconstruction'
        term.
        
        Can either accept `x` as a Tensor, and the (detached) output of a layer is used
        as the input of the next, or a list of tensors which are used as inputs to each
        layer.
        NOTE: This can be used for example to calculate the reconstruction of a noisy X
        for each layer. 

        r_i is linked through gradients to y_i, but y_i isn't linked to y_{i-1}
        """
        y: Tensor
        ys: List[Tensor] = []
        rs: List[Tensor] = []
        for i, (forward_layer, backward_layer) in enumerate(zip(self.forward_net, reversed(self.backward_net))):
            x_i = x[i] if isinstance(x, list) else x.detach()
            y = forward_layer(x_i)
            r = backward_layer(y)
            ys.append(y)
            rs.append(r)
        return ys, rs


# @torch.jit.script
def forward_accumulate(model: nn.Sequential, x: Tensor) -> List[Tensor]:
    """ IDEA: Gather all the forward activations into a list. """
    activations: List[Tensor] = []
    for layer in model:
        x = layer(x)
        activations.append(x)
    return activations

def layerwise_independant_forward_accumulate(model: nn.Sequential, x: Tensor) -> List[Tensor]:
    """ IDEA: Gather all the forward activations into a list, and have layer's output be
    disconnected from that of the previous layer.
    """
    activations: List[Tensor] = []
    for layer in model:
        x = layer(x.detach())
        activations.append(x)
    return activations





@singledispatch
def weight_b_sym(forward_layer: nn.Module, backward_layer: nn.Module) -> None:
    raise NotImplementedError(forward_layer, backward_layer)


@weight_b_sym.register(nn.Conv2d)
def weight_b_sym_conv2d(
    forward_layer: nn.Conv2d, backward_layer: nn.ConvTranspose2d
) -> None:
    assert forward_layer.weight.shape == backward_layer.weight.shape
    with torch.no_grad():
        # NOTE: I guess the transposition isn't needed here?
        backward_layer.weight.data = forward_layer.weight.data


@weight_b_sym.register(nn.Linear)
def weight_b_sym_linear(forward_layer: nn.Linear, backward_layer: nn.Linear) -> None:
    assert forward_layer.in_features == backward_layer.out_features
    assert forward_layer.out_features == backward_layer.in_features
    # TODO: Not sure how this would work if a bias term was used, so assuming we don't
    # have one for now.
    assert forward_layer.bias is None and backward_layer.bias is None
    # assert forward_layer.bias is not None == backward_layer.bias is not None

    with torch.no_grad():
        # NOTE: I guess the transposition isn't needed here?
        backward_layer.weight.data = forward_layer.weight.data.t()


class Reshape(nn.Module):
    def __init__(
        self, target_shape: Tuple[int, ...], source_shape: Tuple[int, ...] = None
    ):
        self.target_shape = target_shape
        self.source_shape = source_shape
        super().__init__()

    def forward(self, inputs):
        if self.source_shape:
            if inputs.shape[1:] != self.source_shape:
                raise RuntimeError(
                    f"Inputs have unexpected shape {inputs.shape}, expected "
                    f"{self.source_shape}."
                )
        return inputs.reshape([inputs.shape[0], *self.target_shape])

    def __repr__(self):
        return f"{type(self).__name__}({self.source_shape} -> {self.target_shape})"


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(
        self, output_size: Tuple[int, ...], input_shape: Tuple[int, ...] = None
    ):
        super().__init__(output_size=output_size)
        self.input_shape: Tuple[int, ...] = ()
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
        elif input.shape[1:] != self.output_shape:
            raise RuntimeError(
                f"Outputs have unexpected shape {out.shape[1:]}, expected "
                f"{self.output_shape}."
            )
        return out
