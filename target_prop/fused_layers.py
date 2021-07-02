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


class LayerOutput(NamedTuple):
    y: Tensor
    r: Optional[Tensor] = None


class TargetPropModule(nn.Module, ABC):
    @abstractmethod
    def ff(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def bb(self, x: Tensor, y: Tensor) -> Tensor:
        pass

    def forward(self, x: Tensor, back: bool = False) -> LayerOutput:
        """ Forward (and optionally backward) pass. """
        y = self.ff(x)
        r = self.bb(x, y) if back else None
        return LayerOutput(y=y, r=r)

    @abstractmethod
    def forward_parameters(self) -> Iterable[nn.Parameter]:
        pass

    @abstractmethod
    def backward_parameters(self) -> Iterable[nn.Parameter]:
        pass


class TargetPropLinear(TargetPropModule):
    def __init__(self, in_size: int, out_size: int, last_layer=False):
        super().__init__()
        self.f = nn.Linear(in_size, out_size)
        self.b = nn.Linear(out_size, in_size)
        self.last_layer = last_layer

    def ff(self, x: Tensor):
        return self.f(x)

    def bb(self, x: Tensor, y: Tensor) -> Tensor:
        return self.b(x)

    def weight_b_normalize(self, dx, dy, dr):

        factor = ((dy ** 2).sum(1)) / ((dx * dr).view(dx.size(0), -1).sum(1))
        factor = factor.mean()

        with torch.no_grad():
            self.b.weight.data = factor * self.b.weight.data

    def forward_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.f.parameters()

    def backward_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.b.parameters()

    def weight_b_sym(self):
        with torch.no_grad():
            self.b.weight.data = self.f.weight.data


class TargetPropConv2d(TargetPropModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        activation: Callable[[Tensor], Tensor] = F.relu,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.f = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.b = nn.ConvTranspose2d(
            out_channels, in_channels, kernel_size=kernel_size, stride=stride
        )

    def ff(self, x: Tensor) -> Tensor:
        return self.activation(self.f(x))

    def bb(self, x: Tensor, y: Tensor) -> Tensor:
        return self.activation(self.b(y, output_size=x.size()))

    def weight_b_normalize(self, dx, dy, dr):
        # first technique: same normalization for all out fmaps

        dy = dy.view(dy.size(0), -1)
        dx = dx.view(dx.size(0), -1)
        dr = dr.view(dr.size(0), -1)

        # print(dy.size())
        # print(dx.size())
        # print(dr.size())

        factor = ((dy ** 2).sum(1)) / ((dx * dr).sum(1))
        factor = factor.mean()
        # factor = 0.5*factor

        with torch.no_grad():
            self.b.weight.data = factor * self.b.weight.data

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

    def forward_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.f.parameters()

    def backward_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.b.parameters()

    def weight_b_sym(self):
        with torch.no_grad():
            # NOTE: I guess the transposition isn't needed here?
            self.b.weight.data = self.f.weight.data


class TargetPropSequentialV1(nn.Sequential, Sequence[TargetPropModule]):
    """IDEA:
    Should we do things this way:
    
    V1: List of layers, each layer handles the forward and "backward" pass (current)
    
        <==layer1==|==layer2==|==layer3==|>
        
    V2: Lists of layers, one for forward pass, one for backward pass?
    
        |--|---|- forward layers ---|---|> 
        <--|---|-- backward layers -|---|
        
    Pros:
    - Chaining layers like this might make things more efficient perhaps?   
    Cons:
    - training might be more complex?
    """

    def forward_parameters(self) -> Iterable[nn.Parameter]:
        for layer in self:
            if isinstance(layer, TargetPropModule):
                yield from layer.forward_parameters()
            else:
                # Maybe raise a warning?
                yield from layer.parameters()

    def backward_parameters(self) -> Iterable[nn.Parameter]:
        for layer in self:
            if isinstance(layer, TargetPropModule):
                yield from layer.backward_parameters()
            else:
                # Maybe raise a warning?
                yield from layer.parameters()

    def weight_b_sym(self):
        for layer in self:
            layer.weight_b_sym()

    def ff(self, x: Tensor) -> Tensor:
        y = x
        for layer in self:
            if isinstance(layer, TargetPropModule):
                y = layer.ff(y)
            else:
                # Maybe raise a warning?
                y = layer(y)
        return y

    def bb(self, x: Tensor, y: Tensor) -> Tensor:
        r = x
        # TODO: Not 100% sure on this one: should we start at the start for the forward
        # and at the end for the backward?
        raise NotImplementedError()

        for layer in reversed(self):
            new_x = layer.bb(x=r, y=y)
        return y

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x)

    def forward_all(self, x: Tensor) -> List[Tensor]:
        """ IDEA: Gather all the forward outputs of all layers into a list. """
        outputs: List[Tensor] = []
        for layer in self:
            x = layer.ff(x)
            outputs.append(x)
        return outputs

