from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Sequence

import pytorch_lightning
import torch
import tqdm
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from simple_parsing import ArgumentParser
from simple_parsing.helpers import list_field, mutable_field
from simple_parsing.helpers.hparams import HyperParameters
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.nn import functional as F
from pathlib import Path


@dataclass
class Config:
    # in_size: int = 784  # input dimension
    # out_size: int = 512  # output dimension
    # in_channels: int = 1  # input channels
    # out_channels: int = 128  # output channels

    data_dir: Path = Path("data")
    num_workers: int = 4

    epochs: int = 15  # number of epochs to train feedback weights
    iter: int = 20  # number of iterationson feedback weights per batch samples
    batch_size: int = 128  # batch dimension
    # device to use
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # noise: float = 0.05  # noise level


from abc import ABC, abstractmethod
from typing import NamedTuple


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
            yield from layer.forward_parameters()

    def backward_parameters(self) -> Iterable[nn.Parameter]:
        for layer in self:
            yield from layer.backward_parameters()

    def weight_b_sym(self):
        for layer in self:
            layer.weight_b_sym()

    def ff(self, x: Tensor) -> Tensor:
        y = x
        for layer in self:
            y = layer.ff(y)
        return y

    def bb(self, x: Tensor, y: Tensor) -> Tensor:
        r = x
        # TODO: Not 100% sure on this one:
        for layer in reversed(self):
            new_x = layer.bb(x=r, y=y)
        return y

class TargetPropSequentialV2(TargetPropModule):
    def __init__(
        self,
        forward_layers: List[TargetPropModule],
        backward_layers: List[TargetPropModule],
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

from functools import singledispatch


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


@dataclass
class OptimizerHParams:
    lr: float = 1e-3
    weight_decay: Optional[float] = None


class Prototype(LightningModule):
    @dataclass
    class HParams(HyperParameters):
        learning_rate_f: float = 0.05
        learning_rate_b: float = 0.5
        
        forward_optim: OptimizerHParams = mutable_field(OptimizerHParams, lr=0.05)
        backward_optim: OptimizerHParams = mutable_field(OptimizerHParams, lr=0.5)
        
        lr_f: float = 0.05  # learning rate
        lr_b: float = 0.5  # learning rate of the feedback weights
        lamb: float = 0.01  # regularization parameter
        beta: float = 0.1  # nudging parameter
        seed: Optional[int] = None  # Random seed to use.
        sym: bool = False  # sets symmetric weight initialization
        jacobian: bool = False  # compute jacobians
        conv: bool = False  # select the conv architecture.
        C: List[int] = list_field(128, 512)  # tab of channels
        noise: List[int] = list_field(0.05, 0.5)  # tab of noise amplitude

    def __init__(self, datamodule: VisionDataModule, hparams: HParams):
        super().__init__()
        self.hp: Prototype.HParams = hparams
        self.datamodule = datamodule
        # self.model = Net(args=self.hparams)
        channels, img_h, img_w = datamodule.dims
        n_classes = datamodule.num_classes
        self.model = TargetPropSequentialV1(
            TargetPropConv2d(channels, 16),
            TargetPropConv2d(16, 32),
            TargetPropConv2d(32, 64),
            TargetPropLinear(32768, n_classes),
        )
        self.loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        forward_optimizer = torch.optim.Adam(
            self.model.forward_parameters(),
            lr=self.hp.learning_rate_f,
            # weight_decay=self.hp.lamb,
        )
        backward_optimizer = torch.optim.Adam(
            self.model.backward_parameters(),
            lr=self.hp.learning_rate_b,
            # weight_decay=self.hp.lamb,
        )
        return [forward_optimizer, backward_optimizer]

    def shared_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch

        # Just debugging, will train "as usual", just backpropagating stuff.
        # y_pred, r =

        optimizers = self.optimizers()
        assert isinstance(optimizers, list)
        forward_optimizer = optimizers[0]
        backward_optimizer = optimizers[1]

        y_pred = self.model.ff(x)
        assert False, y_pred.shape
        loss = self.loss(y_pred, y)
        
        return loss
        # # backward acts like normal backward
        # self.manual_backward(loss, opt_a, retain_graph=True)
        # self.manual_backward(loss, opt_a)
        # opt_a.step()
        # opt_a.zero_grad()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch=batch, batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch=batch, batch_idx=batch_idx)


def main():
    from simple_parsing import ArgumentParser

    parser = ArgumentParser(description="Pytorch lightning version of the prototype")
    parser.add_arguments(Config, dest="config")
    parser.add_arguments(Prototype.HParams, dest="hparams")
    
    args = parser.parse_args()
    
    config: Config = args.config
    hparams: Prototype.HParams = args.hparams
    from pl_bolts.datamodules import CIFAR10DataModule
    from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
    import torchvision.transforms as T
    train_transforms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        cifar10_normalization(),
    ])

    test_transforms = T.Compose([
        T.ToTensor(),
        cifar10_normalization(),
    ])
    datamodule = CIFAR10DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    model = Prototype(datamodule=datamodule, hparams=hparams)
    trainer = Trainer(max_epochs=config.epochs)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()