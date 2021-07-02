from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Sequence, Union

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

from target_prop.layers import AdaptiveAvgPool2d, Reshape, TargetPropSequentialV2
from target_prop.backward_layers import get_backward_equivalent


@dataclass
class OptimizerHParams:
    lr: float = 1e-3
    weight_decay: Optional[float] = None


@dataclass
class Config:
    # in_size: int = 784  # input dimension
    # out_size: int = 512  # output dimension
    # in_channels: int = 1  # input channels
    # out_channels: int = 128  # output channels

    data_dir: Path = Path("data")
    num_workers: int = 4

    epochs: int = 15  # number of epochs to train feedback weights
    batch_size: int = 128  # batch dimension
    # device to use
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # noise: float = 0.05  # noise level


from abc import ABC, abstractmethod
from typing import NamedTuple
from functools import singledispatch

@singledispatch
def weight_b_normalize(backward_layer: nn.Module, dx: Tensor, dy: Tensor, dr: Tensor) -> None:
    """ TODO: I don't yet understand what this is supposed to do. """
    return
    # raise NotImplementedError(f"No idea what this means atm.")


@weight_b_normalize.register
def linear_weight_b_normalize(backward_layer: nn.Linear, dx: Tensor, dy: Tensor, dr: Tensor) -> None:
    # dy = dy.view(dy.size(0), -1)
    # dx = dx.view(dx.size(0), -1)
    # dr = dr.view(dr.size(0), -1)
    
    factor = ((dy**2).sum(1))/((dx*dr).view(dx.size(0), -1).sum(1))
    factor = factor.mean()

    with torch.no_grad():
        backward_layer.weight.data = factor * backward_layer.weight.data


@weight_b_normalize.register
def conv_weight_b_normalize(backward_layer: nn.ConvTranspose2d, dx: Tensor, dy: Tensor, dr: Tensor) -> None:
    #first technique: same normalization for all out fmaps        
    
    dy = dy.view(dy.size(0), -1)
    dx = dx.view(dx.size(0), -1)
    dr = dr.view(dr.size(0), -1)
    
    factor = ((dy**2).sum(1))/((dx*dr).sum(1))
    factor = factor.mean()
    #factor = 0.5*factor

    with torch.no_grad():
        backward_layer.weight.data = factor * backward_layer.weight.data    

    #second technique: fmaps-wise normalization
    '''
    dy_square = ((dy.view(dy.size(0), dy.size(1), -1))**2).sum(-1) 
    dx = dx.view(dx.size(0), dx.size(1), -1)
    dr = dr.view(dr.size(0), dr.size(1), -1)
    dxdr = (dx*dr).sum(-1)
    
    factor = torch.bmm(dy_square.unsqueeze(-1), dxdr.unsqueeze(-1).transpose(1,2)).mean(0)
    
    factor = factor.view(factor.size(0), factor.size(1), 1, 1)
        
    with torch.no_grad():
        self.b.weight.data = factor*self.b.weight.data
    '''


class Prototype(LightningModule):
    @dataclass
    class HParams(HyperParameters):
        learning_rate_f: float = 0.05
        learning_rate_b: float = 0.5

        forward_optim: OptimizerHParams = mutable_field(OptimizerHParams, lr=0.05)
        backward_optim: OptimizerHParams = mutable_field(OptimizerHParams, lr=0.5)

        lr_f: float = 1e-3 # learning rate
        lr_b: float = 1e-3  # learning rate of the feedback weights
        lamb: float = 0.01  # regularization parameter
        beta: float = 0.1  # nudging parameter

        # number of iterations on feedback weights per batch samples
        iterations: int = 20

        seed: Optional[int] = None  # Random seed to use.
        sym: bool = False  # sets symmetric weight initialization
        jacobian: bool = False  # compute jacobians
        conv: bool = False  # select the conv architecture.
        C: List[int] = list_field(128, 512)  # tab of channels
        # noise: List[float] = list_field(0.05, 0.5)  # tab of noise amplitude
        noise: float = 0.05  # tab of noise amplitude

    def __init__(self, datamodule: VisionDataModule, hparams: HParams):
        super().__init__()
        self.hp: Prototype.HParams = hparams
        self.datamodule = datamodule
        # self.model = Net(args=self.hparams)
        self.channels, self.img_h, self.img_w = datamodule.dims
        self.n_classes = datamodule.num_classes
        # self.model = TargetPropSequentialV1(
        #     TargetPropConv2d(channels, 16),
        #     TargetPropConv2d(16, 32),
        #     TargetPropConv2d(32, 64),
        #     # TargetPropConv2d(64, 128),
        #     # AdaptivePooling(4),
        #     nn.Flatten(),
        #     TargetPropLinear(8192, n_classes),
        # )
        self.example_input_array = torch.rand(
            [32, self.channels, self.img_h, self.img_w], device=self.device
        )

        forward_net = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels,
                out_channels=16,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            # NOTE: the output window is [5, 5] with the inputs we currently have.
            nn.ReLU(),
            AdaptiveAvgPool2d((4, 4)),  # Simplifies the construction of the network
            Reshape(source_shape=(64, 4, 4), target_shape=(64 * 4 * 4,)),
            nn.Linear(in_features=64 * 4 * 4, out_features=self.n_classes),
        )
        example_out = forward_net(self.example_input_array)
        out_shape = example_out.shape
        # hidden_dims = forward_net[-1].in_features
        assert example_out.requires_grad
        backward_net = nn.Sequential(
            *[
                get_backward_equivalent(forward_layer)
                for forward_layer in reversed(forward_net)
            ]
        )
        example_in_hat = backward_net(example_out) 
        assert example_in_hat.requires_grad
        assert example_in_hat.shape == self.example_input_array.shape

        self.model = TargetPropSequentialV2(
            forward_layers=forward_net, backward_layers=backward_net,
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
        # TODO: How about we use this frequency instead of the for loop in training_step?
        return (
            {"optimizer": forward_optimizer, "frequency": 1},
            # {"optimizer": backward_optimizer, "frequency": 1},
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.model.forward_net(input)

    def shared_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, phase: str, optimizer_idx: int = None
    ):
        x, y = batch

        # ys, rs = self.model.layerwise_forward_backward(x)

        y_pred = self.model.forward_net(x)
        # Just debugging, will train "as usual" backpropagating stuff all the way
        # through.
        # ys: List[Tensor] = self.model.forward_all(x, allow_grads_between_layers=True)
        # y_pred: Tensor = ys[-1]

        # rs: List[Tensor] = self.model.backward_all(y_pred, allow_grads_between_layers=True)
        # rs = list(reversed(rs))
        loss: Tensor = torch.zeros((), device=self.device)

        last_layer_loss = self.loss(y_pred, y)
        loss += last_layer_loss

        accuracy = y_pred.argmax(-1).eq(y).sum().float().div(len(y_pred))
        self.log("last_layer_loss", last_layer_loss, on_step=True, prog_bar=True, on_epoch=False)
        self.log("Accuracy", accuracy, on_step=True, prog_bar=True, on_epoch=False)
    
        return loss
    
        if optimizer_idx in [None, 1]:
            # Inputs to each layer:
            xs: List[Tensor] = [x] + ys[:-1]
            
            noisy_xs = [x + self.hp.noise * torch.randn_like(x) for x in xs]
            noisy_xs_ys, noisy_xs_rs = self.model.layerwise_forward_backward(noisy_xs)

            delta_ys = [noisy_xs_y - y_i for noisy_xs_y, y_i in zip(noisy_xs_ys, ys)]
            delta_rs = [noisy_xs_r - r_i for noisy_xs_r, r_i in zip(noisy_xs_rs, rs)]

            delta_r_loss = sum(
                - self.hp.noise * delta_r.view(delta_r.shape[0], -1).sum(1).mean()
                for delta_r in delta_rs
            )
            # loss += delta_r_loss
            self.log("delta_r_loss", delta_r_loss, on_step=True, prog_bar=True, on_epoch=False)

            delta_y_loss = sum(
                - self.hp.noise * delta_y.view(delta_y.shape[0], -1).sum(1).mean()
                for delta_y in delta_ys
            )
            # loss += delta_y_loss
            self.log("delta_y_loss", delta_y_loss, on_step=True, prog_bar=True, on_epoch=False)
        assert loss.requires_grad
        return loss

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int = None
    ):
        self.model.train()
        loss = self.shared_step(batch=batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx, phase="train")
        assert loss.requires_grad, (loss, optimizer_idx)
        if optimizer_idx == 1:
            # Still debugging things a bit, so just print the loss but dont backprop it.
            loss = 1e-10 * loss
        return loss
        
        x, y = batch

        optimizers = self.optimizers()
        assert isinstance(optimizers, list) and len(optimizers) == 2
        forward_optimizer = optimizers[0]
        backward_optimizer = optimizers[1]

        if optimizer_idx == 0:
            """ Optimize the 'forward' pass? """
            optimizer = optimizers[optimizer_idx]
            assert isinstance(optimizer, Optimizer)
            optimizer.zero_grad()

            ys, rs = self.model.layerwise_forward_backward(x)
            xs = [x] + [y_i.detach() for y_i in ys[:-1]]
            noisy_xs = [x_i + self.hp.noise * torch.randn_line(x_i) for x_i in xs] 
            # TODO: Is delta_y something we want to set? or somethign we observe from
            # setting delta_x?
            # noisy_ys = [y_i + self.hp.noise * torch.randn_line(y_i) for y_i in xs]
            noisy_ys, noisy_rs = self.model.layerwise_forward_backward(noisy_xs)

            delta_rs = [
                x - noisy_r for x, noisy_r in zip(xs, noisy_rs)
            ]
            # TODO: Shouldn't there be an absolute value somewhere here? 
            loss_b: Tensor = - sum(  # type: ignore
                (self.hp.noise * dr).flatten(start_dim=1).sum(1).mean()
                for i, dr in enumerate(delta_rs) if i != 0
            )
            # y = layers[0]
            # y_temp, r_temp = second_layer(y, back = True)
            # noise = args.noise[id_layer]*torch.randn_like(y)
            # y_noise, r_noise = second_layer(y + noise, back = True)
            # dy = (y_noise - y_temp)
            # dr = (r_noise - r_temp)

            # TODO: Should the ys[-1] be set to the target in this case?
            # delta_ys = [self.hp.noise * torch.randn_like(y_i) for y_i in ys]
            # noisy_ys = [y_i + delta_y for y_i, delta_y in zip(ys, delta_ys)]
            
            
            # ys: List[Tensor] = self.model.forward_all(x)
            # y_pred: Tensor = ys[-1]
            # # rs: List[Tensor] = self.model.backward_all(y_pred)
            # rs: List[Tensor] = [
            #     layer(y) for layer, y in zip(
            #         reversed(self.model.backward_net), ys
            #     )
            # ]
            # rs = list(reversed(rs))
            # # xs: List[Tensor] = [x] + ys[:-1]

            # delta_ys = [self.hp.noise * torch.randn_like(y_i) for y_i in ys]
            # noisy_ys = [y_i + delta_y for y_i, delta_y in zip(ys, delta_ys)]
            # assert len(self.model.backward_net) == len(noisy_ys)
            # noisy_rs = [
            #     layer(noisy_y)
            #     for layer, noisy_y in zip(
            #         reversed(self.model.backward_net), noisy_ys
            #     )
            # ]
            # delta_rs = [r - noisy_r for r, noisy_r in zip(rs, noisy_rs)]
            # # TODO: The first index doesn't contribute to the loss, right?
            # # loss_b = -(noise * dr).view(dr.size(0), -1).sum(1).mean()
            # loss_b: Tensor = - sum(
            #     (self.hp.noise * dr).flatten(start_dim=1).sum(1).mean()
            #     for i, dr in enumerate(delta_rs) if i != 0
            # )
            # self.log

            # if iteration < self.hp.iterations:
            #     loss_b.backward(retain_graph=True)
            # else:
            #     loss_b.backward()

            optimizer.step()
            # renormalize once per sample
            net.layers[id_layer + 1].weight_b_normalize(noise, dy, dr)

            # go to the next layer
            y = net.layers[id_layer + 1](y).detach()

        ys: List[Tensor] = self.model.forward_all(x)
        y_pred: Tensor = ys[-1]

        loss = self.loss(y_pred, y)

        y_temp, r_temp = net.layers[id_layer + 1](y, back=True)
        noise = args.noise[id_layer] * torch.randn_like(y)
        y_noise, r_noise = net.layers[id_layer + 1](y + noise, back=True)
        dy = y_noise - y_temp
        dr = r_noise - r_temp

        loss_b = -(noise * dr).view(dr.size(0), -1).sum(1).mean()

        optimizer_b.zero_grad()

        if iter < args.iter:
            loss_b.backward(retain_graph=True)
        else:
            loss_b.backward()

        optimizer_b.step()

        return loss

        rs: List[Tensor] = self.model.backward_all(y_pred)
        xs: List[Tensor] = [x] + ys[:-1]

        y = self.model.layers[0](data).detach()
        for layer_index, layer in enumerate(self.model):
            for iteration in range(1, self.hp.iterations + 1):
                y_temp, r_temp = net.layers[id_layer + 1](y, back=True)
                noise = args.noise[id_layer] * torch.randn_like(y)
                y_noise, r_noise = net.layers[id_layer + 1](y + noise, back=True)
                dy = y_noise - y_temp
                dr = r_noise - r_temp

                loss_b = -(noise * dr).view(dr.size(0), -1).sum(1).mean()

                optimizer_b.zero_grad()

                if iter < args.iter:
                    loss_b.backward(retain_graph=True)
                else:
                    loss_b.backward()

                optimizer_b.step()

            # renormalize once per sample
            net.layers[id_layer + 1].weight_b_normalize(noise, dy, dr)

            # go to the next layer
            y = net.layers[id_layer + 1](y).detach()

        # ****FORWARD WEIGHTS****#

        y, r = net(data, ind=len(net.layers))

        # compute prediction
        pred = torch.exp(net.logsoft(y))
        target = F.one_hot(target, num_classes=10).float()

        # compute first target on the softmax logits
        t = y + args.beta * (target - pred)

        for i in range(len(net.layers)):

            # update forward weights
            loss_f = 0.5 * ((y - t) ** 2).view(y.size(0), -1).sum(1)
            loss_f = loss_f.mean()

            if i == 0:
                loss = loss_f

            optimizer_f.zero_grad()
            loss_f.backward(retain_graph=True)
            optimizer_f.step()

            # compute previous targets
            if i < len(net.layers) - 1:
                delta = net.layers[-1 - i].bb(r, t) - r
                y, r = net(data, ind=len(net.layers) - 1 - i)
                t = (y + delta).detach()

        train_loss += loss.item()
        _, predicted = pred.max(1)
        _, targets = target.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        self.log({"Loss": train_loss, "Train Acc": correct / total})

    # def validation_step(self, batch, batch_idx: int):
    #     return self.shared_step(batch=batch, batch_idx=batch_idx, optimizer_idx=None)


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

    train_transforms = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = T.Compose([T.ToTensor(), cifar10_normalization(),])
    datamodule = CIFAR10DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    model = Prototype(datamodule=datamodule, hparams=hparams)
    trainer = Trainer(max_epochs=150, gpus=torch.cuda.device_count(), overfit_batches=1)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
