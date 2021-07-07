from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from functools import singledispatch
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pytorch_lightning
import torch
import tqdm
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from simple_parsing import ArgumentParser
from simple_parsing.helpers import list_field, mutable_field
from simple_parsing.helpers.hparams import HyperParameters
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from target_prop.backward_layers import get_backward_equivalent
from target_prop.layers import (
    AdaptiveAvgPool2d,
    Reshape,
    Conv2dReLU,
    ConvTranspose2dReLU,
)
from target_prop.sequential import TargetPropSequential


@dataclass
class OptimizerHParams:
    lr: float = 1e-3
    weight_decay: Optional[float] = 0.0


@dataclass
class Config:
    """ Configuration options for the experiment (not hyper-parameters). """

    data_dir: Path = Path("data")
    num_workers: int = 4

    epochs: int = 15  # number of epochs to train feedback weights
    batch_size: int = 128  # batch dimension
    # device to use
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Prototype(LightningModule):
    @dataclass
    class HParams(HyperParameters):
        learning_rate_f: float = 0.05
        learning_rate_b: float = 0.5
        # Hyper-parameters for the forward optimizer
        forward_optim: OptimizerHParams = mutable_field(OptimizerHParams, lr=1e-3)
        # Hyper-parameters for the "backward" optimizer
        backward_optim: OptimizerHParams = mutable_field(OptimizerHParams, lr=1e-3)

        beta: float = 0.1  # nudging parameter

        # number of iterations on feedback weights per batch samples
        feedback_training_iterations: int = 20

        seed: Optional[int] = None  # Random seed to use.
        sym: bool = False  # sets symmetric weight initialization
        jacobian: bool = False  # compute jacobians
        conv: bool = False  # select the conv architecture.
        C: List[int] = list_field(128, 512)  # tab of channels

        # NOTE: Using a single value rather than one value per layer.
        noise: float = 0.01  # tab of noise amplitude
        # noise: List[float] = list_field(0.05, 0.5)  # tab of noise amplitude

    def __init__(self, datamodule: VisionDataModule, hparams: HParams):
        super().__init__()
        self.hp: Prototype.HParams = hparams
        self.datamodule = datamodule
        # self.model = Net(args=self.hparams)
        self.channels, self.img_h, self.img_w = datamodule.dims
        self.n_classes = datamodule.num_classes
        self.example_input_array = torch.rand(
            [32, self.channels, self.img_h, self.img_w], device=self.device
        )

        # Same architecture as in the original prototype (I think)
        # forward_net = nn.Sequential(
        #     nn.Conv2d(self.channels, 128, kernel_size=(5, 5), stride=(2, 2)),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(128, 512, kernel_size=(5, 5), stride=(2, 2)),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(in_features=8192, out_features=10, bias=True)
        # )

        forward_net = nn.Sequential(
            Conv2dReLU(self.channels, 128, kernel_size=(5, 5), stride=(2, 2)),
            Conv2dReLU(128, 512, kernel_size=(5, 5), stride=(2, 2)),
            Reshape(target_shape=(-1,)),
            # nn.LazyLinear(out_features=10, bias=True)
            nn.Linear(in_features=8192, out_features=10, bias=True),
        )

        # Model using the 'fused' layers (closer to the original 'prototype'.)
        # self.model = TargetPropSequentialV1(
        #     TargetPropConv2d(channels, 16),
        #     TargetPropConv2d(16, 32),
        #     TargetPropConv2d(32, 64),
        #     # TargetPropConv2d(64, 128),
        #     # AdaptivePooling(4),
        #     nn.Flatten(),
        #     TargetPropLinear(8192, n_classes),
        # )

        # forward_net = nn.Sequential(
        #     # NOTE: Using this 'fused' conv + relu layer just to replicate the prototype
        #     Conv2dReLU(self.channels, 6, kernel_size=5, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(6),
        #     Conv2dReLU(6, 16, kernel_size=5, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(16),
        #     Conv2dReLU(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     # AdaptiveAvgPool2d(output_size=(8, 8)),  # [16, 8, 8]
        #     # nn.BatchNorm2d(16),
        #     Conv2dReLU(
        #         16, 32, kernel_size=3, stride=1, padding=0, bias=False
        #     ),  # [32, 6, 6]
        #     # nn.BatchNorm2d(32),
        #     Conv2dReLU(
        #         32, 32, kernel_size=3, stride=1, padding=0, bias=False
        #     ),  # [32, 4, 4]
        #     # nn.BatchNorm2d(32),
        #     Reshape(source_shape=(32, 4, 4), target_shape=(512,)),
        #     # Reshape(source_shape=(64, 4, 4), target_shape=(64 * 4 * 4,)),
        #     nn.Linear(in_features=512, out_features=self.n_classes),
        # )
        example_out: Tensor = forward_net(self.example_input_array)
        out_shape = example_out.shape
        assert example_out.requires_grad

        # Construct the feedback/"backward" network, one layer at a time, using the
        # generic `get_backward_equivalent` function.
        backward_net = nn.Sequential(
            *[
                get_backward_equivalent(forward_layer)
                for forward_layer in reversed(forward_net)
            ]
        )
        example_in_hat: Tensor = backward_net(example_out)
        assert example_in_hat.requires_grad
        assert example_in_hat.shape == self.example_input_array.shape

        self.model = TargetPropSequential(
            forward_layers=forward_net, backward_layers=backward_net,
        )
        # self.automatic_optimization = False

    def configure_optimizers(self):
        forward_optimizer = torch.optim.Adam(
            self.model.forward_parameters(),
            **asdict(self.hp.forward_optim),
            # momentum=0.9,
            # weight_decay=self.hp.lamb,
        )
        backward_optimizer = torch.optim.Adam(
            self.model.backward_parameters(),
            **asdict(self.hp.backward_optim),
            # momentum=0.9,
            # weight_decay=self.hp.lamb,
        )
        # TODO: Figure out a clean way to use one optimizers repeatedly on the same
        # batch in pytorch-lightning.
        # TODO: How about we use this frequency instead of the for loop in training_step?
        # return [forward_optimizer] + [backward_optimizer] * 20
        return [forward_optimizer, backward_optimizer]
        return (
            {"optimizer": forward_optimizer, "frequency": 1},
            {"optimizer": backward_optimizer, "frequency": 1},
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.model.forward_net(input)

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int = None
    ):
        loss = self.shared_step(
            batch=batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx, phase="train"
        )
        assert loss.requires_grad, (loss, optimizer_idx)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        return self.shared_step(
            batch=batch, batch_idx=batch_idx, optimizer_idx=None, phase="val"
        )

    def shared_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        phase: str,
        optimizer_idx: Optional[int] = None,
    ):
        """ Main step, used by `training_step` and `validation_step`.
        """
        x, y = batch

        dtype: Optional[torch.dtype] = self.dtype if isinstance(
            self.dtype, torch.dtype
        ) else None
        # The total loss to be returned.
        loss: Tensor = torch.zeros(1, device=self.device, dtype=dtype)

        # kwargs to the `self.log()` calls, just to make things a bit more tidy
        log_kwargs: Dict = dict(on_step=True, prog_bar=True, on_epoch=False)

        # FIXME: Remove, only used to debug atm:
        # print(f"Batch id {batch_idx}, optimizer: {optimizer_idx}, phase: {phase}")

        if optimizer_idx in [None, 0]:
            # Optimize the forward weights

            # Get the outputs of each layer
            ys: List[Tensor] = self.model.forward_all(
                x, allow_grads_between_layers=False
            )
            y_pred: Tensor = ys[-1]

            # NOTE: Uncomment to allow training "as usual", allowing the gradients to
            # flow all the way through:
            # y_pred = self.model.forward_net(x)
            # return F.cross_entropy(y_pred, y)

            with torch.no_grad():
                # Log the cross-entropy loss (not used for training).
                cross_entropy_loss = F.cross_entropy(y_pred, y)
                self.log("CE Loss", cross_entropy_loss, **log_kwargs)

                accuracy = y_pred.argmax(-1).eq(y).sum().float().div(len(y_pred))
                self.log("Accuracy", accuracy, **log_kwargs)

            # "Normalize" the prediction, which we use to calculate the first target.
            pred = torch.exp(F.log_softmax(y_pred, dim=-1))
            y_onehot = F.one_hot(y, num_classes=self.n_classes).float()

            # compute target of the last layer:
            s_n = y_pred + self.hp.beta * (y_onehot - pred)

            # Get the outputs of the backward networks
            # TODO: If we wanted to be really picky, there's one extra forward-pass
            # happening here in last layer of the backward network (which outputs the
            # 'x' equivalent.
            net_b_outputs = self.model.backward_all(
                s_n, allow_grads_between_layers=False
            )
            targets: List[Tensor] = list(reversed(net_b_outputs))
            targets.pop(0)  # Don't consider the 'target' for the first layer (x)
            targets.append(s_n)  # add the target for the last layer
            # Detach all the targets:
            targets = [target.detach() for target in targets]

            forward_losses: List[Tensor] = [
                0.5 * F.mse_loss(y_i, t_i, reduction="mean")
                # NOTE: equivalent to:
                # 0.5 * ((y_i - t_i) ** 2).view(y_i.size(0), -1).sum(1).mean()
                for y_i, t_i in zip(ys, targets)
            ]
            forward_loss = sum(forward_losses)
            self.log("Floss", forward_loss, **log_kwargs)
            loss += forward_loss

        if optimizer_idx in [None, 1]:
            # ----------- Optimize the feedback weights -------------

            # TODO: Do we want to use the `weight_b_normalize` function? If so, when?

            # Get the outputs for all layers.
            # NOTE: no need for gradients w.r.t. forward parameters.
            with torch.no_grad():
                ys = self.model.forward_all(x)

            # TODO: If we wanted to be a bit picky, we don't need the last `r`, (the r
            # for the first x)
            # rs = self.model.backward_each(ys, forward_ordering=True)
            # rs.pop(0)
            # NOTE: This saves one forward-pass, but makes the code uglier:
            rs = self.model.backward_each(ys[1:], start_layer_index=1)

            # NOTE: This purposefully doesn't include 'x' and the last output.

            # Create a noise vector to be added to the input of each intermediate layer:
            # (NOTE: xs is still a list of detached tensors).
            xs = ys[:-1]
            dxs = [self.hp.noise * torch.randn_like(x_i) for x_i in xs]

            noisy_xs = [x_i + dx_i for x_i, dx_i in zip(xs, dxs)]
            # NOTE: we save one forward-pass (as above) by ignoring the first layer.
            with torch.no_grad():
                noisy_ys = self.model.forward_each(noisy_xs, start_layer_index=1)
            noisy_xrs = self.model.backward_each(noisy_ys, start_layer_index=1)

            dys = [y_noise - y_temp for y_noise, y_temp in zip(noisy_ys, ys[1:])]
            drs = [
                x_noise - x_noise_r for x_noise, x_noise_r in zip(noisy_xs, noisy_xrs)
            ]

            dr_loss = sum(
                -self.hp.noise * dr.view(dr.shape[0], -1).sum(1).mean()
                for dr in drs
            )
            loss += dr_loss
            self.log(
                "dr_loss", dr_loss, **log_kwargs
            )
            
            # for id_layer in range(len(self.model.ba.layers) - 1):
            #     y_temp, r_temp = net.layers[id_layer + 1](y, back = True)
            #     noise = args.noise[id_layer]*torch.randn_like(y)
            #     y_noise, r_noise = net.layers[id_layer + 1](y + noise, back = True)
            #     dy = (y_noise - y_temp)
            #     dr = (r_noise - r_temp)

            #     loss_b = -(noise*dr).view(dr.size(0), -1).sum(1).mean()

            #     optimizer_b.zero_grad()

            #     if iter < args.iter:
            #         loss_b.backward(retain_graph = True)
            #     else:
            #         loss_b.backward()

            #     optimizer_b.step()

        return loss


def main():
    from simple_parsing import ArgumentParser

    parser = ArgumentParser(description="Pytorch lightning version of the prototype")
    parser.add_arguments(Config, dest="config")
    parser.add_arguments(Prototype.HParams, dest="hparams")

    args = parser.parse_args()

    config: Config = args.config
    hparams: Prototype.HParams = args.hparams
    import torchvision.transforms as T
    from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
    from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

    def make_mnist_dm(config: Config) -> MNISTDataModule:
        return MNISTDataModule(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

    def make_cifar10_dm(config: Config) -> CIFAR10DataModule:
        train_transforms = T.Compose(
            [
                T.RandomCrop(28, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                cifar10_normalization(),
            ]
        )
        test_transforms = T.Compose([T.ToTensor(), cifar10_normalization()])
        return CIFAR10DataModule(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            val_transforms=test_transforms,
        )

    datamodule = make_mnist_dm(config=config)
    # datamodule = make_cifar10_dm(config=config)
    model = Prototype(datamodule=datamodule, hparams=hparams)
    trainer = Trainer(
        max_epochs=150, gpus=torch.cuda.device_count(), overfit_batches=1
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
