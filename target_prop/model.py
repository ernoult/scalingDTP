""" Pytorch Lightning version of the model from `prototype.py` with additional
optimizations.

TODO: Add callbacks that compute the jacobians and log images / stuff to wandb.
"""
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import singledispatch
from pathlib import Path
from typing import (
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import pytorch_lightning
import torch
import torchvision.transforms as T
import tqdm
import wandb
from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from simple_parsing import ArgumentParser
from simple_parsing.helpers import choice, list_field, mutable_field
from simple_parsing.helpers.hparams import (
    HyperParameters,
    categorical,
    log_uniform,
    uniform,
)
from simple_parsing.helpers.serialization import Serializable
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torchmetrics.classification import Accuracy

from .backward_layers import get_backward_equivalent
from .config import Config
from .layers import AdaptiveAvgPool2d, Conv2dReLU, ConvTranspose2dReLU, Reshape
from .sequential import TargetPropSequential


@dataclass
class OptimizerHParams(HyperParameters):
    available_optimizers: ClassVar[Dict[str, Type[Optimizer]]] = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
    }
    # BUG: Little bug here, fixing this to 'adam' during sweeps for now.
    # type: str = categorical("sgd", "adam"], default="adam", strict=True)
    type: str = choice(available_optimizers.keys(), default="adam")
    # Learning rate of the optimizer.
    lr: float = log_uniform(1e-4, 1e-1, default=5e-3)
    # Weight decay coefficient.
    weight_decay: Optional[float] = log_uniform(1e-12, 1e-5, default=1e-7)

    def make_optimizer(self, params: Iterable[nn.Parameter]) -> Optimizer:
        optimizer_class = self.available_optimizers[self.type]
        return optimizer_class(  # type: ignore
            params, lr=self.lr, weight_decay=self.weight_decay,
        )


class Model(LightningModule):
    """ Pytorch Lightning version of the prototype from `prototype.py`.
    """

    @dataclass
    class HParams(HyperParameters):
        """ Hyper-Parameters of the model.
        
        TODO: Ask @ernoult what he thinks of these priors for the hyper-parameters.
        """

        # Hyper-parameters for the forward optimizer
        forward_optim: OptimizerHParams = mutable_field(OptimizerHParams, lr=1e-3)
        # Hyper-parameters for the "backward" optimizer
        backward_optim: OptimizerHParams = mutable_field(OptimizerHParams, lr=1e-2)

        # nudging parameter: Used when calculating the first target.
        beta: float = uniform(0.01, 0.5, default=0.1)

        batch_size: int = log_uniform(
            16, 512, default=256, base=2, discrete=True
        )  # batch size

        # Max number of training epochs in total.
        max_epochs: int = 1  # TODO: Fixing this while debugging.
        # max_epochs: int = uniform(3, 50, default=10, discrete=True)

        # Max number of epochs to train for without an improvement to the validation
        # accuracy before the training is stopped.
        early_stopping_patience: int = 3

        # number of iterations on feedback weights per batch samples.
        # NOTE: At the moment, we're only performing a single iteration, and this is the
        # number of noise samples to use instead.
        feedback_training_iterations: int = uniform(1, 20, default=10)

        # seed: Optional[int] = None  # Random seed to use.
        # sym: bool = False  # sets symmetric weight initialization
        # jacobian: bool = False  # compute jacobians

        channels: List[int] = list_field(128, 512)  # tab of channels
        # channels: List[int] = uniform(32, 256, default=128, shape=2)  # tab of channels

        # NOTE: Using a single value rather than one value per layer.
        noise: float = uniform(0.001, 0.5, default=0.01)
        # noise: List[float] = list_field(0.05, 0.5)

    def __init__(self, datamodule: VisionDataModule, hparams: HParams, config: Config):
        super().__init__()
        self.hp: Model.HParams = hparams
        self.datamodule = datamodule
        self.config = config
        # self.model = Net(args=self.hparams)
        self.in_channels, self.img_h, self.img_w = datamodule.dims
        self.n_classes = datamodule.num_classes
        self.example_input_array = torch.rand(  # type: ignore
            [32, self.in_channels, self.img_h, self.img_w], device=self.device
        )
        ## Create the forward achitecture:
        # Same architecture as in the original prototype:
        # forward_net = nn.Sequential(
        #     Conv2dReLU(self.in_channels, 128, kernel_size=(5, 5), stride=(2, 2)),
        #     Conv2dReLU(128, 512, kernel_size=(5, 5), stride=(2, 2)),
        #     Reshape(target_shape=(-1)),
        #     nn.Linear(in_features=8192, out_features=10, bias=True),
        # )
        forward_net = nn.Sequential(
            Conv2dReLU(
                self.in_channels, self.hp.channels[0], kernel_size=(5, 5), stride=(2, 2)
            ),
            *(
                Conv2dReLU(
                    self.hp.channels[i - 1],
                    self.hp.channels[i],
                    kernel_size=(5, 5),
                    stride=(2, 2),
                )
                for i in range(1, len(self.hp.channels))
            ),
            # IDEA: Trying to limit the number of hidden features a bit:
            # AdaptiveAvgPool2d(output_size=(8, 8)),
            Reshape(target_shape=(-1,)),
            nn.LazyLinear(out_features=self.n_classes, bias=True)
            # nn.Linear(in_features=8192, out_features=10, bias=True),
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

        # Larger structure, with batch norm layers:
        # forward_net = nn.Sequential(
        #     # NOTE: Using this 'fused' conv + relu layer just to replicate the prototype
        #     Conv2dReLU(self.in_channels, 6, kernel_size=5, stride=1, padding=1, bias=False),
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
        # Metrics:
        self.accuracy = Accuracy()

        self.save_hyperparameters(
            {
                "hp": self.hp.to_dict(),
                "datamodule": datamodule,
                "config": self.config.to_dict(),
            }
        )
        # kwargs that will get passed to all calls to `self.log()`, just to make things
        # a bit more tidy
        self.log_kwargs: Dict = dict()  # dict(prog_bar=True)
        self.phase: str = ""
        if self.config.seed is not None:
            seed_everything(seed=self.config.seed, workers=True)
        # self.automatic_optimization = False

    def configure_optimizers(self):
        forward_optimizer = self.hp.forward_optim.make_optimizer(
            self.model.forward_parameters(),
        )
        backward_optimizer = self.hp.backward_optim.make_optimizer(
            self.model.backward_parameters(),
        )
        return [forward_optimizer, backward_optimizer]

        # IDEA: Use a learning rate scheduler?
        # from torch.optim.lr_scheduler import ExponentialLR
        # forward_scheduler = {
        #     "scheduler": ExponentialLR(forward_optimizer, 0.99),
        #     "interval": "step",  # called after each training step
        # }
        # return [forward_optimizer, backward_optimizer], [forward_scheduler]

        # NOTE: When using the 'frequency' version below, it trains once every `n`
        # batches, rather than `n` times on the same batch.
        # return (
        #     {"optimizer": forward_optimizer, "frequency": 1},
        #     {"optimizer": backward_optimizer, "frequency": 1},
        # )

    def configure_callbacks(self) -> List[Callback]:
        return [EarlyStopping("val/Accuracy", patience=self.hp.early_stopping_patience)]

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        return self.model.forward_net(input)

    def training_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int = None
    ) -> Tensor:
        loss = self.shared_step(
            batch=batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx, phase="train"
        )
        assert loss.requires_grad, (loss, optimizer_idx)
        return loss

    def validation_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
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
        # Setting this value just so we don't have to pass `phase=...` to `forward_loss`
        # and `feedback_loss` below.
        self.phase = phase

        dtype: Optional[torch.dtype] = self.dtype if isinstance(
            self.dtype, torch.dtype
        ) else None
        # The total loss to be returned.
        loss: Tensor = torch.zeros(1, device=self.device, dtype=dtype)

        # FIXME: Remove, only used to debug atm:
        # print(f"Batch id {batch_idx}, optimizer: {optimizer_idx}, phase: {phase}")
        # TODO: Do we want to use the `weight_b_normalize` function? If so, when?

        if optimizer_idx in [None, 0]:
            # Optimize the forward weights
            forward_loss = self.forward_loss(x, y)
            self.log(f"{phase}/forward loss", forward_loss, **self.log_kwargs)
            loss += forward_loss

        if optimizer_idx in [None, 1]:
            # ----------- Optimize the feedback weights -------------
            feedback_loss = self.feedback_loss(x, y)
            loss += feedback_loss
            self.log(f"{phase}/feedback loss", feedback_loss, **self.log_kwargs)

        return loss

    def forward_loss(self, x: Tensor, y: Tensor) -> Tensor:
        # Get the outputs of each layer
        ys: List[Tensor] = self.model.forward_all(x, allow_grads_between_layers=False)
        y_pred: Tensor = ys[-1]

        # NOTE: Uncomment to allow training "as usual", allowing the gradients to
        # flow all the way through:
        # y_pred = self.model.forward_net(x)
        # return F.cross_entropy(y_pred, y)

        with torch.no_grad():
            # Log the cross-entropy loss (not used for training).
            cross_entropy_loss = F.cross_entropy(y_pred, y)
            self.log(f"{self.phase}/CE Loss", cross_entropy_loss, **self.log_kwargs)

            accuracy = self.accuracy(torch.softmax(y_pred, -1), y)
            # accuracy = y_pred.argmax(-1).eq(y).sum().float().div(len(y_pred))
            self.log(f"{self.phase}/Accuracy", accuracy, **self.log_kwargs)

        # "Normalize" the prediction, which we use to calculate the first target.
        pred = torch.exp(F.log_softmax(y_pred, dim=-1))
        y_onehot = F.one_hot(y, num_classes=self.n_classes).float()

        # compute target of the last layer:
        s_n = y_pred + self.hp.beta * (y_onehot - pred)

        # Get the outputs of the backward networks
        # TODO: If we wanted to be really picky, there's one extra forward-pass
        # happening here in last layer of the backward network (which outputs the
        # 'x' equivalent.
        net_b_outputs = self.model.backward_all(s_n, allow_grads_between_layers=False)
        targets: List[Tensor] = list(reversed(net_b_outputs))
        targets.pop(0)  # Don't consider the 'target' for the first layer (x)
        targets.append(s_n)  # add the target for the last layer
        # Detach all the targets:
        targets = [target.detach() for target in targets]

        forward_losses = 0.5 * torch.stack(
            [
                F.mse_loss(y_i, t_i, reduction="mean")
                # NOTE: equivalent to:
                # 0.5 * ((y_i - t_i) ** 2).view(y_i.size(0), -1).sum(1).mean()
                for y_i, t_i in zip(ys, targets)
            ]
        )
        forward_loss = forward_losses.mean()
        return forward_loss

    def test_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        return self.shared_step(batch, batch_idx=batch_idx, phase="test")

    def current_noise_coefficient(self) -> Union[float, Tensor]:
        return self.hp.noise
        # IDEA: Gradually lower the value of `self.hp.noise` over the course of
        # training?
        coefficient = 1 - (self.current_epoch / self.trainer.max_epochs)
        coefficient = torch.clamp(coefficient, 1e-8, 1)
        return coefficient * self.hp.noise

    def feedback_loss(self, x: Tensor, y: Tensor) -> Tensor:
        # Get the outputs for all layers.
        # NOTE: no need for gradients w.r.t. forward parameters.
        with torch.no_grad():
            ys = self.model.forward_all(x)

        # Input of each intermediate layer
        xs = ys[:-1]

        # NOTE: Using ys[1:] since we don't need the `r` of the first layer (`G(x_1)`)
        # NOTE: This saves one forward-pass, but makes the code a tiny bit uglier:
        rs = self.model.backward_each(ys[1:], start_layer_index=1)

        x_noise_distributions: List[Normal] = [
            Normal(loc=torch.zeros_like(x_i), scale=self.current_noise_coefficient())
            for x_i in xs
            # Normal(loc=torch.zeros_like(x_i), scale=self.hp.noise) for x_i in xs
        ]

        # List of losses, one per iteration.
        dr_losses_list: List[Tensor] = []
        for backward_iteration in range(self.hp.feedback_training_iterations):
            # TODO: Could we get away with sampling a bunch of noise vectors, rather
            # than performing multiple update steps?
            
            # Create a noise vector to be added to the input of each intermediate
            # layer:
            # (NOTE: xs is still a list of detached tensors).

            # IDEA: Could possibly meta-learn the amount of noise to use?
            dxs = [x_noise_dist.rsample() for x_noise_dist in x_noise_distributions]
            # dxs = [self.hp.noise * torch.randn_like(x_i) for x_i in xs]

            noisy_xs = [x_i + dx_i for x_i, dx_i in zip(xs, dxs)]

            # NOTE: we save one forward-pass (as above) by ignoring the first layer.
            with torch.no_grad():
                noisy_ys = self.model.forward_each(noisy_xs, start_layer_index=1)
            noisy_xrs = self.model.backward_each(noisy_ys, start_layer_index=1)

            # Option 1: (adapted from the original `prototype.py`)
            # NOTE: dys aren't currently used, but they could be used by
            # `weight_b_normalize` if we decide to add it back.
            # dys = [y_noise - y_temp for y_noise, y_temp in zip(noisy_ys, ys[1:])]
            # drs = [
            #     x_noise_r - x_noise  for x_noise, x_noise_r in zip(noisy_xs, noisy_xrs)
            # ]
            # iteration_dr_loss = - self.hp.noise * sum(
            #     dr.view(dr.shape[0], -1).sum(1).mean()
            #     for dr in drs
            # )

            # Option 2: Using an 'unsigned' loss, in the sense that one direction isn't
            # favored compared to another.
            # NOTE: The `reduction` argument to `mse_loss` is already 'mean' by default.
            iteration_dr_loss_per_sample = torch.stack(
                [
                    F.mse_loss(x_noise_r, x_noise)
                    for x_noise, x_noise_r in zip(noisy_xs, noisy_xrs)
                ]
            )
            iteration_dr_loss = iteration_dr_loss_per_sample.mean()
            # TODO: Could perhaps do some sort of 'early stopping' here if the dr
            # loss is sufficiently small?
            dr_losses_list.append(iteration_dr_loss)

            # Inspired from this section of `prototype.py`:
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

        dr_losses = torch.stack(dr_losses_list)
        # TODO: Take the average, or the sum here?
        dr_loss: Tensor = dr_losses.sum()
        return dr_loss

