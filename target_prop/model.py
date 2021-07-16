""" Pytorch Lightning version of the model from `prototype.py` with additional
optimizations.

TODO: Add callbacks that compute the jacobians and log images / stuff to wandb.
TODO: Fix the target calculation, which doesn't currently use the `r` term!
"""
# from __future__ import annotations
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import singledispatch
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR

import warnings
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
from target_prop.layers import ConvPoolBlock
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
from .layers import (
    AdaptiveAvgPool2d,
    # Conv2dELU,
    # ConvTranspose2dELU,
    Reshape,
    Sequential,
)
from .feedback_loss import feedback_loss
from .utils import flag
from logging import getLogger

logger = getLogger(__file__)


@dataclass
class OptimizerHParams(HyperParameters):
    available_optimizers: ClassVar[Dict[str, Type[Optimizer]]] = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
    }

    # Type of Optimizer to use.
    type: str = choice(available_optimizers.keys(), default="sgd")
    # BUG: Little bug here, won't search over this in sweeps for now.
    # type: str = categorical("sgd", "adam"], default="adam", strict=True)

    # Learning rate of the optimizer.
    lr: List[float] = log_uniform(1e-4, 1e-1, default=5e-3, shape=2)
    # Weight decay coefficient.
    weight_decay: Optional[float] = log_uniform(1e-9, 1e-2, default=1e-4)

    use_lr_scheduler: bool = False

    def make_optimizer(self, network: nn.Sequential) -> Optimizer:
        optimizer_class = self.available_optimizers[self.type]
        # List of learning rates for each layer.
        n_layers = len(network)
        lrs: List[float] = get_list_of_values(self.lr, out_length=n_layers, name="lr")
        assert len(lrs) == n_layers
        params: List[Dict] = []
        for layer, lr in zip(network, lrs):
            params.append({"params": layer.parameters(), "lr": lr})
        return optimizer_class(  # type: ignore
            params, weight_decay=self.weight_decay,
        )


def default_feedback_loss_fn(x: Tensor, xr: Tensor, noise_scale: float) -> Tensor:
    dr = xr - x
    return -noise_scale * dr.view(dr.size(0), -1).sum(1).mean()


def mse_feedback_loss(x: Tensor, xr: Tensor, noise_scale: float) -> Tensor:
    return noise_scale * F.mse_loss(x, xr)


class Model(LightningModule):
    """ Pytorch Lightning version of the prototype from `prototype.py`.
    """

    @dataclass
    class HParams(HyperParameters):
        """ Hyper-Parameters of the model.

        TODO: Set these values as default (cifar10)
        ```console
        python main.py --batch-size 128 \
        --C 128 128 256 256 512 \
        --iter 20 30 35 55 20 \
        --epochs 90 \
        --lr_b 1e-4 3.5e-4 8e-3 8e-3 0.18 \
        --noise 0.4 0.4 0.2 0.2 0.08 \
        --lr_f 0.08 \
        --beta 0.7 \
        --path CIFAR-10 \
        --scheduler \
        --wdecay 1e-4 \
        ```
        """

        # Channels per conv layer.
        channels: List[int] = list_field(128, 128, 256, 256, 512)

        # Number of training steps for the feedback weights per batch. Can be a list of
        # integers, where each value represents the number of iterations for that layer.
        feedback_training_iterations: List[int] = list_field(20, 30, 35, 55, 20)

        # Number of noise samples to use to get the feedback loss in a single iteration.
        # NOTE: The loss used for each update is the average of these losses.
        feedback_samples_per_iteration: int = uniform(1, 20, default=1)

        # Max number of training epochs in total.
        max_epochs: int = 90

        # Hyper-parameters for the forward optimizer
        # TODO: Usign 0.1 0.2 0.3 seems to be gettign much better results (75% after 1 epoch on MNIST)
        f_optim: OptimizerHParams = mutable_field(
            OptimizerHParams, type="sgd", lr=0.08, use_lr_scheduler=True
        )
        # Hyper-parameters for the "backward" optimizer
        b_optim: OptimizerHParams = mutable_field(
            OptimizerHParams, type="sgd", lr=[1e-4, 3.5e-4, 8e-3, 8e-3, 0.18]
        )

        # nudging parameter: Used when calculating the first target.
        beta: float = uniform(0.1, 1.0, default=0.7)
        # batch size
        batch_size: int = log_uniform(16, 512, default=128, base=2, discrete=True)

        # Max number of epochs to train for without an improvement to the validation
        # accuracy before the training is stopped.
        early_stopping_patience: int = 3

        # seed: Optional[int] = None  # Random seed to use.
        # sym: bool = False  # sets symmetric weight initialization
        # jacobian: bool = False  # compute jacobians

        # NOTE: Using a single value rather than one value per layer.
        noise: List[float] = uniform(
            0.001, 0.5, default_factory=[0.4, 0.4, 0.2, 0.2, 0.08].copy, shape=5
        )

        # Wether to update the feedback weights before the forward weights.
        # TODO: This is different from in `prototype.py`
        feedback_before_forward: bool = flag(True)

        activation: Type[nn.Module] = choice(
            {"relu": nn.ReLU, "elu": nn.ELU,}, default="elu"
        )

        feedback_weight_training_procedure: str = choice(
            "awesome", "sequential", "best_of_both_worlds", default="sequential",
        )
        """
        What kind of precedure to use to train the feedback weights.
        - 'awesome':
            Fully parallel training with multiple noise samples and only one iteration.
            As far as I know, this is the only procedure that can easily be scaled to
            multi-GPU with DP/DDP through Pytorch-Lightning.
        - 'sequential':
            Fully sequential: multiple noise samples per iteration, multiple iterations,
            possibly different number of iterations per layer. This is the same
            configuration as in @ernoult's implementation. Very slow to run.
        - 'best_of_both_worlds':
            Fully parallel, with possibly multiple noise iterations per layer. This has
            some of the benefits of `awesome` (layer-wise parallel training), while also
            preserving the ability to set a different number of updates per layer.
        """

    def __init__(self, datamodule: VisionDataModule, hparams: HParams, config: Config):
        super().__init__()
        self.hp: Model.HParams = hparams
        self.datamodule = datamodule
        self.config = config
        if self.config.seed is not None:
            seed_everything(seed=self.config.seed, workers=True)

        self.in_channels, self.img_h, self.img_w = datamodule.dims
        self.n_classes = datamodule.num_classes
        self.example_input_array = torch.rand(  # type: ignore
            [datamodule.batch_size, *datamodule.dims], device=self.device
        )
        ## Create the forward achitecture:
        self.forward_net = Sequential(
            ConvPoolBlock(
                in_channels=self.in_channels,
                out_channels=self.hp.channels[0],
                activation_type=self.hp.activation,
                input_shape=datamodule.dims,
            ),
            *(
                ConvPoolBlock(
                    in_channels=self.hp.channels[i - 1],
                    out_channels=self.hp.channels[i],
                    activation_type=self.hp.activation,
                )
                for i in range(1, len(self.hp.channels))
            ),
            Reshape(target_shape=(-1,)),
            # NOTE: Using LazyLinear so we don't have to know the hidden size in advance
            nn.LazyLinear(out_features=self.n_classes, bias=True),
        )
        # Pass an example input through the forward net so that all the layers which
        # need to know their inputs/output shapes get a chance to know them.
        # This is necessary for creating the backward network, as some layers
        # (e.g. Reshape) need to know what the input shape is.
        example_out: Tensor = self.forward_net(self.example_input_array)
        out_shape = example_out.shape

        if self.config.debug:
            print(f"Forward net: ")
            print(self.forward_net)

        assert example_out.requires_grad
        # Construct the feedback/"backward" network, one layer at a time, using the
        # generic `get_backward_equivalent` function.
        self.backward_net = Sequential(
            *[
                get_backward_equivalent(forward_layer)
                for forward_layer in reversed(self.forward_net)
            ]
        )

        if self.config.debug:
            print(f"Backward net: ")
            print(self.backward_net)
        # Pass the output of the forward net for the `example_input_array` through the
        # backward net, to check that the backward net is indeed able to recover the
        # inputs (at least in terms of their shape for now).
        example_in_hat: Tensor = self.backward_net(example_out)
        assert example_in_hat.requires_grad
        assert example_in_hat.shape == self.example_input_array.shape
        assert example_in_hat.dtype == self.example_input_array.dtype
        # Metrics:
        self.accuracy = Accuracy()

        if isinstance(self.hp.feedback_training_iterations, list):
            # Reverse the list in-place, Since it will be passed as [G0, ... GN], while
            # the backward net is defined as [GN, ..., G0].
            self.hp.feedback_training_iterations.reverse()

        if isinstance(self.hp.noise, list):
            # Reverse the list in-place, Since it will be passed as [G0, ... GN], while
            # the backward net is defined as [GN, ..., G0].
            self.hp.noise.reverse()

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

        # Set this to False when we will need to perform multiple feedback weight
        # updates per batch, to indicate that we will perform the update ourselves
        # rather than by letting pytorch-lightning do it automatically. :'(
        self.automatic_optimization = self.hp.feedback_training_iterations == 1 or (
            isinstance(self.hp.feedback_training_iterations, list)
            and all(i == 1 for i in self.hp.feedback_training_iterations)
        )

        # Index of the optimizers. Gets influenced by the value of `feedback_before_forward`
        self._feedback_optim_index: int = 0 if self.hp.feedback_before_forward else 1
        self._forward_optim_index: int = 1 if self.hp.feedback_before_forward else 0

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        return self.forward_net(input)

    def training_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int = None
    ) -> Tensor:
        loss = self.shared_step(
            batch=batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx, phase="train"
        )
        if self.automatic_optimization:
            # Should have a loss with gradients if we're using automatic optimization
            # from PL.
            assert False, "debugging, this shouldn't be encountered for now."
            assert self.hp.feedback_weight_training_procedure != "sequential"
            assert loss.requires_grad, (loss, optimizer_idx)
            return loss
        elif isinstance(loss, Tensor):
            # Need to NOT return a Tensor when not using automatic optimization.
            # BUG: Pytorch Lightning complains that we're returning a Tensor, even if
            # it's a float!
            return float(loss.item())
            return None
        return loss

    def training_step_end(self, step_results: Union[Tensor, List[Tensor]]) -> Tensor:
        """ See the `training_step_end` of pytorch-lightning for more info. """
        # TODO: For now we're kinda losing the logs and stuff that happens within the
        # workers in DP (they won't show up in the progress bar for instance).
        # merged_step_results = {
        #     k: sum(v_i.to(self.device) for v_i in v)
        #     for k, v in step_results
        # }
        step_result = (
            step_results if isinstance(step_results, (Tensor, float)) else sum(step_results)
        )

        # TODO: If NOT in automatic differentiation, but still in a scenario where we
        # can do a single update, do it here.
        loss = step_result
        self.log(f"{self.phase}/total loss", loss, prog_bar=True)

        if not self.automatic_optimization and isinstance(loss, Tensor) and loss.requires_grad:
            optimizers = self.optimizers()
            forward_optimizer = optimizers[self._forward_optim_index]
            backward_optimizer = optimizers[self._feedback_optim_index]
            forward_optimizer.zero_grad()
            backward_optimizer.zero_grad()

            self.manual_backward(loss)

            forward_optimizer.step()
            backward_optimizer.step()

            return float(loss)
        elif not self.automatic_optimization:
            return float(step_result)
        assert self.automatic_optimization
        return step_result

    def validation_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        return self.shared_step(
            batch=batch, batch_idx=batch_idx, optimizer_idx=None, phase="val"
        )

    def test_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        return self.shared_step(batch, batch_idx=batch_idx, phase="test")

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

        # TODO: Do we want to use the `weight_b_normalize` function? If so, when?

        if optimizer_idx in [None, self._feedback_optim_index]:
            # ----------- Optimize the feedback weights -------------
            feedback_loss = self.feedback_loss(x, y)
            loss += feedback_loss
            self.log(f"{phase}/feedback loss", feedback_loss, **self.log_kwargs)

        if optimizer_idx in [None, self._forward_optim_index]:
            # ----------- Optimize the forward weights -------------
            forward_loss = self.forward_loss(x, y)
            self.log(f"{phase}/forward loss", forward_loss, **self.log_kwargs)
            loss += forward_loss

        return loss

    def forward_loss(self, x: Tensor, y: Tensor) -> Tensor:
        # Get the outputs of each layer
        ys: List[Tensor] = self.forward_net.forward_all(
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
            self.log(f"{self.phase}/CE Loss", cross_entropy_loss, **self.log_kwargs)

            accuracy = self.accuracy(torch.softmax(y_pred, -1), y)
            # accuracy = y_pred.argmax(-1).eq(y).sum().float().div(len(y_pred))
            self.log(f"{self.phase}/Accuracy", accuracy, prog_bar=True)

        # "Normalize" the prediction, which we use to calculate the first target.
        pred = torch.exp(F.log_softmax(y_pred, dim=-1))
        y_onehot = F.one_hot(y, num_classes=self.n_classes).float()

        # compute target of the last layer:
        s_n = y_pred + self.hp.beta * (y_onehot - pred)

        # Get the outputs of the backward networks (the targets for all the previous
        # layers)
        with torch.no_grad():
            # NOTE: We don't compute the 'target' for the first layer (x), which is the
            # last layer of the backward net.
            net_b_outputs = self.backward_net[:-1].forward_all(s_n)
            # TODO: Need to fix this target calculation, as it doesn't currently use the
            # `r` terms!
            targets: List[Tensor] = list(reversed(net_b_outputs))
            targets.append(s_n.detach())  # add the target for the last layer

        forward_losses = 0.5 * torch.stack(
            [
                F.mse_loss(y_i, t_i)
                # NOTE: equivalent to:
                # 0.5 * ((y_i - t_i) ** 2).view(y_i.size(0), -1).sum(1).mean()
                for y_i, t_i in zip(ys, targets)
            ]
        )
        forward_loss = forward_losses.mean()
        # NOTE: `forward_loss` might not require gradients, for instance during
        # validation.
        if self.phase != "train":
            assert not forward_loss.requires_grad
        if not self.automatic_optimization and forward_loss.requires_grad:
            optimizer = self.optimizers()[self._forward_optim_index]
            optimizer.zero_grad()
            self.manual_backward(forward_loss)
            optimizer.step()

        return forward_loss

    def feedback_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """ Feedback weight training
        
        WIP: Can switch between different ways of training the feedback weights using
        the `sequential_feedback_training` and `feedbacK_training_iterations`
        hyper-parameters.
        
        NOTE: Only the 'full_parallel' version will work with DDP.
        """
        procedures = {
            "sequential": self.train_feedback_sequential,
            "awesome": self.train_feedback_full_parallel,
            "best_of_both_worlds": self.train_feedback_best_of_both_worlds,
        }
        procedure_to_use = procedures[self.hp.feedback_weight_training_procedure]
        return procedure_to_use(x, y)

    def train_feedback_sequential(self, x: Tensor, y: Tensor) -> Tensor:
        backward_optimizer = self.optimizers()[self._feedback_optim_index]

        # 1- Compute the first hidden layer (no grad).
        with torch.no_grad():
            y = self.forward_net[0](x)

        n_layers = len(self.backward_net)

        # Expand the 'noise' hparam to a list of values for each *trained* layer.
        noise_scale_per_layer = self._get_noise_scale_per_layer()
        iterations_per_layer = self._get_iterations_per_layer()

        # Reverse the backward net, just for ease of readability:
        reversed_backward_net = self.backward_net[::-1]

        layer_losses: List[Tensor] = []
        # 2- Layer-wise autoencoder training begins:
        for layer_index in range(1, n_layers):
            forward_layer = self.forward_net[layer_index]
            feedback_layer = reversed_backward_net[layer_index]
            iterations = iterations_per_layer[layer_index]
            noise_scale = noise_scale_per_layer[layer_index]

            if iterations == 0:
                # Skip this layer, since it's not trainable, but still compute the `y`
                # for the next layer.
                with torch.no_grad():
                    y = forward_layer(y)
                continue

            losses_per_iteration: List[Tensor] = []
            for iteration in range(iterations):
                # 3- Train the current autoencoder:
                loss = feedback_loss(
                    feedback_layer,
                    forward_layer,
                    input=y,
                    noise_scale=noise_scale,
                    noise_samples=self.hp.feedback_samples_per_iteration,
                )
                if loss.requires_grad:
                    backward_optimizer.zero_grad()
                    self.manual_backward(loss)
                    backward_optimizer.step()
                    losses_per_iteration.append(loss.detach())
                else:
                    losses_per_iteration.append(loss)

            iteration_losses = torch.stack(losses_per_iteration)
            total_loss_for_layer = iteration_losses.sum()
            layer_losses.append(total_loss_for_layer)
            self.log(
                f"{self.phase}/Feedback Loss [{layer_index}]", total_loss_for_layer
            )
            self.log(
                f"{self.phase}/Feedback Loss iterations [{layer_index}]", iterations
            )
            # logger.debug(
            #     f"layer {layer_index}: losses per iteration: {losses_per_iteration}"
            # )
            # 4- Compute the next hidden layer
            with torch.no_grad():
                y = forward_layer(y)

        return torch.stack(layer_losses).sum()

    def train_feedback_full_parallel(self, x: Tensor, y: Tensor):
        # Get the outputs for all layers.
        # NOTE: no need for gradients w.r.t. forward parameters.
        with torch.no_grad():
            ys = self.forward_net.forward_all(x)

        # Input of each intermediate layer
        xs = ys[:-1]
        # Inputs to each backward layer:
        # NOTE: Using ys[1:] since we don't need the `r` of the first layer (`G(x_1)`)
        yr = ys[1:]
        # Reverse the backward net, just to make the code a bit easier to read:
        reversed_backward_net = self.backward_net[::-1]
        # NOTE: This saves one forward-pass, but makes the code a tiny bit uglier:

        rs = reversed_backward_net[1:].forward_each(ys[1:])

        noise_scale_vector = self.get_noise_scale_per_layer(reversed_backward_net[1:])
        x_noise_distributions: List[Normal] = [
            Normal(loc=torch.zeros_like(x_i), scale=noise_i)
            for x_i, noise_i in zip(xs, noise_scale_vector)
        ]

        # NOTE: Notation for the tensor shapes below:
        # I: number of training iterations
        # S: number of noise samples
        # N: number of layers
        # B: batch dimension
        # X_i: shape of the inputs to layer `i`

        # List of losses, one per iteration.
        # If this is called during training, and there is more than one iteration, then
        # the losses in this list will be detached.
        dr_losses_list: List[Tensor] = []  # [I]

        n_layers = len(self.backward_net)
        if not isinstance(self.hp.feedback_training_iterations, int):
            assert len(set(self.hp.feedback_training_iterations)) == 1
            self.hp.feedback_training_iterations = self.hp.feedback_training_iterations[0]

        iterations: int = self.hp.feedback_training_iterations
        # Only do one iteration when evaluating or testing.
        if self.phase != "train":
            iterations = 1

        noise_scale_per_layer = self._get_noise_scale_per_layer()

        for iteration in range(iterations):
            # List of losses, one per noise sample.
            dr_losses_per_sample: List[Tensor] = []  # [S]

            feedback_losses = [
                feedback_loss(
                    reversed_backward_net[i],
                    self.forward_net[i],
                    noise_scale=noise_scale_per_layer[i],
                    noise_samples=self.hp.feedback_,
                )
                if is_trainable(reversed_backward_net[i]) else 0
                for i in range(n_layers)
            ]
            loss = torch.stack(feedback_losses)
            iteration_dr_loss = iteration_dr_loss.sum(1).mean()  # [1]

            # TODO: Could perhaps do some sort of 'early stopping' here if the dr
            # loss is sufficiently small?
            if self.phase != "train":
                assert not iteration_dr_loss.requires_grad
            if not self.automatic_optimization and iteration_dr_loss.requires_grad:
                optimizer = self.optimizers()[self._feedback_optim_index]
                optimizer.zero_grad()
                self.manual_backward(iteration_dr_loss)
                optimizer.step()
                iteration_dr_loss = iteration_dr_loss.detach()
            self.log(
                f"{self.phase}/dr_loss_{iteration}",
                iteration_dr_loss,
                prog_bar=False,
                logger=True,
            )

            dr_losses_list.append(iteration_dr_loss)

        dr_losses = torch.stack(dr_losses_list)
        # TODO: Take the average, or the sum here?
        dr_loss: Tensor = dr_losses.sum()
        return dr_loss

    def train_feedback_best_of_both_worlds(self, x: Tensor, y: Tensor):
        """ WIP: IDEA: "best of both worlds" way to train the feedback weights:
        
        layers are trained in parallel, but there can also be a different number of
        iterations per layer:
        
        Pseudocode:
        ```
        layers_left_to_train: List[int] = list(range(N))
        iterations = 0
        while layers_left_to_train:
            #  train selected feedback layers in parallel
            ...
            iterations += 1
            # Keep training layers that haven't had enough updates
            layers_left_to_train = [
                i
                for i, n_updates_required_for_layer_i in enumerate(
                    self.hp.feedback_training_iterations
                )
                if n_updates_required_for_layer_i > iterations
            ]
        ```
        """
        # Get the outputs for all layers.
        # NOTE: no need for gradients w.r.t. forward parameters.
        with torch.no_grad():
            ys = self.forward_net.forward_all(x)

        # Input of each intermediate layer
        xs = ys[:-1]
        # Inputs to each backward layer:
        # NOTE: Using ys[1:] since we don't need the `r` of the first layer (`G(x_1)`)
        yr = ys[1:]
        # Reverse the backward net, just to make the code a bit easier to read:
        reversed_backward_net = self.backward_net[::-1]
        # NOTE: This saves one forward-pass, but makes the code a tiny bit uglier:

        rs = reversed_backward_net[1:].forward_each(ys[1:])

        noise_scale_vector = self.get_noise_scale_per_layer(reversed_backward_net[1:])
        x_noise_distributions: List[Normal] = [
            Normal(loc=torch.zeros_like(x_i), scale=noise_i)
            for x_i, noise_i in zip(xs, noise_scale_vector)
        ]

        n_layers = len(self.backward_net)
        iterations: List[int] = get_list_of_values(
            self.hp.feedback_training_iterations,
            out_length=n_layers - 1,
            name="iterations",
        )

        layers_left_to_train = list(i for i in range(n_layers) if iterations[i] > 0)
        # Again, we never train the first feedback layer.
        assert 0 not in layers_left_to_train
        iterations = 0

        while layers_left_to_train:

            # List of losses, one per noise sample.
            dr_losses_per_sample: List[Tensor] = []  # [S]
            x_noise_dists = [
                x_noise_distributions[layer_index]
                for layer_index in layers_left_to_train
            ]

            for sample_index in range(self.hp.feedback_samples_per_iteration):
                # Create a noise vector to be added to the input of each intermediate
                # layer:
                # (NOTE: xs is still a list of detached tensors).
                # IDEA: Could possibly meta-learn the amount of noise to use?
                dxs = [x_noise_dist.rsample() for x_noise_dist in x_noise_distributions]
                # dxs = [self.hp.noise * torch.randn_like(x_i) for x_i in xs]

                noisy_xs = [x_i + dx_i for x_i, dx_i in zip(xs, dxs)]  # [N, B, *X_i]

                # NOTE: we save one forward-pass (as above) by ignoring the first layer.
                with torch.no_grad():
                    # [N, B, *X_{i+1}]
                    noisy_ys = self.forward_net[1:].forward_each(noisy_xs)
                # [N, B, *X_i]
                noisy_xrs = reversed_backward_net[1:].forward_each(noisy_ys)

                # NOTE: dys aren't currently used, but they could be used by
                # `weight_b_normalize` if we decide to add it back, or passed as an
                # input to the loss function.
                # # [N, B, *X_{i+1}]
                # dys = [
                #     y_noise - y_temp  # [B, *X_{i+1}]
                #     for y_noise, y_temp in zip(noisy_ys, ys[1:])
                # ]

                assert len(noise_scale_vector) == len(noisy_xs) == len(noisy_xrs)
                # Use the chosen loss function to get a loss for each backward layer:
                sample_dr_loss_per_layer = torch.stack(
                    [
                        # F.mse_loss(x_noise, x_noise_r)
                        # (x_noise_r - x_noise).flatten(1).sum(1)
                        self.dr_noise_loss_function(
                            x=x_noise, xr=x_noise_r, noise_scale=noise_scale
                        )
                        for noise_scale, x_noise, x_noise_r in zip(
                            noise_scale_vector, noisy_xs, noisy_xrs
                        )
                    ]
                )
                # TODO: Should we sum or mean the per-layer dr losses ?
                dr_losses_per_sample.append(sample_dr_loss_per_layer)

            # Stack along a new first dimension, and then average
            iteration_dr_losses = torch.stack(dr_losses_per_sample)  # [S, N]
            iteration_dr_loss = iteration_dr_losses.sum(1).mean()  # [1]

            # TODO: Could perhaps do some sort of 'early stopping' here if the dr
            # loss is sufficiently small?
            if self.phase != "train":
                assert not iteration_dr_loss.requires_grad
            if not self.automatic_optimization and iteration_dr_loss.requires_grad:
                optimizer = self.optimizers()[self._feedback_optim_index]
                optimizer.zero_grad()
                self.manual_backward(iteration_dr_loss)
                optimizer.step()
                iteration_dr_loss = iteration_dr_loss.detach()
            self.log(
                f"{self.phase}/dr_loss_{iteration}",
                iteration_dr_loss,
                prog_bar=False,
                logger=True,
            )
            dr_losses_list.append(iteration_dr_loss)

    def get_noise_scale_per_layer(self, network: nn.Sequential) -> Tensor:
        n_layers = len(network)
        noise: List[float] = get_list_of_values(
            self.hp.noise, out_length=n_layers, name="noise"
        )
        assert len(noise) == n_layers
        return noise
        # IDEA: Gradually lower the value of `self.hp.noise` over the course of
        # training?
        coefficient = 1 - (self.current_epoch / self.trainer.max_epochs)
        coefficient = torch.clamp(coefficient, 1e-8, 1)
        return coefficient * noise

    def configure_optimizers(self):
        backward_optimizer = self.hp.b_optim.make_optimizer(self.backward_net)
        forward_optimizer = self.hp.f_optim.make_optimizer(self.forward_net)

        forward_optim_config = {"optimizer": forward_optimizer}
        backward_optim_config = {"optimizer": backward_optimizer}

        if self.hp.f_optim.use_lr_scheduler:
            # `main.py` seems to be using a weight scheduler only for the forward weight
            # training.
            forward_optim_config["lr_scheduler"] = {
                "scheduler": CosineAnnealingLR(
                    forward_optimizer, T_max=85, eta_min=1e-5
                ),
                "interval": "epoch",  # called after each training epoch
                "frequency": 1,
            }

        if self.hp.b_optim.use_lr_scheduler:
            # `main.py` seems to be using a weight scheduler only for the forward weight
            # training.
            backward_optim_config["lr_scheduler"] = {
                "scheduler": CosineAnnealingLR(
                    backward_optimizer, T_max=85, eta_min=1e-5
                ),
                "interval": "epoch",  # called after each training epoch
                "frequency": 1,
            }

        if self.hp.feedback_before_forward:
            return [
                backward_optim_config,
                forward_optim_config,
            ]
        else:
            return [
                forward_optim_config,
                backward_optim_config,
            ]
        # NOTE: When using the 'frequency' version below, it trains once every `n`
        # batches, rather than `n` times on the same batch.
        # return (
        #     {"optimizer": forward_optimizer, "frequency": 1},
        #     {"optimizer": backward_optimizer, "frequency": 1},
        # )

    def configure_callbacks(self) -> List[Callback]:
        return [EarlyStopping("val/Accuracy", patience=self.hp.early_stopping_patience)]

    def _get_iterations_per_layer(self) -> List[int]:
        """ Returns the number of iterations to perform for each of the feedback layers.

        NOTE: Returns it in the same order as the backward_net, i.e. [GN ... G0]
        """
        if self.hp.feedback_weight_training_procedure == "awesome":
            return [1 for _ in self.backward_net]
        assert self.hp.feedback_weight_training_procedure in [
            "sequential",
            "best_of_both_worlds",
        ]
        # TODO: Make the number of iterations align directly with the trainable layers.
        # This is required because there may be some layers (e.g. Reshape), which are
        # present in the architecture of the backward network, but aren't trainable.
        n_trainable_layers = sum(map(is_trainable, self.backward_net))

        if is_trainable(self.backward_net[-1]):
            # Don't count the last layer of the backward net (i.e. G_0), since we don't
            # train it.
            n_trainable_layers -= 1

        trainable_layer_iterations = get_list_of_values(
            self.hp.feedback_training_iterations, out_length=n_trainable_layers
        ).copy()

        offset = 0
        iterations_per_layer: List[int] = []
        for layer in self.backward_net:
            # Iterating from GN ... G0,
            if is_trainable(layer) and trainable_layer_iterations:
                iterations_per_layer.append(trainable_layer_iterations.pop(0))
            else:
                iterations_per_layer.append(0)

        assert iterations_per_layer[-1] == 0
        return iterations_per_layer

    def _get_noise_scale_per_layer(self) -> List[float]:
        """ Returns the noise scale for each feedback layer.

        NOTE: Returns it in the same order as the backward_net, i.e. [GN ... G0]
        """
        # TODO: Make the number of iterations align directly with the trainable layers.
        # This is required because there may be some layers (e.g. Reshape), which are
        # present in the architecture of the backward network, but aren't trainable.
        n_trainable_layers = sum(map(is_trainable, self.backward_net))
        trainable_layer_noise_scales = get_list_of_values(
            self.hp.noise, out_length=n_trainable_layers
        ).copy()
        offset = 0
        noise_scale_per_layer: List[int] = []
        for layer in self.backward_net:
            if is_trainable(layer):
                noise_scale_per_layer.append(trainable_layer_noise_scales.pop(0))
            else:
                noise_scale_per_layer.append(0.0)
        return noise_scale_per_layer


def is_trainable(layer: nn.Module) -> bool:
    return any(p.requires_grad for p in layer.parameters())


from typing import TypeVar

V = TypeVar("V")


def get_list_of_values(
    values: Union[V, List[V]], out_length: int, name: str = ""
) -> List[V]:
    """Gets a list of values of length `out_length` from `values`. 
    
    If `values` is a single value, it gets repeated `out_length` times to form the
    output list. 
    If `values` is a list:
        - if it has the right length, it is returned unchanged;
        - if it is too short, the last value is repeated to get the right length;
        - if it is too long, a warning is raised and the extra values are dropped.

    Parameters
    ----------
    values : Union[V, List[V]]
        value or list of values.
    out_length : int
        desired output length.
    name : str, optional
        Name to use in the warning, empty by default.

    Returns
    -------
    List[V]
        List of values of length `out_length`
    """
    out: List[float]
    if isinstance(values, list):
        n_passed_values = len(values)
        if n_passed_values == out_length:
            out = values
        elif n_passed_values < out_length:
            # Repeat the last value.
            out = values + [values[-1]] * (out_length - n_passed_values)
        elif n_passed_values > out_length:
            extra_values = values[out_length:]
            warnings.warn(
                UserWarning(
                    f"{n_passed_values} {name} values passed, but expected "
                    f"{out_length}! Dropping extra values: {extra_values}"
                )
            )
            out = values[:out_length]
    else:
        out = [values] * out_length
    return out
