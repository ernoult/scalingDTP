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
from target_prop.utils import get_list_of_values, is_trainable
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

from target_prop.layers import get_backward_equivalent
from target_prop.optimizer_config import OptimizerConfig as OptimizerHParams
from target_prop.config import Config
from target_prop.layers import (
    AdaptiveAvgPool2d,
    # Conv2dELU,
    # ConvTranspose2dELU,
    Reshape,
    Sequential,
)
from target_prop.feedback_loss import feedback_loss
from target_prop.utils import flag
from logging import getLogger
logger = getLogger(__file__)



class Model(LightningModule, ABC):
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
        feedback_before_forward: bool = flag(True)

        activation: Type[nn.Module] = choice(
            {"relu": nn.ReLU, "elu": nn.ELU,}, default="elu"
        )

        # feedback_weight_training_procedure: str = choice(
        #     "awesome", "sequential", "best_of_both_worlds", default="sequential",
        # )
        # """
        # What kind of precedure to use to train the feedback weights.
        # - 'awesome':
        #     Fully parallel training with multiple noise samples and only one iteration.
        #     As far as I know, this is the only procedure that can easily be scaled to
        #     multi-GPU with DP/DDP through Pytorch-Lightning.
        # - 'sequential':
        #     Fully sequential: multiple noise samples per iteration, multiple iterations,
        #     possibly different number of iterations per layer. This is the same
        #     configuration as in @ernoult's implementation. Very slow to run.
        # - 'best_of_both_worlds':
        #     Fully parallel, with possibly multiple noise iterations per layer. This has
        #     some of the benefits of `awesome` (layer-wise parallel training), while also
        #     preserving the ability to set a different number of updates per layer.
        # """

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
        # Get the "pseudo-inverse" of the forward network:
        self.backward_net = self.forward_net.invert()

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

        # Index of the optimizers. Gets influenced by the value of `feedback_before_forward`
        self._feedback_optim_index: int = 0 if self.hp.feedback_before_forward else 1
        self._forward_optim_index: int = 1 if self.hp.feedback_before_forward else 0

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        y = self.forward_net(input)
        return y
        # r = self.backward_net(y)
        # return y, r

    
    def shared_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        phase: str,
        optimizer_idx: Optional[int] = None,
    ):
        """ Main step, used by the `[training/valid/test]_step` methods.
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
            self.log(f"{phase}/f_loss", feedback_loss, prog_bar=True, **self.log_kwargs)

        if optimizer_idx in [None, self._forward_optim_index]:
            # ----------- Optimize the forward weights -------------
            forward_loss = self.forward_loss(x, y)
            self.log(f"{phase}/b_loss", forward_loss, prog_bar=True, **self.log_kwargs)
            loss += forward_loss

        return loss

    @abstractmethod
    def forward_loss(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def feedback_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """ Feedback weight training
        
        WIP: Can switch between different ways of training the feedback weights using
        the `sequential_feedback_training` and `feedbacK_training_iterations`
        hyper-parameters.
        
        NOTE: Only the 'full_parallel' version will work with DDP.
        """
        raise NotImplementedError
    
    def training_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int = None
    ) -> Union[Tensor, float]:
        loss = self.shared_step(
            batch=batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx, phase="train"
        )
        if self.automatic_optimization:
            # Should have a loss with gradients if we're using automatic optimization
            # from PL.
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
        """ Called with the results of each worker / replica's output.

        See the `training_step_end` of pytorch-lightning for more info.
        """
        # TODO: For now we're kinda losing the logs and stuff that happens within the
        # workers in DP (they won't show up in the progress bar for instance).
        # merged_step_results = {
        #     k: sum(v_i.to(self.device) for v_i in v)
        #     for k, v in step_results
        # }
        merged_step_result = (
            step_results if isinstance(step_results, (Tensor, float)) else sum(step_results)
        )

        # TODO: If NOT in automatic differentiation, but still in a scenario where we
        # can do a single update, do it here.
        loss = merged_step_result
        self.log(f"{self.phase}/total loss", loss, on_step=True, prog_bar=True)

        if not self.automatic_optimization and isinstance(loss, Tensor) and loss.requires_grad:
            forward_optimizer = self.forward_optimizer
            backward_optimizer = self.feedback_optimizer
            forward_optimizer.zero_grad()
            backward_optimizer.zero_grad()

            self.manual_backward(loss)

            forward_optimizer.step()
            backward_optimizer.step()
            return float(loss)

        elif not self.automatic_optimization:
            return float(merged_step_result)

        assert self.automatic_optimization
        return merged_step_result

    def validation_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Union[Tensor, float]:
        return self.shared_step(
            batch=batch, batch_idx=batch_idx, optimizer_idx=None, phase="val"
        )

    def test_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Union[Tensor, float]:
        return self.shared_step(batch, batch_idx=batch_idx, phase="test")

    def get_noise_scale_per_layer(self, network: nn.Sequential) -> List[float]:
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
        feedback_optimizer = self.hp.b_optim.make_optimizer(self.backward_net)
        forward_optimizer = self.hp.f_optim.make_optimizer(self.forward_net)

        feedback_optim_config = {"optimizer": feedback_optimizer}
        forward_optim_config = {"optimizer": forward_optimizer}

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
            feedback_optim_config["lr_scheduler"] = {
                "scheduler": CosineAnnealingLR(
                    feedback_optimizer, T_max=85, eta_min=1e-5
                ),
                "interval": "epoch",  # called after each training epoch
                "frequency": 1,
            }

        if self.hp.feedback_before_forward:
            return [
                feedback_optim_config,
                forward_optim_config,
            ]
        else:
            return [
                forward_optim_config,
                feedback_optim_config,
            ]
        # NOTE: When using the 'frequency' version below, it trains once every `n`
        # batches, rather than `n` times on the same batch.
        # return (
        #     {"optimizer": forward_optimizer, "frequency": 1},
        #     {"optimizer": backward_optimizer, "frequency": 1},
        # )

    @property
    def forward_optimizer(self) -> Optimizer:
        optimizers = self.optimizers()
        assert isinstance(optimizers, list)
        forward_optimizer = optimizers[self._forward_optim_index]
        assert isinstance(forward_optimizer, Optimizer)
        return forward_optimizer

    @property
    def feedback_optimizer(self) -> Optimizer:
        optimizers = self.optimizers()
        assert isinstance(optimizers, list)
        feedback_optimizer = optimizers[self._feedback_optim_index]
        assert isinstance(feedback_optimizer, Optimizer)
        return feedback_optimizer

    def configure_callbacks(self) -> List[Callback]:
        callbacks: List[Callback] = []
        if self.hp.early_stopping_patience != 0:
            # If early stopping is enabled, add a PL Callback for it:
            callbacks.append(EarlyStopping("val/accuracy", patience=self.hp.early_stopping_patience, verbose=True))
        return callbacks

    def _get_noise_scale_per_layer(self) -> List[float]:
        """ Returns the noise scale for each feedback layer.

        NOTE: Returns it in the same order as the backward_net, i.e. [GN ... G0]
        """
        # TODO: Make the number of iterations align directly with the trainable layers.
        # This is required because there may be some layers (e.g. Reshape), which are
        # present in the architecture of the backward network, but aren't trainable.
        n_trainable_layers = sum(map(is_trainable, self.backward_net))
        trainable_layer_noise_scales: List[float] = get_list_of_values(
            self.hp.noise, out_length=n_trainable_layers
        ).copy()
        offset = 0
        noise_scale_per_layer: List[float] = []
        for layer in self.backward_net:
            if is_trainable(layer):
                noise_scale_per_layer.append(trainable_layer_noise_scales.pop(0))
            else:
                noise_scale_per_layer.append(0.0)
        return noise_scale_per_layer
