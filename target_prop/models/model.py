""" Pytorch Lightning version of the model from `prototype.py` with additional
optimizations.

TODO: Add callbacks that compute the jacobians and log images / stuff to wandb.
"""
# from __future__ import annotations
import textwrap
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from simple_parsing.helpers import choice, list_field
from simple_parsing.helpers.hparams import HyperParameters, log_uniform, uniform
from target_prop._weight_operations import init_symetric_weights
from target_prop.config import Config
from target_prop.layers import (
    MaxPool2d,
    Reshape,
    invert,
)  # Conv2dELU,; ConvTranspose2dELU,
from target_prop.optimizer_config import OptimizerConfig
from target_prop.utils import flag, is_trainable
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
from torchmetrics.classification import Accuracy

T = TypeVar("T")
logger = getLogger(__name__)


class BaseModel(LightningModule, ABC):
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
        # NOTE: f_optim.use_lr_scheduler is equivalent to the previous '--scheduler' argument:
        # > Use of a learning rate scheduler for the forward weights.
        f_optim: OptimizerConfig = OptimizerConfig(
            type="sgd", lr=0.08, weight_decay=1e-4,
        )
        # Hyper-parameters for the "backward" optimizer
        b_optim: OptimizerConfig = OptimizerConfig(
            type="sgd", lr=[1e-4, 3.5e-4, 8e-3, 8e-3, 0.18], weight_decay=None,
        )

        # nudging parameter: Used when calculating the first target.
        beta: float = uniform(0.1, 1.0, default=0.7)
        # batch size
        batch_size: int = log_uniform(16, 512, default=128, base=2, discrete=True)

        # Max number of epochs to train for without an improvement to the validation
        # accuracy before the training is stopped.
        early_stopping_patience: int = 3

        # seed: Optional[int] = None  # Random seed to use.

        # Sets symmetric weight initialization
        init_symetric_weights: bool = False

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
        self.hp: BaseModel.HParams = hparams
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
        channels = [self.in_channels] + self.hp.channels
        self.forward_net = nn.Sequential(
            *(
                nn.Sequential(
                    OrderedDict(
                        conv=nn.Conv2d(
                            channels[i],
                            channels[i + 1],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        rho=nn.ELU(),
                        # NOTE: Even though `return_indices` is `False` here, we're actually passing
                        # the indices to the backward net for this layer through a "magic bridge".
                        # We use `return_indices=False` here just so the layer doesn't also return
                        # the indices in its forward pass.
                        pool=MaxPool2d(kernel_size=2, stride=2, return_indices=False),
                        # pool=nn.AvgPool2d(kernel_size=2),
                    )
                )
                for i in range(0, len(channels) - 1)
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

        if self.config.debug:
            print(f"Forward net: ")
            print(self.forward_net)

        assert example_out.requires_grad
        # Get the "pseudo-inverse" of the forward network:
        # TODO: Initializing the weights of the backward net with the transpose of the weights of
        # the forward net, and will check if the gradients are similar.
        self.backward_net: nn.Sequential = invert(self.forward_net)
        if self.hp.init_symetric_weights:
            logger.info(f"Initializing the backward net with symetric weights.")
            init_symetric_weights(self.forward_net, self.backward_net)

        # Expand these values to get one value for each feedback layer to train.
        # NOTE: These values are aligned with the layers of the feedback net.
        self.feedback_noise_scales = self._get_noise_scale_per_feedback_layer(
            forward_ordering=False
        )
        self.feedback_lrs = self._get_learning_rate_per_feedback_layer(
            forward_ordering=False
        )
        self.feedback_iterations = self._get_iterations_per_feedback_layer(
            forward_ordering=False
        )

        if self.config.debug:
            print(f"Backward net: ")
            N = len(self.backward_net)
            for i, (layer, lr, noise, iterations) in list(
                enumerate(
                    zip(
                        self.backward_net,
                        self.feedback_lrs,
                        self.feedback_noise_scales,
                        self.feedback_iterations,
                    )
                )
            ):
                print(
                    f"Layer {i} (G[{N-i}]): LR: {lr}, noise: {noise}, iterations: {iterations}"
                )
                print(textwrap.indent(str(layer), prefix="\t"))
                if i == N - 1:
                    # The last layer of the backward_net (the layer closest to the input) is not
                    # currently being trained, so we expect it to not have these parameters.
                    assert lr == 0
                    assert noise == 0
                    assert iterations == 0
                elif any(p.requires_grad for p in layer.parameters()):
                    # For any of the trainable layers in the backward net (except the last one), we
                    # expect to have positive values:
                    assert lr > 0
                    assert noise > 0
                    assert iterations > 0
                else:
                    # Non-Trainable layers (e.g. Reshape) are not trained.
                    assert lr == 0
                    assert noise == 0
                    assert iterations == 0

        # TODO: Fix the feedback optimizer's learning rates not aligning with the backward_net.

        # Pass the output of the forward net for the `example_input_array` through the
        # backward net, to check that the backward net is indeed able to recover the
        # inputs (at least in terms of their shape for now).
        example_in_hat: Tensor = self.backward_net(example_out)
        assert example_in_hat.requires_grad
        assert example_in_hat.shape == self.example_input_array.shape
        assert example_in_hat.dtype == self.example_input_array.dtype
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
        # self.phase: str = ""

        # Index of the optimizers. Gets influenced by the value of `feedback_before_forward`
        self._feedback_optim_index: int = 0 if self.hp.feedback_before_forward else 1
        self._forward_optim_index: int = 1 if self.hp.feedback_before_forward else 0

        self.trainer: Trainer  # type: ignore

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore
        # Dummy forward pass, not used in practice. We just implement it so that PL can
        # display the input/output shapes of our networks.
        y = self.forward_net(input)
        r = self.backward_net(y)
        return y, r

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
        assert self.phase == phase, (self.phase, phase)
        # self.phase = phase

        dtype: Optional[torch.dtype] = self.dtype if isinstance(
            self.dtype, torch.dtype
        ) else None
        # The total loss to be returned.
        loss: Tensor = torch.zeros(1, device=self.device, dtype=dtype)

        # TODO: Do we want to use the `weight_b_normalize` function? If so, when?
        print(f"optimizer index: {optimizer_idx}")
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
        # raise NotImplementedError
        # FIXME: Sanity check: Use standard backpropagation for training rather than TP.
        logits = self.forward_net.forward_all(x, allow_grads_between_layers=True)[-1]
        accuracy = self.accuracy(torch.softmax(logits, -1), y)
        self.log(f"{self.phase}/accuracy", accuracy, prog_bar=True)
        return F.cross_entropy(logits, y, reduction="mean")
        # return self.criterion(logits, y)

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
        return self.shared_step(
            batch=batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx, phase="train"
        )

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

    def _align_values_with_backward_net(self, values: List[T], default: T) -> List[T]:
        """ Aligns the values in `values` so that they are aligned with the trainable
        layers in the backward net.
        The last layer of the backward net (G_0) is also never trained.

        Inut: values=[1e-4, 3.5e-4, 8e-3, 8e-3, 0.18], default=0
        Output: [0, 1e-4, 3.5e-4, 8e-3, 8e-3, 0, 0.18]

        NOTE: Assumes that the input is given in the *backward* order Gn, Gn-1, ..., G0
        NOTE: Returns the values in the *backward* order (same as the backward_net: [Gn, ..., G0])
        """
        n_layers_that_need_a_value = sum(map(is_trainable, self.backward_net))
        # Don't count the last layer of the backward net (i.e. G_0), since we don't
        # train it.
        n_layers_that_need_a_value -= 1
        if len(values) != n_layers_that_need_a_value:
            raise ValueError(
                f"There are {n_layers_that_need_a_value} layers that need a value, but we were "
                f"given {len(values)} values! (values={values})\n "
                f"NOTE: The order of the values would be reversed before using them with the backward net."
            )

        values_left = values.copy()

        values_per_layer: List[T] = []
        for layer in self.backward_net:
            if is_trainable(layer) and values_left:
                values_per_layer.append(values_left.pop(0))
            else:
                values_per_layer.append(default)
        assert values_per_layer[-1] == default
        return values_per_layer

    def _get_iterations_per_feedback_layer(
        self, forward_ordering: bool = True
    ) -> List[int]:
        """ Returns the number of iterations to perform for each of the layers in
        `self.backward_net`.

        NOTE: Returns it in the same order as the forward_net ([G0, ..., GN] if
        `forward_ordering` is True (default), else returns it as [GN ... G0].
        """
        iterations_per_layer: List[int] = self.hp.feedback_training_iterations
        assert isinstance(iterations_per_layer, list)
        # Reverse it, since it's passed in 'forward' order and the `align_values...`
        # function expects to receive 'reversed' values.
        iterations_per_layer = list(reversed(iterations_per_layer))
        iterations_per_layer = self._align_values_with_backward_net(
            iterations_per_layer, default=0
        )
        if forward_ordering:
            iterations_per_layer.reverse()
        return iterations_per_layer

    def _get_noise_scale_per_feedback_layer(
        self, forward_ordering: bool = True
    ) -> List[float]:
        """ Returns the noise scale for each feedback layer.

        If `forward_ordering` is False, returns it in the same order as the
        backward_net: [GN ... G0], and the opposite if `forward_ordering` is True.  
        """
        noise_scale_per_layer = self.hp.noise
        assert isinstance(noise_scale_per_layer, list)
        # Reverse it, since it's passed in 'forward' order and the `align_values...`
        # function expects to receive 'reversed' values.
        noise_scale_per_layer = list(reversed(noise_scale_per_layer))
        noise_scale_per_layer = self._align_values_with_backward_net(
            noise_scale_per_layer, default=0.0
        )
        if forward_ordering:
            noise_scale_per_layer.reverse()
        return noise_scale_per_layer

    def _get_learning_rate_per_feedback_layer(
        self, forward_ordering: bool = False
    ) -> List[float]:
        """ Returns the learning rate for each feedback layer.

        If `forward_ordering` is False, returns it in the same order as the
        backward_net: [GN ... G0], and the opposite if `forward_ordering` is True.  
        """
        lr_per_layer = self.hp.b_optim.lr
        assert isinstance(lr_per_layer, list), lr_per_layer
        # Reverse it, since it's passed in 'forward' order and the `align_values...`
        # function expects to receive 'reversed' values.
        lr_per_layer = list(reversed(lr_per_layer))
        lr_per_layer = self._align_values_with_backward_net(lr_per_layer, default=0.0)
        if forward_ordering:
            lr_per_layer.reverse()
        return lr_per_layer

    def configure_optimizers(self) -> Tuple[Dict, Dict]:
        """ Creates the optimizers for the forward and feedback net as well as their LR schedules.
        """
        # Create the optimizers using the config class for it in `self.hp`.
        forward_optimizer = self.hp.f_optim.make_optimizer(self.forward_net)
        forward_optim_config = {"optimizer": forward_optimizer}

        # NOTE: We pass the learning rates in the same order as the feedback net:
        lrs_per_feedback_layer = self._get_learning_rate_per_feedback_layer(
            forward_ordering=False
        )
        feedback_optimizer = self.hp.b_optim.make_optimizer(
            self.backward_net, learning_rates_per_layer=lrs_per_feedback_layer
        )
        feedback_optim_config = {"optimizer": feedback_optimizer}

        if self.hp.b_optim.use_lr_scheduler:
            # NOTE: By default we don't use a scheduler for the feedback optimizer.
            feedback_optim_config["lr_scheduler"] = {
                "scheduler": CosineAnnealingLR(
                    feedback_optimizer, T_max=85, eta_min=1e-5
                ),
                "interval": "epoch",  # called after each training epoch
                "frequency": 1,
            }

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

        if self.hp.feedback_before_forward:
            # NOTE: This is the default behaviour (as expected).
            return (
                feedback_optim_config,
                forward_optim_config,
            )
        else:
            return (
                forward_optim_config,
                feedback_optim_config,
            )

    @property
    def forward_optimizer(self) -> Optimizer:
        """Returns The optimizer of the forward net. """
        optimizers = self.optimizers()
        assert isinstance(optimizers, list)
        forward_optimizer = optimizers[self._forward_optim_index]
        assert isinstance(forward_optimizer, Optimizer)
        return forward_optimizer

    @property
    def feedback_optimizer(self) -> Optimizer:
        """Returns The optimizer of the feedback/backward net. """
        optimizers = self.optimizers()
        assert isinstance(optimizers, list)
        feedback_optimizer = optimizers[self._feedback_optim_index]
        assert isinstance(feedback_optimizer, Optimizer)
        return feedback_optimizer

    def configure_callbacks(self) -> List[Callback]:
        callbacks: List[Callback] = []
        if self.hp.early_stopping_patience != 0:
            # If early stopping is enabled, add a PL Callback for it:
            callbacks.append(
                EarlyStopping(
                    "val/accuracy",
                    patience=self.hp.early_stopping_patience,
                    verbose=True,
                )
            )
        return callbacks

    @property
    def phase(self) -> Literal["train", "val", "test", "predict"]:
        """ Returns one of 'train', 'val', 'test', or 'predict', depending on the current phase. 
        
        Used as a prefix when logging values.
        """
        if self.trainer.training:
            return "train"
        if self.trainer.validating:
            return "val"
        if self.trainer.testing:
            return "test"
        if self.trainer.predicting:
            return "predict"
        # NOTE: This doesn't work when inside the sanity check!
        if (
            self.trainer.state.stage
            and self.trainer.state.stage.value == "sanity_check"
        ):
            return "val"
        raise RuntimeError(f"unexpected trainer state: {self.trainer.state}")
