from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import List, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from simple_parsing.helpers import list_field
from simple_parsing.helpers.hparams import log_uniform, uniform
from target_prop.config import Config
from target_prop.optimizer_config import OptimizerConfig
from torch import Tensor, nn

from .vanilla_dtp import VanillaDTP

logger = getLogger(__name__)


class TargetProp(VanillaDTP):
    """ Target Propagation (TP)."""

    @dataclass
    class HParams(VanillaDTP.HParams):
        """ Hyper-Parameters of the model.

        TODO: The parameters for this model haven't been optimized yet. The values below are
        those from DTP-J. 
        """

        # batch size
        batch_size: int = log_uniform(16, 512, default=128, base=2, discrete=True)

        # Max number of training epochs in total.
        max_epochs: int = 90

        # Hyper-parameters for the "backward" optimizer
        b_optim: OptimizerConfig = OptimizerConfig(
            type="sgd", lr=[1e-4, 3.5e-4, 8e-3, 8e-3, 0.18], momentum=0.9
        )
        # The scale of the gaussian random variable in the feedback loss calculation.
        noise: List[float] = uniform(  # type: ignore
            0.001, 0.5, default_factory=[0.4, 0.4, 0.2, 0.2, 0.08].copy, shape=5
        )

        # Hyper-parameters for the forward optimizer
        # NOTE: On mnist, usign 0.1 0.2 0.3 gives decent results (75% @ 1 epoch)
        f_optim: OptimizerConfig = OptimizerConfig(
            type="sgd", lr=0.08, weight_decay=1e-4, momentum=0.9
        )
        # Use of a learning rate scheduler for the forward weights.
        scheduler: bool = True
        # nudging parameter: Used when calculating the first target.
        beta: float = uniform(0.01, 1.0, default=0.7)

        # Number of training steps for the feedback weights per batch. Can be a list of
        # integers, where each value represents the number of iterations for that layer.
        feedback_training_iterations: List[int] = list_field(20, 30, 35, 55, 20)

        # Number of noise samples to use to get the feedback loss in a single iteration.
        # NOTE: The loss used for each update is the average of these losses.
        feedback_samples_per_iteration: int = uniform(1, 20, default=1)

        # Max number of epochs to train for without an improvement to the validation
        # accuracy before the training is stopped. When 0, no early stopping is used.
        early_stopping_patience: int = 0

        # Sets symmetric weight initialization. Useful for debugging.
        init_symetric_weights: bool = False

        # TODO: Add a Callback class to compute and plot jacobians, if that's interesting.
        # jacobian: bool = False  # compute jacobians

        # Step interval for creating and logging plots.
        plot_every: int = 10

    def __init__(self, datamodule: VisionDataModule, hparams: "TargetProp.HParams", config: Config):
        super().__init__(datamodule, hparams, config)
        self.hp: TargetProp.HParams

    def compute_target(self, i: int, G: nn.Module, hs: List[Tensor], prev_target: Tensor) -> Tensor:
        """Compute the target of the previous forward layer. given ,
        the associated feedback layer, the activations for each layer, and the target of the current
        layer.

        Parameters
        ----------
        i : int
            the index of the forward layer for which we want to compute a target
        G : nn.Module
            the associated feedback layer
        hs : List[Tensor]
            the activations for each layer
        prev_target : Tensor
            The target of the next forward layer.

        Returns
        -------
        Tensor
            The target to use to train the forward layer at index `i`.
        """
        # NOTE: Target propagation:
        return G(prev_target)
        # NOTE: Difference target propagation (both Vanilla and DTP-J):
        # return hs[i - 1] - G(hs[i]) + G(prev_target)
        # Cooler ordering, from the Meuleman's DTP paper:
        # return G(prev_target) + (hs[i - 1] - G(hs[i]))

    def layer_feedback_loss(
        self,
        *,
        feedback_layer: nn.Module,
        forward_layer: nn.Module,
        input: Tensor,
        output: Tensor,
        noise_scale: Union[float, Tensor],
        noise_samples: int = 1,
    ) -> Tensor:
        # NOTE: The feedback loss in Target Propagation is the same as in (Vanilla)
        # Difference Target Propagation (as far as I can tell.)
        # TODO: Confirm this with @ernoult.
        return super().layer_feedback_loss(
            feedback_layer=feedback_layer,
            forward_layer=forward_layer,
            input=input,
            output=output,
            noise_scale=noise_scale,
            noise_samples=noise_samples,
        )
