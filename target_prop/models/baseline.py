"""Pytorch Lightning image classifier.

Uses regular backprop.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from logging import getLogger
from typing import Any

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning.callbacks import Callback, EarlyStopping
from simple_parsing import field
from torch import Tensor
from torch.nn import functional as F

from target_prop.config import MiscConfig
from target_prop.config.optimizer_config import OptimizerConfig
from target_prop.config.scheduler_config import (
    CosineAnnealingLRConfig,
    LRSchedulerConfig,
)
from target_prop.models.model import Model, PhaseStr, StepOutputDict
from target_prop.networks import Network

logger = getLogger(__name__)


class BaselineModel(Model):
    """Baseline model that uses normal backpropagation."""

    @dataclass
    class HParams(Model.HParams):
        """Hyper-Parameters of the baseline model."""

        # Arguments to be passed to the LR scheduler.
        lr_scheduler: LRSchedulerConfig = field(default_factory=CosineAnnealingLRConfig)
        # Use of a learning rate scheduler.
        use_scheduler: bool = True

        # Max number of training epochs in total.
        max_epochs: int = 90

        # Hyper-parameters for the forward optimizer
        f_optim: OptimizerConfig = field(
            default_factory=functools.partial(OptimizerConfig, type="sgd", lr=[0.05])
        )

        # batch size
        batch_size: int = 128

        # Max number of epochs to train for without an improvement to the validation
        # accuracy before the training is stopped.
        early_stopping_patience: int = 0

    def __init__(
        self,
        datamodule: VisionDataModule,
        network: Network,
        hparams: HParams | None = None,
        config: MiscConfig | None = None,
    ):
        super().__init__(datamodule=datamodule, network=network, hparams=hparams, config=config)
        self.hp: BaselineModel.HParams
        self.automatic_optimization = True

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        # Dummy forward pass, not used in practice. We just implement it so that PL can
        # display the input/output shapes of our networks.
        logits = self.forward_net(input)
        return logits

    def shared_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        phase: PhaseStr,
    ) -> StepOutputDict:
        x, y = batch
        logits = self.forward_net(x)
        loss = F.cross_entropy(logits, y, reduction="none")
        return {"logits": logits, "y": y, "loss": loss}

    def configure_optimizers(self) -> dict:
        """Creates the optimizers and the LR scheduler (if needed)."""
        # Create the optimizers using the config class for it in `self.hp`.
        optimizer = self.hp.f_optim.make_optimizer(self.forward_net)
        optim_config: dict[str, Any] = {"optimizer": optimizer}

        if self.hp.use_scheduler:
            lr_scheduler = self.hp.lr_scheduler.make_scheduler(optimizer)
            scheduler_config: dict[str, Any] = {
                "scheduler": lr_scheduler,
                "interval": self.hp.lr_scheduler.interval,
                "frequency": self.hp.lr_scheduler.frequency,
            }
            optim_config["lr_scheduler"] = scheduler_config
        return optim_config

    def configure_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = super().configure_callbacks()
        if self.hp.early_stopping_patience != 0:
            # If early stopping is enabled, add a PL Callback for it:
            callbacks.append(
                EarlyStopping(
                    "val/accuracy",
                    mode="max",
                    patience=self.hp.early_stopping_patience,
                    verbose=True,
                )
            )
        return callbacks
