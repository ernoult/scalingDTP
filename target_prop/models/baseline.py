""" Pytorch Lightning image classifier. Uses regular backprop.
"""
# from __future__ import annotations
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List, Tuple, Type, TypeVar

import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from simple_parsing.helpers import choice, list_field
from simple_parsing.helpers.hparams import HyperParameters, log_uniform
from target_prop.config import Config
from target_prop.layers import MaxPool2d, Reshape
from target_prop.optimizer_config import OptimizerConfig
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
from torchmetrics.classification import Accuracy

T = TypeVar("T")
logger = getLogger(__name__)


class BaselineModel(LightningModule, ABC):
    """Baseline model that uses normal backpropagation."""

    @dataclass
    class HParams(HyperParameters):
        """Hyper-Parameters of the baseline model."""

        # Max number of training epochs in total.
        max_epochs: int = 90

        # Hyper-parameters for the forward optimizer
        f_optim: OptimizerConfig = OptimizerConfig(type="adam", lr=3e-4)
        # Use of a learning rate scheduler.
        scheduler: bool = False

        # batch size
        batch_size: int = log_uniform(16, 512, default=128, base=2, discrete=True)

        # Max number of epochs to train for without an improvement to the validation
        # accuracy before the training is stopped.
        early_stopping_patience: int = 5

    def __init__(
        self,
        datamodule: VisionDataModule,
        network: nn.Sequential,
        hparams: HParams,
        config: Config,
        network_hparams: HyperParameters,
    ):
        super().__init__()
        # NOTE: Can't exactly set the `hparams` attribute because it's a special property of PL.
        self.hp: BaselineModel.HParams = hparams
        self.net_hp = network_hparams
        self.config = config
        if self.config.seed is not None:
            seed_everything(seed=self.config.seed, workers=True)

        # NOTE: Setting this property allows PL to infer the shapes and number of params.
        self.example_input_array = torch.rand(  # type: ignore
            [datamodule.batch_size, *datamodule.dims],
            device=self.device,
            # names=["B", "C", "H", "W"],  # NOTE: cudnn conv doesn't yet support named inputs.
        )

        # Create the forward achitecture
        self.forward_net = network

        if self.config.debug:
            _ = self.forward(self.example_input_array)
            print(f"Forward net: ")
            print(self.forward_net)

        # Metrics:
        self.accuracy = Accuracy()

        self.save_hyperparameters(
            {
                "hp": self.hp.to_dict(),
                "config": self.config.to_dict(),
                "model_type": type(self).__name__,
                "net_hp": self.net_hp.to_dict(),
                "net_type": type(self.hp).__name__,
            }
        )
        self.trainer: Trainer  # type: ignore

        # Dummy forward pass to initialize the weights of the lazy modules (required for DP/DDP)
        _ = self(self.example_input_array)

    def create_trainer(self) -> Trainer:
        # IDEA: Would perhaps be useful to add command-line arguments for DP/DDP/etc.
        return Trainer(
            max_epochs=self.hp.max_epochs,
            gpus=1,
            track_grad_norm=False,
            accelerator=None,
            # NOTE: Not sure why but seems like they are still reloading them after each epoch!
            reload_dataloaders_every_epoch=False,
            terminate_on_nan=self.automatic_optimization,
            logger=WandbLogger() if not self.config.debug else None,
            limit_train_batches=self.config.limit_train_batches,
            limit_val_batches=self.config.limit_val_batches,
            limit_test_batches=self.config.limit_test_batches,
        )

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        # Dummy forward pass, not used in practice. We just implement it so that PL can
        # display the input/output shapes of our networks.
        logits = self.forward_net(input)
        return logits

    def shared_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        phase: str,
    ) -> Tensor:
        """Main step, used by the `[training/valid/test]_step` methods."""
        x, y = batch
        # Setting this value just so we don't have to pass `phase=...` to `forward_loss`
        # and `feedback_loss` below.
        logits = self.forward_net(x)

        loss = F.cross_entropy(logits, y, reduction="mean")

        probs = torch.softmax(logits, -1)
        accuracy = self.accuracy(probs, y)
        self.log(f"{phase}/accuracy", accuracy, prog_bar=True)
        self.log(f"{phase}/F_loss", loss, prog_bar=phase == "train")
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        return self.shared_step(batch=batch, batch_idx=batch_idx, phase="train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        return self.shared_step(batch=batch, batch_idx=batch_idx, phase="val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        return self.shared_step(batch, batch_idx=batch_idx, phase="test")

    def configure_optimizers(self) -> Dict:
        """Creates the optimizers and the LR scheduler (if needed)."""
        # Create the optimizers using the config class for it in `self.hp`.
        optimizer = self.hp.f_optim.make_optimizer(self.forward_net)
        optim_config: Dict[str, Any] = {"optimizer": optimizer}

        if self.hp.scheduler:
            # `main.py` seems to be using a weight scheduler only for the forward weight
            # training.
            optim_config["lr_scheduler"] = {
                "scheduler": CosineAnnealingLR(optimizer, T_max=85, eta_min=1e-5),
                "interval": "epoch",  # called after each training epoch
                "frequency": 1,
            }
        return optim_config

    def configure_callbacks(self) -> List[Callback]:
        callbacks: List[Callback] = []
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
