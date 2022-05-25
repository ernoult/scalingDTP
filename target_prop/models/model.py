from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, TypedDict

import torch
from matplotlib.pyplot import step
from numpy import require
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import Callback, LightningModule, Trainer
from simple_parsing.helpers.serialization.serializable import Serializable
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import Accuracy

from target_prop.networks.network import Network
from target_prop.scheduler_config import LRSchedulerConfig

if typing.TYPE_CHECKING:
    from target_prop.config import Config

PhaseStr = Literal["train", "val", "test"]


class RequiredOutputs(TypedDict):
    """The dictionary format that is expected to be returned from `training/val/test_step`."""

    logits: Tensor
    """The un-normalized logits."""

    y: Tensor
    """ The class labels. """


class StepOutputDict(RequiredOutputs, total=False):
    """The dictionary format that is expected to be returned from `training/val/test_step`."""

    loss: Tensor | float
    """ Optional loss tensor that can be returned by those methods."""


class Model(LightningModule, ABC):
    """Base class for all the models (a.k.a. learning algorithms) of the repo.

    The networks themselves are created separately.
    """

    @dataclass
    class HParams(Serializable):
        lr_scheduler: Optional[LRSchedulerConfig] = None
        """ Configuration for the learning rate scheduler. """

        batch_size: int = 128
        """ batch size """

    hp: Model.HParams
    net_hp: Network.HParams

    def __init__(
        self,
        datamodule: VisionDataModule,
        network: Network,
        hparams: Model.HParams | None = None,
        config: Config | None = None,
    ):
        super().__init__()
        self.datamodule = datamodule
        # IDEA: Could actually implement a `self.HParams` instance method that would choose the
        # default value contextually, based on the choice of datamodule! However Hydra already
        # kinda does that for us already.
        self.hp = hparams or self.HParams()
        self.net_hp = network.hparams
        self.config = config or Config()

        assert isinstance(network, nn.Module)
        self.forward_net = network.to(self.config.device)
        # NOTE: Setting this property allows PL to infer the shapes and number of params.
        self.example_input_array = torch.rand(  # type: ignore
            [datamodule.batch_size, *datamodule.dims], device=self.config.device
        )

        # IDEA: Could use a dict of metrics from torchmetrics instead of just accuracy:
        # self.supervised_metrics: dist[str, Metrics]
        self.accuracy = Accuracy()
        self.top5_accuracy = Accuracy(top_k=5)
        self.trainer: Trainer  # type: ignore

        _ = self.forward_net(self.example_input_array)
        print(f"Forward net: ")
        print(self.forward_net)

        self.save_hyperparameters(
            {
                # TODO: Update these keys, which will also change the runs on wandb.
                "hp": self.hp.to_dict(),
                "config": self.config.to_dict(),
                "model_type": type(self).__name__,
                "net_hp": self.net_hp.to_dict(),
                "net_type": type(self.forward_net).__name__,
            }
        )

    def predict(self, x: Tensor) -> Tensor:
        """Predict the classification labels."""
        return self.forward_net(x).argmax(-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_net(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> StepOutputDict:
        """Performs a training step."""
        return self.shared_step(batch=batch, batch_idx=batch_idx, phase="train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> StepOutputDict:
        """Performs a validation step."""
        return self.shared_step(batch=batch, batch_idx=batch_idx, phase="val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> StepOutputDict:
        """Performs a test step."""
        return self.shared_step(batch=batch, batch_idx=batch_idx, phase="test")

    @abstractmethod
    def shared_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, phase: PhaseStr
    ) -> StepOutputDict:
        """Performs a training/validation/test step.

        This must return a dictionary with at least the 'y' and 'logits' keys, and an optional
        `loss` entry. This is so that the training of the model is easier to parallelize the
        training accross GPUs:
        - the cross entropy loss gets calculated using the global batch size
        - the main metrics are logged inside `training_step_end` (supposed to be better for DP/DDP)
        """

    def training_step_end(self, step_output: StepOutputDict) -> StepOutputDict:
        return self.shared_step_end(step_output, phase="train")

    def validation_step_end(self, step_output: StepOutputDict) -> StepOutputDict:
        return self.shared_step_end(step_output, phase="val")

    def test_step_end(self, step_output: StepOutputDict) -> StepOutputDict:
        return self.shared_step_end(step_output, phase="test")

    def shared_step_end(self, step_output: StepOutputDict, phase: PhaseStr) -> StepOutputDict:
        required_entries = ("logits", "y")
        if not isinstance(step_output, dict):
            raise RuntimeError(
                f"Expected the {phase} step method to output a dictionary with at least the "
                f"{required_entries} keys, but got an output of type {type(step_output)} instead!"
            )
        if not all(k in step_output for k in required_entries):
            raise RuntimeError(
                f"Expected all the following keys to be in the output of the {phase} step "
                f"method: {required_entries}"
            )
        logits = step_output["logits"]
        y = step_output["y"]
        if "loss" in step_output:
            loss = step_output["loss"].mean()
        else:
            loss = F.cross_entropy(logits, y, reduction="mean")

        probs = torch.softmax(logits, -1)
        self.log(f"{phase}/accuracy", self.accuracy(probs, y), prog_bar=True)
        self.log(f"{phase}/top5_accuracy", self.top5_accuracy(probs, y))
        self.log(f"{phase}/F_loss", loss, prog_bar=phase == "train")

        output = step_output.copy()
        if "loss" in step_output:
            # Replace the loss with its mean if it was there.
            output["loss"] = loss
        return output

    # TODO: Log the metrics here, so they are consistent between all the models.?
    # def training_step_end(self, step_results: Tensor | list[Tensor]) -> Tensor:
    #     """Called with the results of each worker / replica's output.

    #     See the `training_step_end` of pytorch-lightning for more info.
    #     """

    def configure_callbacks(self) -> list[Callback]:
        """Use this to add some callbacks that should always be included with the model."""
        callbacks = []
        if self.hp.use_scheduler and self.trainer and self.trainer.logger:
            from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

            return [LearningRateMonitor()]
        return []
