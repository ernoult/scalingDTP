""" Defines the Config class, which contains the options of the experimental setup.
"""
import os
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Any, Callable, ClassVar, Optional, Type
from simple_parsing.helpers import field
from simple_parsing.helpers.serialization.serializable import Serializable

import torch

# from pl_bolts.datamodules import ImageNet32DataModule
# from pl_bolts.datamodules.imagenet_datamodule import imagenet32_normalization
from simple_parsing.helpers import flag
from torch import Tensor

from pl_bolts.datamodules.vision_datamodule import VisionDataModule

logger = get_logger(__name__)
Transform = Callable[[Tensor], Tensor]

from target_prop.datasets.dataset_config import DatasetConfig


@dataclass
class Config(Serializable):
    """Configuration options for the experiment (not hyper-parameters)."""

    # Random seed.
    seed: Optional[int] = 123

    # Debug mode: enables more verbose logging, and turns off logging to wandb.
    # NOTE: Currently also limits the max epochs to 1.
    debug: bool = flag(False)

    # Which device to use.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.seed is None:
            g = torch.Generator(device=self.device)
            self.seed = g.seed()
        array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if array_task_id is not None and self.seed is not None:
            logger.info(
                f"Adding {array_task_id} to base seed ({self.seed}) since this job is "
                f"#{array_task_id} in an array of jobs."
            )
            self.seed += int(array_task_id)
            logger.info(f"New seed: {self.seed}")

