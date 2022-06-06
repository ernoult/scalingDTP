"""Defines the Config class, which contains the options of the experimental setup."""
import os
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Optional

import torch
from simple_parsing.helpers import flag
from simple_parsing.helpers.serialization.serializable import Serializable

logger = get_logger(__name__)


@dataclass
class MiscConfig(Serializable):
    """Miscelaneous configuration options for the experiment (not hyper-parameters)."""

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
