""" Defines the Config class, which contains the options of the experimental setup.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Optional, Type

from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, MNISTDataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from simple_parsing.helpers import choice
from simple_parsing.helpers.serialization import Serializable
import os


@dataclass
class Config(Serializable):
    """ Configuration options for the experiment (not hyper-parameters). """

    available_datasets: ClassVar[Dict[str, Type[VisionDataModule]]] = {
        "mnist": MNISTDataModule,
        "cifar10": CIFAR10DataModule,
        "imagenet": ImagenetDataModule,  # TODO: Not yet tested.
    }
    # Which dataset to use.
    dataset: str = choice(available_datasets.keys(), default="cifar10")
    
    # Directory where the dataset is to be downloaded. Uses the "DATA_DIR" environment
    # variable, if present, else a local "data" directory.
    data_dir: Path = Path(os.environ.get("DATA_DIR", "data"))
    # Number of workers to use in the dataloader.
    num_workers: int = 16
    # Wether to pin the memory, which is good when using CUDA tensors.
    pin_memory: bool = True
    # Random seed.
    seed: Optional[int] = 123
    # Portion of the dataset to reserve for validation
    val_split: float = 0.2
    # Wether to shuffle the dataset or not.
    shuffle: bool = True

    # Debug mode: enables more verbose logging, and turns off logging to wandb.
    # NOTE: Currently also limits the max epochs to 1. 
    debug: bool = False

    def make_datamodule(self, batch_size: int) -> VisionDataModule:
        datamodule_class = self.available_datasets[self.dataset]
        return datamodule_class(
            data_dir=self.data_dir,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            val_split=self.val_split,
            seed=self.seed,
            shuffle=self.shuffle,
        )
