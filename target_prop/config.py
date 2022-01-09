""" Defines the Config class, which contains the options of the experimental setup.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Dict, Optional, Type

import torch
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.datamodules.imagenet_datamodule import imagenet_normalization
from pytorch_lightning import LightningDataModule
from simple_parsing.helpers import choice, flag
from simple_parsing.helpers.serialization import Serializable
from torch import Tensor
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from target_prop.datasets import CIFAR10DataModule, cifar10_normalization

Transform = Callable[[Tensor], Tensor]


@dataclass
class Config(Serializable):
    """Configuration options for the experiment (not hyper-parameters)."""

    available_datasets: ClassVar[Dict[str, Type[LightningDataModule]]] = {
        "cifar10": CIFAR10DataModule,
        "imagenet": ImagenetDataModule,  # TODO: Not yet tested.
    }
    normalization_transforms: ClassVar[Dict[str, Callable[[], Transform]]] = {
        "cifar10": cifar10_normalization,
        "imagenet": imagenet_normalization,  # TODO: Not yet tested.
    }

    # Which dataset to use.
    dataset: str = choice(available_datasets.keys(), default="cifar10")
    # Directory where the dataset is to be downloaded. Uses the "DATA_DIR" environment
    # variable, if present, else a local "data" directory.
    data_dir: Path = Path(os.environ.get("DATA_DIR", "data"))
    # Number of workers to use in the dataloader.
    num_workers: int = torch.multiprocessing.cpu_count()
    # Wether to pin the memory, which is good when using CUDA tensors.
    pin_memory: bool = True
    # Random seed.
    seed: Optional[int] = 123
    # Wether to shuffle the dataset or not.
    shuffle: bool = True

    # Debug mode: enables more verbose logging, and turns off logging to wandb.
    # NOTE: Currently also limits the max epochs to 1.
    debug: bool = flag(False)

    # Size of the random crop for training.
    # TODO: Might have to use a different value for imagenet.
    image_crop_size: int = 32

    # Which device to use.
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        if self.seed is None:
            g = torch.Generator(device=self.device)
            self.seed = g.seed()

    def make_datamodule(self, batch_size: int) -> LightningDataModule:

        datamodule_class = self.available_datasets[self.dataset]
        normalization_transform = self.normalization_transforms.get(self.dataset)
        train_transform: Optional[Callable] = None
        test_transform: Optional[Callable] = None
        if normalization_transform is not None:
            # NOTE: Taking these directly from the main.py for CIFAR-10. These might not be the
            # right kind of transforms to use for ImageNet.
            train_transform = Compose(
                [
                    RandomHorizontalFlip(0.5),
                    RandomCrop(size=self.image_crop_size, padding=4, padding_mode="edge"),
                    ToTensor(),
                    normalization_transform(),
                ]
            )

            test_transform = Compose(
                [
                    ToTensor(),
                    normalization_transform(),
                ]
            )
        return datamodule_class(
            data_dir=self.data_dir,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            seed=self.seed or 123,  # NOTE: Seed here needs to be an int, not None!,
            shuffle=self.shuffle,
            train_transforms=train_transform,
            val_transforms=train_transform,
            test_transforms=test_transform,
        )
