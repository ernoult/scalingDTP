""" Defines the Config class, which contains the options of the experimental setup.
"""
import os
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, ClassVar, Dict, Optional, Type, Union

import torch

# from pl_bolts.datamodules import ImageNet32DataModule
# from pl_bolts.datamodules.imagenet_datamodule import imagenet32_normalization
from pytorch_lightning import LightningDataModule
from simple_parsing.helpers import choice, flag
from simple_parsing.helpers.serialization import Serializable
from torch import Tensor, nn
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from target_prop.datasets import (
    CIFAR10DataModule,
    ImageNet32DataModule,
    cifar10_3xstd_normalization,
    cifar10_normalization,
    imagenet32_normalization,
)
from target_prop.wandb_utils import LoggedToWandb

logger = get_logger(__name__)
Transform = Callable[[Tensor], Tensor]


@dataclass
class Config(Serializable, LoggedToWandb):
    """Configuration options for the experiment (not hyper-parameters)."""

    _stored_at_key: ClassVar[str] = "config"
    available_datasets: ClassVar[Dict[str, Type[LightningDataModule]]] = {
        "cifar10": CIFAR10DataModule,
        "cifar10_3xstd": CIFAR10DataModule,
        "imagenet32": ImageNet32DataModule,
    }
    normalization_transforms: ClassVar[Dict[str, Callable[[], Transform]]] = {
        "cifar10": cifar10_normalization,
        "cifar10_3xstd": cifar10_3xstd_normalization,
        "imagenet32": imagenet32_normalization,
    }

    # Which dataset to use.
    dataset: str = choice(available_datasets.keys(), default="cifar10")
    # Directory where the dataset is to be downloaded. Uses the "DATA_DIR" environment
    # variable, if present, else a local "data" directory.
    data_dir: Path = Path(os.environ.get("DATA_DIR", "data"))
    # Number of workers to use in the dataloader.
    num_workers: int = int(os.environ.get("SLURM_CPUS_PER_TASK", torch.multiprocessing.cpu_count()))
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

    # Limit the number of train/val/test batches. Useful for quick debugging or unit testing.
    limit_train_batches: Union[int, float] = 1.0
    limit_val_batches: Union[int, float] = 1.0
    limit_test_batches: Union[int, float] = 1.0

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
        # NOTE: We don't pass a seed to the datamodule constructor here, because we assume that the
        # train/val/test split is properly seeded with a fixed value already, and we don't want to
        # contaminate the train/val/test splits during sweeps!
        return datamodule_class(
            data_dir=self.data_dir,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            train_transforms=train_transform,
            val_transforms=train_transform,
            test_transforms=test_transform,
        )


from typing import Any, Callable, TypeVar, Union, overload

from simple_parsing.helpers.serialization import encode, register_decoding_fn
from simple_parsing.helpers.serialization.decoding import _register

T = TypeVar("T")
from typing_extensions import ParamSpec


def register_decode(some_type: Type[T]):
    """Register a decoding function for the type `some_type`."""

    def wrapper(f: Callable[[Any], T]) -> Callable[[Any], T]:
        _register(some_type, f)
        return f

    return wrapper


@encode.register(torch.device)
def _encode_device(v: torch.device) -> str:
    return v.type


@register_decode(torch.device)
def _decode_device(v: str) -> torch.device:
    return torch.device(v)
