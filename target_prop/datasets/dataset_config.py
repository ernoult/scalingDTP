import enum
import os
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Callable, ClassVar, Dict, Generic, Type, TypeVar
from simple_parsing import choice

import torch
from pl_bolts.datamodules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
    ImagenetDataModule,
    MNISTDataModule,
)
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from torch import Tensor
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

from target_prop.datasets import CIFAR10DataModule as CIFAR10NoValDataModule
from target_prop.datasets import ImageNet32DataModule as ImageNet32NoValDataModule

Transform = Callable[[Tensor], Tensor]

logger = get_logger(__name__)

D = TypeVar("D", bound=VisionDataModule)


class DatasetTypes(enum.Enum):
    """ Enum of the types of datamodules available. """


from abc import ABC
from typing import Optional, Union, Any


class NoValDataModule(VisionDataModule, ABC):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.0,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(
            data_dir,
            val_split,
            num_workers,
            normalize,
            batch_size,
            seed,
            shuffle,
            pin_memory,
            drop_last,
            *args,
            **kwargs
        )
        self.val_split = 0.0


from target_prop.utils.hydra_utils import LoadableFromHydra
from target_prop.utils.wandb_utils import LoggedToWandb


@dataclass
class DatasetConfig(LoadableFromHydra, LoggedToWandb):
    _stored_at_key: ClassVar[Optional[str]] = "dataset"

    available_datasets: ClassVar[Dict[str, Type[VisionDataModule]]] = {  # type: ignore
        "mnist": MNISTDataModule,
        # MNIST_noval: # TODO: Add this when we add Sean's mnist datamodule.
        "cifar10": CIFAR10DataModule,
        # "cifar10_noval": CIFAR10NoValDataModule,
        # "imagenet32_noval": ImageNet32NoValDataModule,
        "imagenet": ImagenetDataModule,
        "fmnist": FashionMNISTDataModule,
    }
    # Which dataset to use.
    dataset: str = choice(*available_datasets.keys(), default="cifar10")
    # Directory to look for the datasets.
    data_dir: str = os.environ.get("SLURM_TMPDIR", "data")
    # Number of workers to use in the dataloader.
    num_workers: int = int(os.environ.get("SLURM_CPUS_PER_TASK", torch.multiprocessing.cpu_count()))
    # Wether to pin the memory, which is good when using CUDA tensors.
    pin_memory: bool = True
    # Wether to shuffle the dataset or not.
    shuffle: bool = True

    # Wether to apply the standard normalization transform of the associated dataset to the images.
    normalize: bool = True

    # Size of the random crop for training.
    image_crop_size: int = 32
    # `pad` argument to the random crop padding.
    image_crop_pad: int = 4

    # Percentage of data to reserve for a validation set.
    val_split: float = 0.1

    def make_datamodule(self, batch_size: int) -> VisionDataModule:
        datamodule_class: Type[VisionDataModule] = self.available_datasets[self.dataset]
        datamodule = datamodule_class(
            data_dir=self.data_dir,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            normalize=self.normalize,
            val_split=self.val_split,
        )
        # NOTE: The standard transforms includes ToTensor and normalization.
        # We are adding the RandomFlip and the RandomCrop.
        assert datamodule.train_transforms is None
        assert datamodule.val_transforms is None
        assert datamodule.test_transforms is None

        default_transforms = datamodule.default_transforms()
        assert isinstance(default_transforms, Compose)
        if self.normalize:
            assert len(default_transforms.transforms) == 2
            assert isinstance(default_transforms.transforms[0], ToTensor)
            assert isinstance(default_transforms.transforms[1], Normalize)
        else:
            assert len(default_transforms.transforms) == 1
            assert isinstance(default_transforms.transforms[0], ToTensor)

        train_transforms = Compose(
            [
                RandomHorizontalFlip(0.5),
                RandomCrop(
                    size=self.image_crop_size, padding=self.image_crop_pad, padding_mode="edge"
                ),
                *default_transforms.transforms,
            ]
        )

        datamodule.train_transforms = train_transforms
        datamodule.val_transforms = default_transforms
        datamodule.test_transforms = default_transforms

        # NOTE: Taking these directly from the main.py for CIFAR-10. These might not be the
        # right kind of transforms to use for ImageNet.
        # NOTE: We don't pass a seed to the datamodule constructor here, because we assume that the
        # train/val/test split is properly seeded with a fixed value already, and we don't want to
        # contaminate the train/val/test splits during sweeps!
        return datamodule

