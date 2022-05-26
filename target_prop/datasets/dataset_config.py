import os
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, ClassVar, Dict, Type, TypeVar

import torch
from pl_bolts.datamodules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
    ImagenetDataModule,
    MNISTDataModule,
)
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from simple_parsing import choice
from simple_parsing.helpers.serialization.serializable import Serializable
from torch import Tensor
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from target_prop.datasets import ImageNet32DataModule as ImageNet32NoValDataModule

FILE = Path(__file__)
REPO_ROOTDIR = FILE.parent.parent  # The root of the repo.
Transform = Callable[[Tensor], Tensor]
D = TypeVar("D", bound=VisionDataModule)

logger = get_logger(__name__)


def get_datamodule(dataset: str, batch_size: int, **kwargs) -> VisionDataModule:
    return DatasetConfig(dataset=dataset, **kwargs).make_datamodule(batch_size=batch_size)


import functools


@dataclass
class DatasetConfig(Serializable):
    available_datasets: ClassVar[Dict[str, Type[VisionDataModule]]] = {  # type: ignore
        "mnist": MNISTDataModule,
        # MNIST_noval: # TODO: Add this when we add Sean's mnist datamodule.
        "cifar10": CIFAR10DataModule,
        "cifar10_noval": functools.partial(CIFAR10DataModule, val_split=0.0),
        "imagenet32_noval": ImageNet32NoValDataModule,
        "imagenet": ImagenetDataModule,
        "fmnist": FashionMNISTDataModule,
    }
    # Which dataset to use.
    dataset: str = choice(*available_datasets.keys(), default="cifar10")
    # Directory to look for the datasets.
    data_dir: str = os.environ.get("SLURM_TMPDIR", str(REPO_ROOTDIR / "data"))
    # Number of workers to use in the dataloader.
    num_workers: int = int(os.environ.get("SLURM_CPUS_PER_TASK", torch.multiprocessing.cpu_count()))
    # Wether to pin the memory, which is good when using CUDA tensors.
    # pin_memory: bool = True
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

    # When set to `True`, uses the much larger standard deviation in the normalization transform
    # to match the legacy implementation (~3x std)
    use_legacy_std: bool = False

    def make_datamodule(self, batch_size: int) -> VisionDataModule:
        datamodule_class: Type[VisionDataModule] = self.available_datasets[self.dataset]
        datamodule = datamodule_class(
            data_dir=self.data_dir,
            batch_size=batch_size,
            num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            normalize=self.normalize,
            val_split=0.0 if self.dataset.endswith("_noval") else self.val_split,
        )
        # NOTE: The standard transforms includes ToTensor and normalization.
        # We are adding the RandomFlip and the RandomCrop.
        # assert datamodule.train_transforms is None
        # assert datamodule.val_transforms is None
        # assert datamodule.test_transforms is None

        default_transforms = datamodule.default_transforms()
        assert isinstance(default_transforms, Compose)
        if self.normalize:
            assert len(default_transforms.transforms) == 2
            assert isinstance(default_transforms.transforms[0], ToTensor)
            assert isinstance(default_transforms.transforms[1], Normalize)
            if self.use_legacy_std:
                if not self.dataset.startswith("cifar10"):
                    raise NotImplementedError(
                        f"What's the 'legacy std' to use for dataset {self.dataset}?"
                    )
                norm: Normalize = default_transforms.transforms[1]
                new_norm = Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.6069, 0.5982, 0.603),
                    inplace=norm.inplace,
                )
                default_transforms.transforms[1] = new_norm
        else:

            assert len(default_transforms.transforms) == 1
            assert isinstance(default_transforms.transforms[0], ToTensor)

        train_transforms = Compose(
            [
                RandomHorizontalFlip(0.5),
                RandomCrop(
                    size=self.image_crop_size,
                    padding=self.image_crop_pad,
                    padding_mode="edge",
                ),
                *default_transforms.transforms,
            ]
        )

        datamodule._train_transforms = train_transforms
        datamodule._val_transforms = default_transforms
        datamodule._test_transforms = default_transforms

        # NOTE: Taking these directly from the main.py for CIFAR-10. These might not be the
        # right kind of transforms to use for ImageNet.
        # NOTE: We don't pass a seed to the datamodule constructor here, because we assume that the
        # train/val/test split is properly seeded with a fixed value already, and we don't want to
        # contaminate the train/val/test splits during sweeps!
        return datamodule
