from __future__ import annotations
from abc import ABC
import functools
import os
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, ClassVar, Dict, Type, TypeVar
from pytorch_lightning import LightningDataModule

import torch
from pl_bolts.datamodules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
    ImagenetDataModule,
    MNISTDataModule,
)
from hydra_zen import MISSING
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

from target_prop.datasets.imagenet32_datamodule import ImageNet32DataModule

from hydra_zen import builds, just, make_config, make_custom_builds_fn, instantiate
from hydra_zen.typing import Partial, PartialBuilds
from hydra.core.config_store import ConfigStore


FILE = Path(__file__)
REPO_ROOTDIR = FILE.parent.parent.parent  # The root of the repo.
DATA_DIR = os.environ.get("SLURM_TMPDIR", str(REPO_ROOTDIR / "data"))
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", torch.multiprocessing.cpu_count()))

logger = get_logger(__name__)


Transform = Callable[[Tensor], Tensor]
D = TypeVar("D", bound=VisionDataModule)

# NOTE:
# datamodule_config_class = builds(CIFAR10DataModule, data_dir=DATA_DIR, ...)
# datamodule_config = datamodule_config_class()
# datamodule = instantiate(datamodule_config) OR datamodule_config()


def get_datamodule(dataset: str, batch_size: int, **kwargs) -> VisionDataModule:
    return DatasetConfig(dataset=dataset, **kwargs).make_datamodule(batch_size=batch_size)


from torchvision import transforms
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from hydra_zen import MISSING

from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pl_bolts.datamodules.cifar10_datamodule import cifar10_normalization
from target_prop.datasets.imagenet32_datamodule import imagenet32_normalization


def validate_datamodule(datamodule: D) -> D:
    """ Checks that the transforms / things are setup correctly. Returns the same datamodule.
    
    TODO: Could be a good occasion to wrap the datamodule with some kind of adapter that moves
    stuff to the right location!
    """
    if not datamodule.normalize:
        remove_normalization_from_transforms(datamodule)
    else:
        # todo: maybe check that the normalization transform is present everywhere?
        pass
    return datamodule


def remove_normalization_from_transforms(datamodule: D) -> None:
    transform_properties = (
        datamodule.train_transforms,
        datamodule.val_transforms,
        datamodule.test_transforms,
    )
    for transform_list in transform_properties:
        if transform_list is None:
            continue
        assert isinstance(transform_list, transforms.Compose)
        if isinstance(transform_list.transforms[-1], transforms.Normalize):
            t = transform_list.transforms.pop(-1)
            logger.info(f"Removed normalization transform {t} since normalize=False")
        if any(isinstance(t, Normalize) for t in transform_list.transforms):
            raise RuntimeError(
                f"Unable to remove all the normalization transforms from datamodule {datamodule}: "
                f"{transform_list}"
            )


TrainTransformsConfig = builds(
    transforms.Compose,
    transforms=[
        builds(transforms.RandomHorizontalFlip, p=0.5),
        builds(transforms.RandomCrop, size=32, padding=4, padding_mode="edge"),
        builds(transforms.ToTensor),
    ],
)


@dataclass
class CallableConfig:
    """ Little mixin that makes it possible to call the config, like adam_config() -> Adam,
    instead of having to use `instantiate`.
    """

    def __call__(self, *args, **kwargs):
        return instantiate(self, *args, **kwargs)


DatasetConfig = builds(
    LightningDataModule,
    builds_bases=(Serializable, CallableConfig,),
    dataclass_name="DatasetConfig",
)

# TODO: Could maybe register with different defaults based on which environment/cluster we're in?
# note: Could also interpolate the data_dir or other params from hydra like "${data_dir}"
VisionDatasetConfig = builds(
    VisionDataModule,
    data_dir=DATA_DIR,
    num_workers=NUM_WORKERS,
    val_split=0.1,
    builds_bases=(DatasetConfig,),
    populate_full_signature=True,
    dataclass_name="VisionDataModuleConfig",
)


def mnist_normalization():
    return transforms.Normalize(mean=0.5, std=0.5)


mnist_config = builds(
    MNISTDataModule,
    train_transforms=TrainTransformsConfig(
        transforms=[
            builds(transforms.RandomHorizontalFlip, p=0.5),
            builds(transforms.RandomCrop, size=28, padding=4, padding_mode="edge"),
            builds(transforms.ToTensor),
            builds(mnist_normalization),
        ],
    ),
    builds_bases=(VisionDatasetConfig,),
)

fmnist_config = builds(FashionMNISTDataModule, builds_bases=(mnist_config,),)

cifar10_config = builds(
    CIFAR10DataModule,
    train_transforms=TrainTransformsConfig(
        transforms=[
            builds(transforms.RandomHorizontalFlip, p=0.5),
            builds(transforms.RandomCrop, size=32, padding=4, padding_mode="edge"),
            builds(transforms.ToTensor),
            builds(cifar10_normalization),
        ],
    ),
    builds_bases=(VisionDatasetConfig,),
)

imagenet32_config = builds(
    ImageNet32DataModule,
    val_split=-1,
    num_images_per_val_class=50,  # Slightly different.
    normalize=True,
    train_transforms=TrainTransformsConfig(
        transforms=[
            builds(transforms.RandomHorizontalFlip, p=0.5),
            builds(transforms.RandomCrop, size=32, padding=4, padding_mode="edge"),
            builds(transforms.ToTensor),
            builds(imagenet32_normalization),
        ]
    ),
    builds_bases=(VisionDatasetConfig,),
    # populate_full_signature=True,
)


cs = ConfigStore.instance()
cs.store(group="dataset", name="base", node=DatasetConfig)
cs.store(group="dataset", name="cifar10", node=cifar10_config)
cs.store(group="dataset", name="mnist", node=mnist_config)
cs.store(group="dataset", name="fmnist", node=fmnist_config)
cs.store(group="dataset", name="imagenet32", node=imagenet32_config)
