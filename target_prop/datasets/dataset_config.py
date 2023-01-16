from __future__ import annotations

import os
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, TypeVar

import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate
from pl_bolts.datamodules import FashionMNISTDataModule, MNISTDataModule
from pl_bolts.datamodules.cifar10_datamodule import (
    CIFAR10DataModule,
    cifar10_normalization,
)
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningDataModule
from simple_parsing.helpers.serialization.serializable import Serializable
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import Compose, Normalize

from target_prop.datasets.imagenet32_datamodule import (
    ImageNet32DataModule,
    imagenet32_normalization,
)
from target_prop.utils.hydra_utils import builds

FILE = Path(__file__)


REPO_ROOTDIR = FILE.parent.parent.parent  # The root of the repo.

SLURM_TMPDIR: Path | None = (
    Path(os.environ["SLURM_TMPDIR"]) if "SLURM_TMPDIR" in os.environ else None
)
SLURM_JOB_ID: int | None = int(os.environ["SLURM_JOB_ID"]) if "SLURM_JOB_ID" in os.environ else None

logger = get_logger(__name__)
if not SLURM_TMPDIR and SLURM_JOB_ID is not None:
    # This happens when running with `mila code`!
    _slurm_tmpdir = Path(f"/Tmp/slurm.{SLURM_JOB_ID}.0")
    if _slurm_tmpdir.exists():
        SLURM_TMPDIR = _slurm_tmpdir
DATA_DIR = SLURM_TMPDIR or (REPO_ROOTDIR / "data")


NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", torch.multiprocessing.cpu_count()))

logger = get_logger(__name__)


Transform = Callable[[Tensor], Tensor]
D = TypeVar("D", bound=VisionDataModule)


def get_config(group: str, name: str):
    cs = ConfigStore.instance()
    return cs.load(f"{group}/{name}.yaml").node


def get_datamodule(
    dataset: str, batch_size: int = 64, use_legacy_std: bool = False, **kwargs
) -> VisionDataModule:
    """Backward-compatibility function for fetching the datamodule with the given name."""
    if use_legacy_std and dataset != "cifar10_3xstd":
        if dataset == "cifar10":
            dataset = "cifar10_3xstd"
    config_node = get_config("dataset", dataset)
    return instantiate(config_node, batch_size=batch_size, **kwargs)


def validate_datamodule(datamodule: D) -> D:
    """Checks that the transforms / things are setup correctly. Returns the same datamodule.

    TODO: Could be a good occasion to wrap the datamodule with some kind of adapter that moves
    stuff to the right location!
    """
    if not datamodule.normalize:
        remove_normalization_from_transforms(datamodule)
    else:
        # todo: maybe check that the normalization transform is present everywhere?
        pass
    return datamodule


def remove_normalization_from_transforms(datamodule: VisionDataModule) -> None:
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
    """Little mixin that makes it possible to call the config, like adam_config() -> Adam, instead
    of having to use `instantiate`."""

    def __call__(self, *args, **kwargs):
        return instantiate(self, *args, **kwargs)


# A DatasetConfig is a class that configures a particular type of LightningDataModule.
# When instantiated (using `instantiate` of hydra, or just called like a function), it creates an
# instance of the type of LightningDataModule it configures.
DatasetConfig = builds(
    LightningDataModule,
    builds_bases=(
        Serializable,
        CallableConfig,  # Calling the config does the same thing as calling instantiate
    ),
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
    normalize=True,
    populate_full_signature=True,
    shuffle=True,
    pin_memory=True,
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

fmnist_config = builds(
    FashionMNISTDataModule,
    builds_bases=(mnist_config,),
)


def cifar10_3xstd_normalization() -> transforms.Normalize:
    # pl_bolts's normalization:
    # mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    # std=[x / 255.0 for x in [63.0, 62.1, 66.7]],

    # legacy 3x normalization:
    return Normalize(mean=(0.4914, 0.4822, 0.4465), std=(3 * 0.2023, 3 * 0.1994, 3 * 0.2010))


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

_cifar10_3x_val_transforms = builds(
    Compose,
    transforms=[builds(transforms.ToTensor), builds(cifar10_3xstd_normalization)],
)
cifar10_3xstd_config = builds(
    CIFAR10DataModule,
    builds_bases=(cifar10_config,),
    train_transforms=TrainTransformsConfig(
        transforms=[
            builds(transforms.RandomHorizontalFlip, p=0.5),
            builds(transforms.RandomCrop, size=32, padding=4, padding_mode="edge"),
            builds(transforms.ToTensor),
            # builds(cifar10_normalization),
            builds(cifar10_3xstd_normalization),
        ],
    ),
    test_transforms=_cifar10_3x_val_transforms,
    val_transforms=_cifar10_3x_val_transforms,
)
_torchvision_dir = Path("/network/datasets/torchvision")
TORCHVISION_DIR: Path | None = None
if _torchvision_dir.exists() and _torchvision_dir.is_dir():
    TORCHVISION_DIR = _torchvision_dir

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
    readonly_datasets_dir=TORCHVISION_DIR,
    builds_bases=(VisionDatasetConfig,),
)


cs = ConfigStore.instance()
cs.store(group="dataset", name="base", node=DatasetConfig)
cs.store(group="dataset", name="cifar10", node=cifar10_config)
cs.store(group="dataset", name="cifar10_3xstd", node=cifar10_3xstd_config)
cs.store(group="dataset", name="mnist", node=mnist_config)
cs.store(group="dataset", name="fmnist", node=fmnist_config)
cs.store(group="dataset", name="imagenet32", node=imagenet32_config)
