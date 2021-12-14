import os
import pdb
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def cifar10_normalization():
    # From PL bolts
    # normalize = transforms.Normalize(
    #     mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #     std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    # )
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465), std=(3 * 0.2023, 3 * 0.1994, 3 * 0.2010)
    )
    return normalize


class CIFAR10DataModule(LightningDataModule):

    EXTRA_ARGS: dict = {}
    name: str = "cifar10"
    #: Dataset class to use
    dataset_cls: Dataset = CIFAR10
    #: A tuple describing the shape of the data
    dims: tuple = (3, 32, 32)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Adapted from PL Bolts CIFAR10DataModule.

        Args:
            data_dir: Where to save/load the data
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_samples(self) -> int:
        return len(self.dataset_train)

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Saves files to data_dir."""
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            train_transforms = (
                self.default_transforms()
                if self.train_transforms is None
                else self.train_transforms
            )
            test_transforms = (
                self.default_transforms() if self.test_transforms is None else self.test_transforms
            )

            dataset_train = self.dataset_cls(
                self.data_dir, train=True, transform=train_transforms, **self.EXTRA_ARGS
            )
            dataset_test = self.dataset_cls(
                self.data_dir, train=False, transform=test_transforms, **self.EXTRA_ARGS
            )
            self.dataset_train = dataset_train
            self.dataset_val = dataset_test

        if stage == "test" or stage is None:
            test_transforms = (
                self.default_transforms() if self.test_transforms is None else self.test_transforms
            )
            dataset_test = self.dataset_cls(
                self.data_dir, train=False, transform=test_transforms, **self.EXTRA_ARGS
            )
            self.dataset_test = dataset_test

    @abstractmethod
    def default_transforms(self) -> Callable:
        """Default transform for the dataset."""
        if self.normalize:
            cf10_transforms = transforms.Compose([transforms.ToTensor(), cifar10_normalization()])
        else:
            cf10_transforms = transforms.Compose([transforms.ToTensor()])
        return cf10_transforms

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader."""
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader."""
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
