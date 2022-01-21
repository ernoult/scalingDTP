import os
import pdb
import torch
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision import transforms


def mnist_normalization():
    # From PL bolts
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    return normalize


class MNISTDataModule(LightningDataModule):

    EXTRA_ARGS: dict = {}
    name: str = "mnist"
    #: Dataset class to use
    dataset_cls: Dataset = MNIST
    #: A tuple describing the shape of the data
    dims: tuple = (1, 28, 28)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
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
        Adapted from PL Bolts MNISTDataModule.

        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
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
        self.val_split = val_split,
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

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_transforms = (
                self.default_transforms()
                if self.train_transforms is None
                else self.train_transforms
            )

            dataset_train = self.dataset_cls(
                self.data_dir, train=True, transform=train_transforms, **self.EXTRA_ARGS
            )

            train_length = len(dataset_train)
            val_length = int(self.val_split*train_length)
            dataset_train, dataset_val = random_split(
                dataset_train,
                [train_length - val_length, val_length],
                generator=torch.Generator().manual_seed(self.seed)
            )

            self.dataset_train = dataset_train
            self.dataset_val = dataset_val

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
            mnist_transforms = transforms.Compose([transforms.ToTensor(), mnist_normalization()])
        else:
            mnist_transforms = transforms.Compose([transforms.ToTensor()])
        return mnist_transforms


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