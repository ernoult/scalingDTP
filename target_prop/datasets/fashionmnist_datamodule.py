import os
import pdb
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Union


from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import FashionMNIST


class FashionMNISTDataModule(LightningDataModule):

    EXTRA_ARGS: dict = {}
    name: str = "fashion_mnist"
    #: Dataset class to use
    dataset_cls: Dataset = FashionMNIST
    #: A tuple describing the shape of the data
    dims: tuple = (1, 28, 28)

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
        .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/
            wp-content/uploads/2019/02/Plot-of-a-Subset-of-Images-from-the-Fashion-MNIST-Dataset.png
            :width: 400
            :alt: Fashion MNIST

        Specs:
            - 10 classes (1 per type)
            - Each image is (1 x 28 x 28)

        Standard FashionMNIST, train, val, test splits and transforms

        Transforms::

            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor()
            ])

        Example::

            from pl_bolts.datamodules import FashionMNISTDataModule

            dm = FashionMNISTDataModule('.')
            model = LitModel()

            Trainer().fit(model, dm)

        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
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
    def num_classes(self):
        """
        Return:
            10
        """
        return 10

    def prepare_data(self):
        """
        Saves FashionMNIST files to data_dir
        """
        FashionMNIST(self.data_dir, train=True, download=True, transform=transform_lib.ToTensor())
        FashionMNIST(self.data_dir, train=False, download=True, transform=transform_lib.ToTensor())


    def setup(self, stage: Optional[str] = None) -> None:

        # Assign train/val datasets for use in dataloaders
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

    def train_dataloader(self, batch_size=32, transforms=None):
        """
        FashionMNIST train set removes a subset to use for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.train_transforms or self._default_transforms()

        dataset = FashionMNIST(self.data_dir, train=True, download=False, transform=transforms)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self, batch_size=32, transforms=None):
        """
        FashionMNIST val set uses a subset of the training set for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.val_transforms or self._default_transforms()

        dataset = FashionMNIST(self.data_dir, train=True, download=True, transform=transforms)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, batch_size=32, transforms=None):
        """
        FashionMNIST test set uses the test split

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.test_transforms or self._default_transforms()

        dataset = FashionMNIST(self.data_dir, train=False, download=False, transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def _default_transforms(self):
        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor()
        ])
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