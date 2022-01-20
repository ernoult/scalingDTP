import os
import pdb
import pickle
from abc import abstractmethod
from logging import getLogger
from typing import Any, Callable, List, Optional, Union

import numpy as np
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets.vision import VisionDataset

logger = getLogger(__name__)


def imagenet32_normalization():
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return normalize


def imagenet32_3xstd_normalization():
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(3*0.229, 3*0.224, 3*0.225))
    return normalize


class ImageNet32Dataset(VisionDataset):
    """
    Downsampled ImageNet 32x32 Dataset.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.base_folder = "imagenet32"
        self.train = train  # training set or test set
        self.split = "train" if self.train else "val"
        self.split_folder = f"out_data_{self.split}"
        self._load_dataset()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _load_dataset(self):
        self.data = []
        self.targets = []

        # Load the picked numpy arrays
        logger.info(f"Loading ImageNet32 {self.split} dataset...")
        for i in range(1, 11):
            file_name = "train_data_batch_" + str(i)
            file_path = os.path.join(self.root, self.base_folder, self.split_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])
        self.targets = [t - 1 for t in self.targets]
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
        logger.info(f"Loaded {len(self.data)} images from ImageNet32 {self.split} split")


class ImageNet32DataModule(LightningDataModule):

    EXTRA_ARGS: dict = {}
    name: str = "imagenet32"
    #: Dataset class to use
    dataset_cls: Dataset = ImageNet32Dataset
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
        return 1000

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Saves files to data_dir."""
        self.dataset_cls(self.data_dir, train=True)
        self.dataset_cls(self.data_dir, train=False)

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
            in32_transforms = transforms.Compose(
                [transforms.ToTensor(), imagenet32_normalization()]
            )
        else:
            in32_transforms = transforms.Compose([transforms.ToTensor()])
        return in32_transforms

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
