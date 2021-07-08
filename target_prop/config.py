from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Optional, Type

from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, MNISTDataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from simple_parsing.helpers import choice
from simple_parsing.helpers.serialization import Serializable
import os


@dataclass
class Config(Serializable):
    """ Configuration options for the experiment (not hyper-parameters). """

    available_datasets: ClassVar[Dict[str, Type[VisionDataModule]]] = {
        "mnist": MNISTDataModule,
        "cifar10": CIFAR10DataModule,
        "imagenet": ImagenetDataModule,  # TODO: Not yet tested.
    }

    dataset: str = choice(available_datasets.keys(), default="mnist")
    data_dir: Path = Path(os.environ.get("DATA_DIR", "data"))
    num_workers: int = 16
    pin_memory: bool = True
    seed: Optional[int] = 123
    val_split: float = 0.2
    shuffle: bool = True

    def make_datamodule(self, batch_size: int) -> VisionDataModule:
        datamodule_class = self.available_datasets[self.dataset]
        return datamodule_class(
            data_dir=self.data_dir,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            val_split=self.val_split,
            seed=self.seed,
            shuffle=self.shuffle,
        )
