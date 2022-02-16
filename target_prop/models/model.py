from dataclasses import dataclass
from typing import ClassVar, Dict, Optional
import typing

from simple_parsing import Serializable, field

from target_prop.scheduler_config import LRSchedulerConfig
from target_prop.networks.network import Network
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from simple_parsing.helpers.hparams.hparam import log_uniform


if typing.TYPE_CHECKING:
    from target_prop.config import Config

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

from pytorch_lightning import LightningModule, Trainer
from abc import ABC


class Model(Protocol):
    @dataclass
    class HParams(Serializable):
        # Where objects of this type can be parsed from in the wandb configs.
        _stored_at_key: ClassVar[str] = "net_hp"

        # Arguments to be passed to the LR scheduler.
        lr_scheduler: Optional[LRSchedulerConfig] = None

        # batch size
        batch_size: int = log_uniform(16, 512, default=128, base=2, discrete=True)

    hp: "Model.HParams"
    net_hp: Network.HParams

    def __init__(
        self,
        datamodule: VisionDataModule,
        network: Network,
        hparams: "Model.HParams",
        config: "Config",
        network_hparams: Network.HParams = None,
    ):
        ...
