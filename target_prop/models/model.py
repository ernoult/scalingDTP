from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import ClassVar, Optional

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from simple_parsing import Serializable

from target_prop.networks.network import Network
from target_prop.scheduler_config import LRSchedulerConfig

if typing.TYPE_CHECKING:
    from target_prop.config import Config

from typing_extensions import Protocol, Self


class Model(Protocol):
    @dataclass
    class HParams(Serializable):
        # Where objects of this type can be parsed from in the wandb configs.
        _stored_at_key: ClassVar[str] = "net_hp"

        # Arguments to be passed to the LR scheduler.
        lr_scheduler: Optional[LRSchedulerConfig] = None

        # batch size
        batch_size: int = 128

    hp: Model.HParams
    net_hp: Network.HParams

    def __init__(
        self,
        datamodule: VisionDataModule,
        network: Network,
        hparams: "Model.HParams",
        config: "Config",
        network_hparams: Network.HParams | None = None,
    ):
        ...
