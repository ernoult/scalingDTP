from abc import ABC
from dataclasses import dataclass
from typing import ClassVar
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from target_prop.networks.network import Network
from target_prop.wandb_utils import LoggedToWandb
from target_prop.config import Config
from pl_bolts.datamodules.vision_datamodule import VisionDataModule

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class Model(Protocol):
    @dataclass
    class HParams(HyperParameters, LoggedToWandb):
        # Where objects of this type can be parsed from in the wandb configs.
        _stored_at_key: ClassVar[str] = "net_hp"

    def __init__(
        self,
        datamodule: VisionDataModule,
        network: Network,
        hparams: HParams,
        config: Config,
        network_hparams: Network.HParams,
    ):
        ...
