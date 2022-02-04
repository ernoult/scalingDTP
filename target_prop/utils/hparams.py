from simple_parsing.helpers.hparams.hyperparameters import HyperParameters as _HyperParameters
from dataclasses import dataclass
from typing import ClassVar, Any, Type, TypeVar
from simple_parsing.helpers.serialization.serializable import Serializable
from pathlib import Path
from .wandb_utils import LoggedToWandb
from .hydra_utils import LoadableFromHydra


@dataclass
class HyperParameters(_HyperParameters, LoggedToWandb, LoadableFromHydra):
    pass

