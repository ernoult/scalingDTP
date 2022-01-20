from abc import ABC
from dataclasses import dataclass
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class Network(Protocol):
    @dataclass
    class HParams(HyperParameters):
        pass

    def __init__(self, in_channels: int, n_classes: int, hparams: HParams = None):
        ...
