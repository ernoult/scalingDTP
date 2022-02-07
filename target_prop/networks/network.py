from abc import ABC
from dataclasses import dataclass
from typing import ClassVar, Type
from simple_parsing import choice
from target_prop.utils.hparams import HyperParameters

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

from torch import nn

activations = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
}


class Network(Protocol):
    @dataclass
    class HParams(HyperParameters):
        # Where objects of this type can be parsed from in the wandb configs.
        _stored_at_key: ClassVar[str] = "net_hp"
        activation: str = choice(*activations.keys(), default="elu")

        def __post_init__(self):
            super().__post_init__()
            self.activation_class: Type[nn.Module] = activations[self.activation]

    hparams: "Network.HParams"

    def __init__(self, in_channels: int, n_classes: int, hparams: HParams = None):
        ...
