from abc import ABC
from dataclasses import dataclass
from typing import Callable, ClassVar, Type
from simple_parsing import choice
from simple_parsing.helpers.serialization.serializable import Serializable

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

from torch import Tensor, nn

activations = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
}


class Network(Callable, Protocol):
    @dataclass
    class HParams(Serializable):
        activation: str = choice(*activations.keys(), default="elu")
        batch_size: int = 128

        def __post_init__(self):
            self.activation_class: Type[nn.Module] = activations[self.activation]

    hparams: "Network.HParams"

    def __init__(self, in_channels: int, n_classes: int, hparams: HParams = None):
        ...
