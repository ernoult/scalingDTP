from __future__ import annotations

from dataclasses import dataclass

from simple_parsing import choice
from simple_parsing.helpers.serialization.serializable import Serializable
from torch import Tensor, nn
from typing_extensions import Protocol

activations = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
}


class Network(Protocol):
    @dataclass
    class HParams(Serializable):
        activation: str = choice(*activations.keys(), default="elu")

        def __post_init__(self):
            self.activation_class: type[nn.Module] = activations[self.activation]

    hparams: Network.HParams

    def __init__(self, in_channels: int, n_classes: int, hparams: HParams | None = None):
        ...

    def __call__(self, input: Tensor) -> Tensor:
        ...
