from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sized

from simple_parsing import choice
from simple_parsing.helpers.serialization.serializable import Serializable
from torch import Tensor, nn
from typing_extensions import Protocol

activations = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
}


class Network(Iterable[nn.Module], Sized, Protocol):
    """Protocol that describes what we expect to find as attributes and methods on a network.

    Networks don't necessarily need to inherit from this, they just need to match the attributes and
    methods defined here.
    """

    @dataclass
    class HParams(Serializable):
        """Dataclass containing the Hyper-Parameters of the network."""

        activation: str = choice(*activations.keys(), default="elu")
        """ Choice of activation function to use. """

        def __post_init__(self):
            self.activation_class: type[nn.Module] = activations[self.activation]

    hparams: Network.HParams

    def __init__(self, in_channels: int, n_classes: int, hparams: HParams | None = None):
        ...

    def __call__(self, input: Tensor) -> Tensor:
        ...
