import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Type

import numpy as np
import torch
from simple_parsing.helpers.fields import choice, list_field
from simple_parsing.helpers.serialization.serializable import Serializable
from torch import nn
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig(Serializable):
    """Configuration options for the optimizer.

    TODO: Replace the choice of optimizer type with a _target_'ed config for Hydra, so that we can
    add other optimizers easily, and avoid having all the union of all their constructor arguments
    as fields here.
    NOTE: Having the fused constructor arguments here makes it possible to switch between different
    optimizer types during a sweep. There must however be a good way of doing it cleanly with
    Hydra, though.
    """

    # Class variable that holds the types of optimizers that are available.
    available_optimizers: ClassVar[Dict[str, Type[Optimizer]]] = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
    }

    # Type of Optimizer to use.
    type: str = choice(*available_optimizers.keys(), default="sgd")

    # Learning rate of the optimizer.
    lr: List[float] = list_field(4e-3)

    # Weight decay coefficient.
    weight_decay: Optional[float] = None

    # Momentum term to pass to SGD.
    # NOTE: This value is only used with SGD, not with Adam.
    momentum: float = 0.9

    def __post_init__(self):
        if isinstance(self.lr, np.ndarray):
            self.lr = self.lr.tolist()

    def make_optimizer(self, network: nn.Module, lrs: Optional[List[float]] = None) -> Optimizer:
        """Create the optimizer, using the options set in this object."""
        optimizer_class = self.available_optimizers[self.type]
        # List of learning rates for each layer.

        optimizer_kwargs: Dict[str, Any] = {}
        params = network.parameters()
        # TODO: This is ugly AF.

        lr = lrs if lrs is not None else self.lr

        if isinstance(lr, float):
            params = network.parameters()
            optimizer_kwargs["lr"] = lr
        elif len(lr) == 1:
            params = network.parameters()
            optimizer_kwargs["lr"] = lr[0]
        else:
            # Multiple learning rates, one per layer.
            assert isinstance(network, nn.Sequential), "can only give lrs per layer for Sequential."
            if len(network) != len(lr):
                raise RuntimeError(
                    f"need one lr per layer, but got network of {len(network)} layers, and lrs of "
                    f"{lr}"
                )
            params = []
            for i, (layer, lr) in enumerate(zip(network, lr)):
                logger.debug(f"Layer at index {i} (of type {type(layer)}) has lr of {lr}")
                params.append({"params": layer.parameters(), "lr": lr})

        if self.weight_decay is not None:
            optimizer_kwargs["weight_decay"] = self.weight_decay
        if optimizer_class is torch.optim.SGD:
            optimizer_kwargs["momentum"] = self.momentum

        logger.debug(
            f"optimizer kwargs for network of type {type(network).__name__}: {optimizer_kwargs}"
        )
        return optimizer_class(  # type: ignore
            params,
            **optimizer_kwargs,
        )
