from simple_parsing.helpers.fields import choice
from simple_parsing.helpers.hparams import HyperParameters, log_uniform
import torch
from target_prop.utils import get_list_of_values
from typing import Any, ClassVar, Dict, Type, List, Optional, Union
from dataclasses import dataclass
from torch.optim.optimizer import Optimizer
from torch import nn
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig(HyperParameters):
    """ Configuration options for an optimizer.
    """

    # Class variable that holds the types of optimizers that are available.
    available_optimizers: ClassVar[Dict[str, Type[Optimizer]]] = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
    }

    # Type of Optimizer to use.
    type: str = choice(available_optimizers.keys(), default="sgd")
    # BUG: Little bug here, won't search over this in sweeps for now.
    # categorical("sgd", "adam"], default="adam", strict=True)

    # Learning rate of the optimizer.
    lr: Union[List[float], float] = log_uniform(1e-4, 1e-1, default=5e-3, shape=2)
    # Weight decay coefficient.
    weight_decay: Optional[float] = None

    # Momentum term to pass to SGD.
    # NOTE: This value is only used with SGD, not with Adam.
    momentum: float = 0.9

    def make_optimizer(
        self, network: nn.Sequential, lr: float = None, learning_rates_per_layer: List[float] = None
    ) -> Optimizer:
        """ Create the optimizer, using the options set in this object """
        optimizer_class = self.available_optimizers[self.type]
        # List of learning rates for each layer.
        n_layers = len(network)

        optimizer_kwargs: Dict[str, Any] = {}
        params = network.parameters()
        if lr is not None:
            assert learning_rates_per_layer is None
            optimizer_kwargs["lr"] = lr
        elif learning_rates_per_layer:
            assert len(learning_rates_per_layer) == n_layers
            params = []
            for i, (layer, lr) in enumerate(zip(network, learning_rates_per_layer)):
                logger.debug(f"Layer at index {i} (of type {type(layer)}) has lr of {lr}")
                params.append({"params": layer.parameters(), "lr": lr})
        elif isinstance(self.lr, list):
            assert len(self.lr) == n_layers
            params = []
            for i, (layer, lr) in enumerate(zip(network, self.lr)):
                logger.debug(f"Layer at index {i} (of type {type(layer)}) has lr of {lr}")
                params.append({"params": layer.parameters(), "lr": lr})
        else:
            assert isinstance(self.lr, float)
            optimizer_kwargs["lr"] = self.lr

        if self.weight_decay is not None:
            optimizer_kwargs["weight_decay"] = self.weight_decay
        if optimizer_class is torch.optim.SGD:
            optimizer_kwargs["momentum"] = self.momentum

        logger.debug(f"optimizer kwargs: {optimizer_kwargs}")
        return optimizer_class(  # type: ignore
            params, **optimizer_kwargs,
        )
