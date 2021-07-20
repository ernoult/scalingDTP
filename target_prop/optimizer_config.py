from simple_parsing.helpers.fields import choice
from simple_parsing.helpers.hparams import HyperParameters, log_uniform
import torch
from target_prop.utils import get_list_of_values
from typing import ClassVar, Dict, Type, List, Optional
from dataclasses import dataclass
from torch.optim.optimizer import Optimizer
from torch import nn 


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
    lr: List[float] = log_uniform(1e-4, 1e-1, default=5e-3, shape=2)
    # Weight decay coefficient.
    weight_decay: Optional[float] = log_uniform(1e-9, 1e-2, default=1e-4)

    use_lr_scheduler: bool = False

    def make_optimizer(self, network: nn.Sequential) -> Optimizer:
        """ Create the optimizer, using the options set in this object """
        optimizer_class = self.available_optimizers[self.type]
        # List of learning rates for each layer.
        n_layers = len(network)
        lrs: List[float] = get_list_of_values(self.lr, out_length=n_layers, name="lr")
        assert len(lrs) == n_layers
        params: List[Dict] = []
        for layer, lr in zip(network, lrs):
            params.append({"params": layer.parameters(), "lr": lr})
        return optimizer_class(  # type: ignore
            params, weight_decay=self.weight_decay,
        )
