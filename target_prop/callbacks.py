import contextlib
from os import name
from typing import Dict, Optional, Tuple, List
from pytorch_lightning import Callback, Trainer, LightningModule
import torch
import warnings
from torch import nn

from target_prop.metrics import compute_dist_angle
from .models.dtp import DTP
from target_prop.layers import forward_all
from torch import Tensor
from torch.nn import functional as F, parameter
from target_prop.utils import named_trainable_parameters
from logging import getLogger as get_logger

logger = get_logger(__name__)


class CompareToBackpropCallback(Callback):
    """ TODO: Create a PL callback that calculates and logs the angles between the
    forward and backward weights?
    """

    def __init__(self, temp_beta: float = 0.005) -> None:
        """Callback that compares the weight updates from DTP with those obtained from backprop.   

        Parameters
        ----------
        temp_beta : float, optional
            The value of `beta` to use during the forward loss calculation, by default 0.005
        """
        super().__init__()
        self.last_batch: Optional[Tuple[Tensor, Tensor]] = None
        self.temp_beta = temp_beta

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: DTP,
        outputs: List[Tensor],
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # Keep the last batch.
        # Ideally, we'd be able to know which batch is the last, but we can't really do that atm,
        # so we just always overwrite this so it contains the last batch.
        self.last_batch = batch

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, unused=None) -> None:
        if not isinstance(pl_module, DTP):
            raise NotImplementedError(
                f"This callback currently only works with the DTP model (or its variants: TP, "
                f"VanillaDTP, ParallelDTP), but not with models of type {type(pl_module)}"
            )
        assert self.last_batch is not None
        x, y = self.last_batch

        for param_name, param in named_trainable_parameters(pl_module.forward_net):
            if param.grad is not None and (param.grad != 0).any():
                warnings.warn(
                    RuntimeWarning(
                        "There are non-zero grads on the model parameters that will be wiped out "
                        "by this callback!"
                    )
                )
                break

        distances, angles = comparison_with_backprop_gradients(
            model=pl_module, x=x, y=y, temp_beta=self.temp_beta
        )

        prefix = "/BackpropComparison"
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    f"{prefix}/distance/{parameter_name}": distance
                    for parameter_name, distance in distances.items()
                }
            )
            trainer.logger.log_metrics(
                {
                    f"{prefix}/angle/{parameter_name}": angle
                    for parameter_name, angle in angles.items()
                }
            )
        else:
            for parameter_name, distance in distances.items():
                logger.debug(f"{prefix}/distance/{parameter_name}: {distance}")
            for parameter_name, angle in angles.items():
                logger.debug(f"{prefix}/angle/{parameter_name}: {angle}")


def comparison_with_backprop_gradients(
    model: DTP, x: Tensor, y: Tensor, temp_beta: float
) -> Tuple[Dict[str, float], Dict[str, float]]:
    backprop_grads = get_backprop_grads(model=model, x=x, y=y)
    dtp_grads = get_dtp_grads(model=model, x=x, y=y, temp_beta=temp_beta)

    scaled_dtp_grads = {key: (1 / temp_beta) * grad for key, grad in dtp_grads.items()}
    assert backprop_grads.keys() == scaled_dtp_grads.keys()

    distances: Dict[str, float] = {}
    angles: Dict[str, float] = {}
    with torch.no_grad():
        # Same parameters should have gradients, regardless of if backprop or DTP is used.
        for parameter_name, backprop_gradient in backprop_grads.items():
            scaled_dtp_grad = scaled_dtp_grads[parameter_name]
            distance, angle = compute_dist_angle(backprop_gradient, scaled_dtp_grad)
            distances[parameter_name] = distance
            angles[parameter_name] = angle
    return distances, angles


def get_backprop_grads(model: DTP, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
    # Get the normal backprop loss and gradients:
    forward_net = model.forward_net
    # Clear out the gradients.
    forward_net.zero_grad()

    logits = forward_all(forward_net, x, allow_grads_between_layers=True)[-1]
    backprop_loss = F.cross_entropy(logits, y)
    backprop_loss.backward()
    backprop_grads = {
        name: p.grad.detach().clone()
        for name, p in named_trainable_parameters(forward_net)
        if p.grad is not None
    }
    # NOTE: This doesn't modify the values in `backprop_grads`, since we detach and clone above.
    forward_net.zero_grad()
    # IDEA: Maybe reset the grads to what they were before somehow, to make this a "stateless"
    # function?
    return backprop_grads


def get_dtp_grads(model: DTP, x: Tensor, y: Tensor, temp_beta: float = 0.005) -> Dict[str, Tensor]:
    forward_net = model.forward_net
    forward_net.zero_grad()
    with temporarily_change_beta(model=model, temp_beta=temp_beta):
        forward_output = model.forward_loss(x, y, phase="train")

    # NOTE: The entries in that dictionary should be the same for all versions of DTP.
    forward_loss = forward_output["loss"]
    forward_loss_per_layer = forward_output["layer_losses"]
    forward_loss.backward()
    dtp_grads = {
        name: p.grad.detach().clone()
        for name, p in named_trainable_parameters(forward_net)
        if p.grad is not None
    }
    # NOTE: Since we detach and clone, we're not changing the values in dtp_grads here.
    # This is just to make sure that we don't interfere with the first update of the next epoch.
    forward_net.zero_grad()
    return dtp_grads


@contextlib.contextmanager
def temporarily_change_beta(model: DTP, temp_beta: float = 0.005):
    starting_beta = model.hp.beta
    model.hp.beta = temp_beta
    logger.debug(f"Temporarily setting beta to {temp_beta} for this callback.")
    yield
    model.hp.beta = starting_beta
    logger.debug(f"Value of beta reset to {model.hp.beta}.")

