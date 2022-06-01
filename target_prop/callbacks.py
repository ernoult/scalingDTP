from __future__ import annotations

import contextlib
import warnings
from logging import getLogger as get_logger
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torch.nn import functional as F

from target_prop.layers import forward_all
from target_prop.metrics import compute_dist_angle
from target_prop.networks.network import Network
from target_prop.utils.utils import named_trainable_parameters

logger = get_logger(__name__)

if TYPE_CHECKING:
    # Avoid circular imports.
    from target_prop.models.dtp import DTP


class CompareToBackpropCallback(Callback):
    """Callback that compares the weight updates from DTP with those obtained from backprop."""

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
        pl_module: LightningModule,
        outputs: Any,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        # Keep the last batch.
        # Ideally, we'd be able to know which batch is the last, but we can't really do that atm,
        # so we just always overwrite this so it contains the last batch.
        self.last_batch = batch

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, unused=None) -> None:
        from target_prop.models.dtp import DTP

        if not isinstance(pl_module, DTP):
            raise NotImplementedError(
                f"This callback currently only works with the DTP model (or its variants: TP, "
                f"VanillaDTP, ParallelDTP), but not with models of type {type(pl_module)}"
            )
        assert self.last_batch is not None
        x, y = self.last_batch
        x = x.to(pl_module.device)
        y = y.to(pl_module.device)
        for param_name, param in named_trainable_parameters(pl_module.forward_net):
            if param.grad is not None and (param.grad != 0).any():
                warnings.warn(
                    RuntimeWarning(
                        "There are non-zero grads on the model parameters that will be wiped out "
                        "by this callback!"
                    )
                )
                break

        x = x.to(pl_module.device)
        y = y.to(pl_module.device)
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
            distance, angle = compute_dist_angle(scaled_dtp_grad, backprop_gradient)
            distances[parameter_name] = distance
            angles[parameter_name] = angle
    return distances, angles


def get_backprop_grads(model: Union[DTP, Network], x: Tensor, y: Tensor) -> Dict[str, Tensor]:
    # Get the normal backprop loss and gradients:
    if hasattr(model, "forward_net"):
        forward_net = model.forward_net
    else:
        forward_net = model
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


class FeedbackTrainingFigureCallback(Callback):
    """TODO: (refactoring): Move the 'feedback training figure' stuff from DTP to this callback.

    The `on_before_backward`/etc methods seem to work fine even with manual optimization.
    The only problem is that this callback would need to know what layer / iteration the model is
    currently backpropagating for, which might be a bit tricky.
    """
