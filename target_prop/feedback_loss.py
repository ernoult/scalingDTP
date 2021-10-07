from __future__ import annotations
from functools import singledispatch
from torch import nn, Tensor
import torch
from torch.optim.optimizer import Optimizer

from logging import getLogger
from typing import Union, List

logger = getLogger(__name__)
import torch


@singledispatch
def get_feedback_loss(
    feedback_layer: nn.Module,
    forward_layer: nn.Module,
    input: Tensor,
    output: Tensor,
    noise_scale: float | Tensor,
    noise_samples: int = 1,
) -> float | Tensor:
    """ Computes the loss for the feedback weights, given the feedback layer and its
    accompanying forward module.
    
    Returns the loss for a single iteration.
    Can optionally use more than one noise sample per iteration.
    """
    # QUESTION: TODO: Should we 'recurse' into the sequential blocks for getting the
    # feedback loss? Or consider the whole block as a single "layer"?
    x = input
    y = output

    r = feedback_layer(y)

    noise_sample_losses = []
    for sample in range(noise_samples):
        # TODO: Use CUDA streams to make this faster, since all iterations are distinct,
        # computations could perhaps be parallelized on the hardware level. 
        # with torch.cuda.Stream():

        # 2- Perturbate x <-- x + noise and redo x--> y --> r
        dx = noise_scale * torch.randn_like(x)
        with torch.no_grad():
            y_noise = forward_layer(x + dx)

        r_noise = feedback_layer(y_noise)

        # Distance between `r` and the reconstructed `r`.
        dr = r_noise - r

        # 3- Perturbate y <-- y + noise and redo y --> r
        dy = noise_scale * torch.randn_like(y)
        r_noise_y = feedback_layer(y + dy)

        dr_y = r_noise_y - r

        # 4- Compute the loss
        dr_loss = -2 * (dx * dr).flatten(1).sum(1).mean()
        dy_loss = (dr_y ** 2).flatten(1).sum(1).mean()

        # print(dr_loss.item(), dy_loss.item())

        sample_loss = dr_loss + dy_loss
        noise_sample_losses.append(sample_loss)

    feedback_losses = torch.stack(noise_sample_losses)
    return feedback_losses.mean()


from .layers import Reshape

# Some layers don't have any feedback loss, so we just return 0 to save some compute.
# (NOTE: The returned value from above would also be 0.)


@get_feedback_loss.register(Reshape)
def _(
    backward_layer: Reshape,
    forward_layer: Reshape,
    input: Tensor,
    noise_scale: float | Tensor,
    noise_samples: int = 1,
) -> float:
    return 0.0

