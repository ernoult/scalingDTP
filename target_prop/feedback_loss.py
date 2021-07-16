from __future__ import annotations
from functools import singledispatch
from torch import nn, Tensor
import torch
from torch.optim.optimizer import Optimizer

from .layers import ConvPoolBlock, ConvTransposePoolBlock
from logging import getLogger
from typing import Union, List
logger = getLogger(__file__)


@singledispatch
def feedback_loss(
    backward_layer: nn.Module,
    forward_layer: nn.Module,
    input: Tensor,
    noise_scale: float | Tensor,
    noise_samples: int = 1,
) -> float | Tensor:
    """ Computes the loss for the feedback weights, given the feedback layer and its
    accompanying forward module.
    
    Returns the loss for a single iteration.
    """
    x = input
    # 1- Compute y = F(input) and r=G(y)
    with torch.no_grad():
        y = forward_layer(x)
    r = backward_layer(y)
    
    noise_sample_losses = []
    for sample in range(noise_samples):
        # TODO: Use CUDA streams to make this faster:
        # with torch.cuda.Stream():
            
        # y, (r, ind) = self(x, back=True)
        noise = noise_scale * torch.randn_like(x)

        # 2- Perturbate x <-- x + noise and redo x--> y --> r
        with torch.no_grad():
            y_noise = forward_layer(x + noise)
        r_noise = backward_layer(y_noise)
        # _, (r_noise, ind_noise) = self(x + noise, back=True)

        dr = r_noise - r
        
        # 3- Perturbate y <-- y + noise and redo y --> r
        noise_y = noise_scale * torch.randn_like(y)
        r_noise_y = backward_layer(y + noise_y)

        # noise_y = sigma * torch.randn_like(y)
        # r_noise_y = self.bb(x, y + noise_y, ind)
        dr_y = r_noise_y - r

        # 4- Compute the loss
        sample_loss = (
            -2 * (noise * dr).flatten(1).sum(1).mean()
            + (dr_y ** 2).flatten(1).sum(1).mean()
        )
        noise_sample_losses.append(sample_loss)
    
    feedback_losses = torch.stack(noise_sample_losses)
    return feedback_losses.mean()

# def weight_f_train(self, y, t, optimizer):
#     """
#     Trains forward weights
#     """

#     # 1- Compute MSE between feedforward prediction and associated target
#     loss_f = 0.5 * ((y - t) ** 2).view(y.size(0), -1).sum(1)
#     loss_f = loss_f.mean()
#     optimizer.zero_grad()

#     # 2- Update forward weights
#     loss_f.backward(retain_graph=True)
#     optimizer.step()

#     return loss_f
