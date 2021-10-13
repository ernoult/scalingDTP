from __future__ import annotations
from functools import singledispatch
from torch import nn, Tensor
import torch
from torch.optim.optimizer import Optimizer
from contextlib import nullcontext
from logging import getLogger
from typing import Union, List

logger = getLogger(__name__)
import torch


def get_feedback_loss(
    *,
    feedback_layer: nn.Module,
    forward_layer: nn.Module,
    input: Tensor,
    output: Tensor,
    noise_scale: float | Tensor,
    noise_samples: int = 1,
    use_separate_streams: bool = False,
    synchronize: bool = False,
) -> Tensor:
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
    # NOTE: Not sure if this makes things faster or not, but seems interesting to test out.
    if use_separate_streams:
        streams = [torch.cuda.Stream() for _ in range(noise_samples)]

    for i, sample in enumerate(range(noise_samples)):
        # TODO: Use CUDA streams to make this faster, since all iterations are distinct,
        # computations could perhaps be parallelized on the hardware level.
        with (torch.cuda.stream(streams[i]) if use_separate_streams else nullcontext()):

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
            # NOTE: Original code:
            # (-2*(noise*dr).view(dr.size(0), -1).sum(1).mean()
            #  + (dr_y**2).view(dr_y.size(0), -1).sum(1).mean())
            dr_loss = -2 * (dx * dr).flatten(1).sum(1).mean()
            dy_loss = (dr_y ** 2).flatten(1).sum(1).mean()

            # print(dr_loss.item(), dy_loss.item())

            sample_loss = dr_loss + dy_loss
            noise_sample_losses.append(sample_loss)

    if use_separate_streams and synchronize:
        # streams[0].
        torch.cuda.synchronize()

    feedback_losses = torch.stack(noise_sample_losses, dim=0)
    return feedback_losses.mean(dim=0)


def repeat_batch(v: Tensor, n: int, copy: bool = True) -> Tensor:
    if copy:
        return v.repeat([n, *[1 for _ in v.shape[1:]]])
    else:
        return torch.cat([v for _ in range(n)], dim=0)


def split_batch(batched_v: Tensor, n: int) -> List[Tensor]:
    assert batched_v.shape[0] % n == 0
    original_batch_size = batched_v.shape[0] // n
    return [
        batched_v[i * original_batch_size : (i + 1) * original_batch_size, ...]
        for i in range(n)
    ]


def get_feedback_loss_parallel(
    *,
    feedback_layer: nn.Module,
    forward_layer: nn.Module,
    input: Tensor,
    output: Tensor,
    noise_scale: float | Tensor,
    noise_samples: int = 1,
) -> Tensor:
    """ Computes the loss for the feedback weights, given the feedback layer and its
    accompanying forward module.
    
    Returns the loss for a single iteration.
    Can optionally use more than one noise sample per iteration.
    """
    # QUESTION: TODO: Should we 'recurse' into the sequential blocks for getting the
    # feedback loss? Or consider the whole block as a single "layer"?
    x = input
    y = output
    n = noise_samples

    # IDEA: Tile x and use a larger batch size for the forward and backward computation.
    batch_x = repeat_batch(x, n=n)
    batch_y = repeat_batch(y, n=n)

    r = feedback_layer(y)
    batch_r = repeat_batch(r, n=n)

    # IDEA: Could roll the noise vector, instead of sampling a truly different value for each index,
    # saving some memory.
    # 2- Perturbate x <-- x + noise and redo x--> y --> r
    batch_dx = noise_scale * torch.randn_like(batch_x)

    with torch.no_grad():
        batch_y_noise = forward_layer(batch_x + batch_dx)
    # NOTE: Order of operations is important here: We want
    batch_r_noise = feedback_layer(batch_y_noise)

    # Distance between `r` and the reconstructed `r`.
    batch_dr = batch_r_noise - batch_r

    # 3- Perturbate y <-- y + noise and redo y --> r
    batch_dy = noise_scale * torch.randn_like(batch_y)
    batch_r_noise_y = feedback_layer(batch_y + batch_dy)
    batch_dr_y = batch_r_noise_y - batch_r

    # 4- Compute the loss
    # NOTE: Original code:
    # (-2*(noise*dr).view(dr.size(0), -1).sum(1).mean()
    #  + (dr_y**2).view(dr_y.size(0), -1).sum(1).mean())
    # batch_dr_loss = -2 * (batch_dx * batch_dr).flatten(1).sum(1).mean()
    # batch_dy_loss = (batch_dr_y ** 2).flatten(1).sum(1).mean()

    # TODO: The 'mean' term here is a bit different now, since it acts across samples too.
    # Is that ok?
    batch_dr_loss = -2 * (batch_dx * batch_dr).flatten(1).sum(1).mean()
    batch_dy_loss = (batch_dr_y ** 2).flatten(1).sum(1).mean()

    # TODO: Check that this gives the same result as the sequential version.
    # TODO: BatchNorm will behave differently here, if we ever use it.
    batch_sample_loss = batch_dr_loss + batch_dy_loss
    return batch_sample_loss
