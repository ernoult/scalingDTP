from __future__ import annotations

from contextlib import nullcontext
from functools import singledispatch
from logging import getLogger
from typing import List, Union

import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

import torch
from contextlib import nullcontext
from logging import getLogger
from typing import Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from target_prop.utils import repeat_batch
from typing import List
from dataclasses import dataclass
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from target_prop.config import Config
from target_prop.optimizer_config import OptimizerConfig
from simple_parsing.helpers.hparams import log_uniform, uniform, categorical
from simple_parsing.helpers import list_field

logger = getLogger(__name__)
from .dtp import DTP


class VanillaDTP(DTP):
    """ (Vanilla) Difference Target Propagation (DTP)."""

    @dataclass
    class HParams(DTP.HParams):
        """ Hyper-Parameters of the model.

        TODO: The parameters for this (vanilla) DTP haven't been optimized yet. The values below are
        those from DTP-J. 
        """

        # batch size
        batch_size: int = log_uniform(16, 512, default=128, base=2, discrete=True)

        # Max number of training epochs in total.
        max_epochs: int = 90

        # Hyper-parameters for the "backward" optimizer
        b_optim: OptimizerConfig = OptimizerConfig(
            type="sgd", lr=[1e-4, 3.5e-4, 8e-3, 8e-3, 0.18], momentum=0.9
        )
        # The scale of the gaussian random variable in the feedback loss calculation.
        noise: List[float] = uniform(  # type: ignore
            0.001, 0.5, default_factory=[0.4, 0.4, 0.2, 0.2, 0.08].copy, shape=5
        )

        # Hyper-parameters for the forward optimizer
        # NOTE: On mnist, usign 0.1 0.2 0.3 gives decent results (75% @ 1 epoch)
        f_optim: OptimizerConfig = OptimizerConfig(
            type="sgd", lr=0.08, weight_decay=1e-4, momentum=0.9
        )
        # Use of a learning rate scheduler for the forward weights.
        scheduler: bool = True
        # nudging parameter: Used when calculating the first target.
        beta: float = uniform(0.01, 1.0, default=0.7)

        # Number of training steps for the feedback weights per batch. Can be a list of
        # integers, where each value represents the number of iterations for that layer.
        feedback_training_iterations: List[int] = list_field(20, 30, 35, 55, 20)

        # Number of noise samples to use to get the feedback loss in a single iteration.
        # NOTE: The loss used for each update is the average of these losses.
        feedback_samples_per_iteration: int = uniform(1, 20, default=1)

        # Max number of epochs to train for without an improvement to the validation
        # accuracy before the training is stopped. When 0, no early stopping is used.
        early_stopping_patience: int = 0

        # Sets symmetric weight initialization. Useful for debugging.
        init_symetric_weights: bool = False

        # TODO: Add a Callback class to compute and plot jacobians, if that's interesting.
        # jacobian: bool = False  # compute jacobians

        # Step interval for creating and logging plots.
        plot_every: int = 10

    def __init__(self, datamodule: VisionDataModule, hparams: "VanillaDTP.HParams", config: Config):
        super().__init__(datamodule, hparams, config)
        self.hp: VanillaDTP.HParams

    def compute_target(self, i: int, G: nn.Module, hs: List[Tensor], prev_target: Tensor) -> Tensor:
        """Compute the target of the previous forward layer. given ,
        the associated feedback layer, the activations for each layer, and the target of the current
        layer.

        Parameters
        ----------
        i : int
            the index of the forward layer for which we want to compute a target
        G : nn.Module
            the associated feedback layer
        hs : List[Tensor]
            the activations for each layer
        prev_target : Tensor
            The target of the next forward layer.

        Returns
        -------
        Tensor
            The target to use to train the forward layer at index `i`.
        """
        # NOTE: Target propagation:
        # return G(prev_target)
        # NOTE: Difference target propagation (both Vanilla and DTP-J):
        # return hs[i - 1] - G(hs[i]) + G(prev_target)
        return G(prev_target) + (hs[i - 1] - G(hs[i]))  # cooler ordering, from the other paper.

    def layer_feedback_loss(
        self,
        *,
        feedback_layer: nn.Module,
        forward_layer: nn.Module,
        input: Tensor,
        output: Tensor,
        noise_scale: Union[float, Tensor],
        noise_samples: int = 1,
    ) -> Tensor:
        # NOTE: The feedback loss in DTP is the same as in DP (as far as I can tell.)
        # TODO: Confirm this with @ernoult.
        return vanilla_DTP_feedback_loss(
            feedback_layer=feedback_layer,
            forward_layer=forward_layer,
            input=input,
            output=output,
            noise_scale=noise_scale,
            noise_samples=noise_samples,
        )


def vanilla_DTP_feedback_loss(
    *,
    feedback_layer: nn.Module,
    forward_layer: nn.Module,
    input: Tensor,
    output: Tensor,
    noise_scale: Union[float, Tensor],
    noise_samples: int = 1,
    use_separate_streams: bool = False,
    synchronize: bool = False,
) -> Tensor:
    """ Computes the loss for the feedback weights, given the feedback layer and its
    accompanying forward module.
    
    Returns the loss for a single iteration.
    Can optionally use more than one noise sample per iteration.
    """
    x = input
    y = output

    noise_sample_losses = []
    # NOTE: Not sure if this makes things faster or not, but seems interesting to test out.
    streams = []
    if use_separate_streams:
        streams = [torch.cuda.Stream() for _ in range(noise_samples)]

    for i in range(noise_samples):
        # TODO: Use CUDA streams to make this faster, since all iterations are distinct,
        # computations could perhaps be parallelized on the hardware level.
        with (torch.cuda.stream(streams[i]) if use_separate_streams else nullcontext()):

            # 2- Perturbate x <-- x + noise and redo x--> y --> r
            dx = noise_scale * torch.randn_like(x)
            x_noise = x + dx

            with torch.no_grad():
                y_noise = forward_layer(x + dx)

            x_r_noise = feedback_layer(y_noise)

            # Distance between `r` and the reconstructed `r`.
            dr = x_noise - x_r_noise
            loss = (dr ** 2).flatten(1).sum(1).mean()
            # NOTE: Should be equivalent to this, but I don't trust it anymore:
            # loss = F.mse_loss(x_noise, x_r_noise, reduction="none")
            noise_sample_losses.append(loss)

    if use_separate_streams and synchronize:
        # streams[0].
        torch.cuda.synchronize()

    feedback_losses = torch.stack(noise_sample_losses, dim=0)
    return feedback_losses.mean(dim=0)


def vanilla_DTP_feedback_loss_parallel(
    *,
    feedback_layer: nn.Module,
    forward_layer: nn.Module,
    input: Tensor,
    output: Tensor,
    noise_scale: Union[float, Tensor],
    noise_samples: int = 1,
) -> Tensor:
    """ Computes the loss for the feedback weights, given the feedback layer and its
    accompanying forward module.

    Returns the loss for a single iteration.
    Can optionally use more than one noise sample per iteration.

    TODO: The current implementation for both of these will have the exact same memory usage!
    The 'true' way to make this a sequential/parallel process would be to accumulate the
    gradients between different batches (one per noise sample), rather than accumulate the
    average of the forward pass and then backpropagate once, which actually needs each noise
    batch to be in memory.
    """
    # TODO: Check that this gives exactly the same result as the sequential version.
    # NOTE: BatchNorm might behave differently here because of the larger batch, if we ever use it.
    x = input
    y = output
    n = noise_samples
    # batch size
    b = input.shape[0]

    # IDEA: Tile x and use a larger batch size for the forward and backward computation.
    batch_x = repeat_batch(x, n=n)

    # IDEA: Could roll the noise vector, instead of sampling a truly different value for each index,
    # saving some memory.
    # 2- Perturbate x <-- x + noise and redo x--> y --> r
    batch_dx = noise_scale * torch.randn_like(batch_x)
    batch_x_noise = batch_x + batch_dx
    with torch.no_grad():
        batch_y_noise = forward_layer(batch_x_noise)

    # NOTE: Order of operations is important here: We want
    batch_xr_noise = feedback_layer(batch_y_noise)

    # TODO: Should we take an average or a sum over the samples dimension?
    dr = batch_xr_noise - batch_x
    loss = (dr ** 2).flatten(1).sum(1).mean()
    # NOTE: Should be equivalent to this, but I don't trust it anymore:
    # loss = F.mse_loss(batch_x_noise, batch_xr_noise)
    return loss
