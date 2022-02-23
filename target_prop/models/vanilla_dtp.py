from contextlib import nullcontext
from dataclasses import dataclass
from functools import singledispatch
from logging import getLogger
from typing import List, Union

import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from simple_parsing.helpers import list_field
from simple_parsing.helpers.hparams import categorical, uniform
from target_prop.networks.network import Network
from target_prop.config import Config
from target_prop.optimizer_config import OptimizerConfig
from target_prop.utils.utils import repeat_batch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

logger = getLogger(__name__)
from .dtp import DTP, FeedbackOptimizerConfig, ForwardOptimizerConfig


class VanillaDTP(DTP):
    """(Vanilla) Difference Target Propagation (DTP)."""

    @dataclass
    class HParams(DTP.HParams):
        """Hyper-Parameters of the model.

        This model inherits the same hyper-parameters and hyper-parameter priors as DTP, but with
        slightly different default values.
        TODO: The hyper-parameters for this (vanilla) DTP haven't been tuned yet.
        """

        # Hyper-parameters for the optimizer of the feedback weights (backward net).
        b_optim: FeedbackOptimizerConfig = FeedbackOptimizerConfig(
            type="sgd", lr=[1e-4, 3.5e-4, 8e-3, 8e-3, 0.18], momentum=0.9
        )

        # Hyper-parameters for the forward optimizer
        # NOTE: Different default value for the LR than DTP, since this is so high it produces NANs
        # almost instantly.
        f_optim: ForwardOptimizerConfig = ForwardOptimizerConfig(
            type="sgd", lr=1e-3, weight_decay=1e-4, momentum=0.9
        )

    def __init__(
        self,
        datamodule: VisionDataModule,
        network: Network,
        hparams: "VanillaDTP.HParams",
        config: Config,
        network_hparams: Network.HParams,
    ):
        super().__init__(
            datamodule=datamodule,
            network=network,
            hparams=hparams,
            config=config,
            network_hparams=network_hparams,
        )
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
        """The feedback loss calculation is a bit different in (Vanilla) DTP vs DTP-J."""
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
    """Computes the loss for the feedback weights, given the feedback layer and its
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
    """Computes the loss for the feedback weights, given the feedback layer and its
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
