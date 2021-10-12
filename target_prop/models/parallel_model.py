from target_prop.feedback_loss import get_feedback_loss, get_feedback_loss_parallel
from .model import BaseModel
from torch import Tensor
from typing import List, Optional, Tuple, Union
from torch import nn
import torch
from dataclasses import dataclass
from .sequential_model import SequentialModel
from target_prop.layers import forward_all
from target_prop.utils import is_trainable
from simple_parsing.helpers.hparams import uniform
from simple_parsing.helpers import field, list_field

try:
    from typing import Final
except ImportError:
    from typing_extensions import Final


class ParallelModel(SequentialModel):
    """ "Parallel" version of the sequential model, uses more noise samples but a single
    iteration for the training of the feedback weights, which makes it possible to use
    the automatic optimization and distributed training features of PyTorch-Lightning.
    """

    @dataclass
    class HParams(SequentialModel.HParams):
        """ HParams of the Parallel model. """

        # Number of training steps for the feedback weights per batch.
        # In the case of this parallel model, this parameter can't be changed and is fixed to 1.
        feedback_training_iterations: List[int] = list_field(
            default_factory=[1].copy, cmd=False
        )

        # Number of noise samples to use to get the feedback loss in a single iteration.
        # NOTE: The loss used for each update is the average of these losses.
        feedback_samples_per_iteration: int = uniform(1, 20, default=10)

        def __post_init__(self):
            super().__post_init__()
            self.feedback_training_iterations = [1 for _ in self.b_optim.lr]

    def __init__(self, datamodule, hparams, config):
        super().__init__(datamodule, hparams, config)
        # Here we can do automatic optimization, since we don't need to do multiple
        # sequential optimization steps per batch ourselves.
        self.automatic_optimization = True
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        # Set the number of feedback training iterations to 1.

    def training_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int = None
    ) -> Union[Tensor, float]:
        # NOTE: The only difference here is the optimizer idx that gets passed to shared_step.
        return self.shared_step(
            batch=batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx, phase="train"
        )

    def shared_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        phase: str,
        optimizer_idx: Optional[int] = None,
    ):
        """ Main step, used by the `[training/valid/test]_step` methods.
        
        NOTE: In the case of this Parallel model, we use the automatic optimization from PL.
        This means that we return a 'live' loss tensor here, rather than perform the optimzation
        manually.
        """
        x, y = batch
        # Setting this value just so we don't have to pass `phase=...` to `forward_loss`
        # and `feedback_loss` below.
        assert self.phase == phase, (self.phase, phase)
        # self.phase = phase

        dtype: Optional[torch.dtype] = self.dtype if isinstance(
            self.dtype, torch.dtype
        ) else None
        # The total loss to be returned.
        loss: Tensor = torch.zeros(1, device=self.device, dtype=dtype)

        if optimizer_idx in [None, self._feedback_optim_index]:
            # ----------- Optimize the feedback weights -------------
            feedback_loss = self.feedback_loss(x, y)
            loss += feedback_loss
            self.log(f"{phase}/f_loss", feedback_loss, prog_bar=True, **self.log_kwargs)

        if optimizer_idx in [None, self._forward_optim_index]:
            # ----------- Optimize the forward weights -------------
            forward_loss = self.forward_loss(x, y)
            self.log(f"{phase}/b_loss", forward_loss, prog_bar=True, **self.log_kwargs)
            loss += forward_loss

        return loss

    def forward_loss(self, x: Tensor, y: Tensor) -> Tensor:
        # NOTE: Could use the same exact forward loss as the sequential model, at the
        # moment.
        return super().forward_loss(x=x, y=y)

    def feedback_loss(self, x: Tensor, y: Tensor) -> Tensor:
        feedback_optimizer = self.feedback_optimizer

        n_layers = len(self.backward_net)
        # Reverse the backward net, just for ease of readability.
        reversed_backward_net = self.backward_net[::-1]
        # Also reverse these values so they stay aligned with the net above.
        noise_scale_per_layer = list(reversed(self.feedback_noise_scales))
        # NOTE: Could also use a different number of samples per layer!
        noise_samples_per_layer = [
            self.hp.feedback_samples_per_iteration for _ in range(n_layers)
        ]

        # NOTE: We can compute all the ys for all the layers up-front, because we don't
        # update the forward weights.
        # 1- Compute the forward activations (no grad).
        with torch.no_grad():
            ys: List[Tensor] = forward_all(
                self.forward_net, x, allow_grads_between_layers=False
            )

        # List of losses, distances, and angles for each layer (with multiple iterations per layer).
        # NOTE: Skipping the first layer
        # NOTE: Each of the loops below is independent. Would be nice to parallelize this somehow.
        layer_losses: List[Tensor] = [
            get_feedback_loss_parallel(
                feedback_layer=reversed_backward_net[i],
                forward_layer=self.forward_net[i],
                input=ys[i - 1],
                output=ys[i],
                noise_scale=noise_scale_per_layer[i],
                noise_samples=10,
                # TODO: Not sure if using different streams really makes this faster. Need to check.
                # use_separate_streams=True,
                # synchronize=False,
            )
            for i in range(1, n_layers)
            if is_trainable(reversed_backward_net[i])
        ]
        # Loss will now have shape [`n_layers`, `n_samples`]
        loss = torch.stack(layer_losses, dim=0)
        return loss.sum()

    def training_step_end(self, step_results: Union[Tensor, List[Tensor]]) -> Tensor:
        """ Called with the results of each worker / replica's output.

        See the `training_step_end` of pytorch-lightning for more info.
        """
        # TODO: For now we're kinda losing the logs and stuff that happens within the
        # workers in DP (they won't show up in the progress bar for instance).
        # merged_step_results = {
        #     k: sum(v_i.to(self.device) for v_i in v)
        #     for k, v in step_results
        # }
        merged_step_result = (
            step_results
            if isinstance(step_results, (Tensor, float))
            else sum(step_results)
        )

        # TODO: If NOT in automatic differentiation, but still in a scenario where we
        # can do a single update, do it here.
        loss = merged_step_result
        self.log(f"{self.phase}/total loss", loss, on_step=True, prog_bar=True)

        if (
            not self.automatic_optimization
            and isinstance(loss, Tensor)
            and loss.requires_grad
        ):
            forward_optimizer = self.forward_optimizer
            backward_optimizer = self.feedback_optimizer
            forward_optimizer.zero_grad()
            backward_optimizer.zero_grad()

            self.manual_backward(loss)

            forward_optimizer.step()
            backward_optimizer.step()
            return float(loss)

        elif not self.automatic_optimization:
            return float(merged_step_result)

        assert self.automatic_optimization
        return merged_step_result
