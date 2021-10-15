from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import wandb
from simple_parsing.helpers import list_field
from simple_parsing.helpers.hparams import uniform
from target_prop.feedback_loss import get_feedback_loss_parallel
from target_prop.layers import forward_all
from target_prop.metrics import compute_dist_angle
from target_prop.optimizer_config import OptimizerConfig
from target_prop.utils import is_trainable
from torch import Tensor, nn

from .dtp import DTP
from .utils import make_stacked_feedback_training_figure

logger = get_logger(__name__)


class ParallelDTP(DTP):
    """ "Parallel" variant of the DTP algorithm.
    
    Performs a single fused update for all feedback weights (a single "iteration") but uses multiple
    noise samples per "iteration".

    By avoiding to have multiple sequential, layer-wise updates within a single step, it becomes
    possible to use the automatic optimization and distributed training features of
    PyTorch-Lightning.
    
    NOTE: The "sequential" version of DTP can also use multiple noise samples per iteration. The
    default value for that parameter is set to 1 by default in DTP in order to reproduce @ernoult's
    experiments exactly.
    """
    @dataclass
    class HParams(DTP.HParams):
        """ HParams of the Parallel model. """

        # Number of training steps for the feedback weights per batch.
        # In the case of this parallel model, this parameter can't be changed and is fixed to 1.
        feedback_training_iterations: List[int] = list_field(
            default_factory=[1].copy, cmd=False
        )

        # Number of noise samples to use to get the feedback loss in a single iteration.
        # NOTE: The loss used for each update is the average of these losses.
        feedback_samples_per_iteration: int = uniform(1, 20, default=10)

        # Hyper-parameters for the "backward" optimizer
        b_optim: OptimizerConfig = OptimizerConfig(
            type="adam",
            lr=[3e-4] * 5,
            # type="sgd", lr=[1e-4, 3.5e-4, 8e-3, 8e-3, 0.18], momentum=0.9
        )
        # The scale of the gaussian random variable in the feedback loss calculation.
        noise: List[float] = uniform(
            0.001, 0.5, default_factory=[0.4, 0.4, 0.2, 0.2, 0.08].copy, shape=5
        )
        # Hyper-parameters for the forward optimizer
        # NOTE: On mnist, usign 0.1 0.2 0.3 gives decent results (75% @ 1 epoch)
        f_optim: OptimizerConfig = OptimizerConfig(
            type="adam", lr=[3e-4], weight_decay=1e-4,  # momentum=0.9
        )
        # Use of a learning rate scheduler for the forward weights.
        scheduler: bool = True
        # nudging parameter: Used when calculating the first target.
        beta: float = uniform(0.01, 1.0, default=0.7)

        def __post_init__(self):
            super().__post_init__()
            self.feedback_training_iterations = [1 for _ in self.channels]

    def __init__(self, datamodule, hparams, config):
        super().__init__(datamodule, hparams, config)
        # Here we can do automatic optimization, since we don't need to do multiple
        # sequential optimization steps per batch ourselves.
        self.automatic_optimization = True
        self.criterion = nn.CrossEntropyLoss(reduction="none")

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

        dtype: Optional[torch.dtype] = self.dtype if isinstance(
            self.dtype, torch.dtype
        ) else None
        # The total loss to be returned.
        loss: Tensor = torch.zeros(1, device=self.device, dtype=dtype)

        if optimizer_idx in [None, 0]:
            # ----------- Optimize the feedback weights -------------
            feedback_loss = self.feedback_loss(x, y, phase=phase)
            loss += feedback_loss
            self.log(f"{phase}/B_loss", feedback_loss, prog_bar=phase == "train")

        if optimizer_idx in [None, 1]:
            # ----------- Optimize the forward weights -------------
            forward_loss = self.forward_loss(x, y, phase=phase)
            self.log(f"{phase}/F_loss", forward_loss, prog_bar=phase == "train")
            loss += forward_loss

        return loss

    def forward_loss(self, x: Tensor, y: Tensor, phase: str) -> Tensor:
        # NOTE: Could use the same exact forward loss as the sequential model, at the
        # moment.
        return super().forward_loss(x=x, y=y, phase=phase)

    def feedback_loss(self, x: Tensor, y: Tensor, phase: str) -> Tensor:
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
        # NOTE: Each of the loops below is independent. Would be nice to also parallelize this
        # somehow.
        layer_losses: List[Tensor] = [
            # NOTE: Could also use the `get_feedback_loss` as well, there should be no difference in
            # the results.
            # get_feedback_loss(
            get_feedback_loss_parallel(
                feedback_layer=reversed_backward_net[i],
                forward_layer=self.forward_net[i],
                input=ys[i - 1],
                output=ys[i],
                noise_scale=noise_scale_per_layer[i],
                noise_samples=self.hp.feedback_samples_per_iteration,
                # TODO: Not sure if using different streams really makes this faster. Need to check.
                # use_separate_streams=True,
                # synchronize=False,
            )
            for i in range(1, n_layers)
            if is_trainable(reversed_backward_net[i])
        ]
        # Loss will now have shape [`n_layers`, `n_samples`]
        loss = torch.stack(layer_losses, dim=0)

        if self.training and self.global_step % self.hp.plot_every == 0:
            # NOTE: This here gets calculated *before* the update. Would be nice to show the
            # difference between before and after the update instead.
            # TODO: Move this to something like `on_zero_grad` or something?
            with torch.no_grad():
                # Compute the angle and distance for debugging the training of the
                # feedback weights:
                layer_distance, layer_angle = zip(
                    *[
                        compute_dist_angle(
                            self.forward_net[i], reversed_backward_net[i]
                        )
                        for i in range(1, n_layers)
                        if is_trainable(reversed_backward_net[i])
                    ]
                )
            # NOTE: Have to add the 'iterations' dimension, since it's used in the sequential model.
            layer_distances = torch.as_tensor(layer_distance).reshape(
                [len(layer_distance), 1]
            )
            layer_angles = torch.as_tensor(layer_angle).reshape(
                [len(layer_distance), 1]
            )
            layer_losses_for_plot = (
                torch.stack(layer_losses).detach().reshape(layer_distances.shape)
            )
            fig = make_stacked_feedback_training_figure(
                all_values=[layer_angles, layer_distances, layer_losses_for_plot],
                row_titles=["angles", "distances", "losses"],
                title_text=(
                    f"Evolution of various metrics during feedback weight training "
                    f"(global_step={self.global_step})"
                ),
            )
            fig_name = f"feedback_training_{self.global_step}"
            figures_dir = Path(self.trainer.log_dir or ".") / "figures"
            figures_dir.mkdir(exist_ok=True, parents=False)
            save_path: Path = figures_dir / fig_name
            fig.write_image(str(save_path.with_suffix(".png")))
            logger.info(f"Figure saved at path {save_path.with_suffix('.png')}")
            # TODO: Figure out why exactly logger.info isn't showing up.
            print(f"Figure saved at path {save_path.with_suffix('.png')}")

            if self.config.debug:
                # Also save an HTML version when debugging.
                fig.write_html(
                    str(save_path.with_suffix(".html")), include_plotlyjs="cdn"
                )

            if wandb.run:
                wandb.log({"feedback_training": fig})

        return loss.sum()

    def training_step_end(self, step_results: Union[Tensor, List[Tensor]]) -> Tensor:
        """ Called with the results of each worker / replica's output.

        See the `training_step_end` of pytorch-lightning for more info.
        """
        # TODO: Actually debug it with DP/DDP. Used to work with DP, haven't tested it in a while.
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
        loss = merged_step_result
        # TODO: Move all the logs to this once we used DDP.
        # self.log(f"{self.phase}/total loss", loss, on_step=True, prog_bar=True)
        return loss
