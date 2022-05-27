from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import wandb
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from simple_parsing import field
from simple_parsing.helpers import list_field
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from target_prop.config import Config
from target_prop.feedback_loss import get_feedback_loss_parallel
from target_prop.layers import forward_all
from target_prop.metrics import compute_dist_angle
from target_prop.models.model import PhaseStr, StepOutputDict
from target_prop.networks import Network
from target_prop.optimizer_config import OptimizerConfig
from target_prop.scheduler_config import CosineAnnealingLRConfig, LRSchedulerConfig

from .dtp import DTP
from .utils import make_stacked_feedback_training_figure

logger = get_logger(__name__)


class ParallelDTP(DTP):
    """Parallel variant of the DTP algorithm.

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
        """HParams of the Parallel model."""

        # Arguments to be passed to the LR scheduler.
        lr_scheduler: LRSchedulerConfig = field(default_factory=CosineAnnealingLRConfig)

        # Use of a learning rate scheduler for the optimizer of the forward weights.
        use_scheduler: bool = False

        # Number of training steps for the feedback weights per batch.
        # In the case of this parallel model, this parameter can't be changed and is fixed to 1.
        feedback_training_iterations: List[int] = list_field(default_factory=[1].copy, cmd=False)

        # Number of noise samples to use to get the feedback loss in a single iteration.
        # NOTE: The loss used for each update is the average of these losses.
        feedback_samples_per_iteration: int = 10

        # Hyper-parameters for the "backward" optimizer
        b_optim: OptimizerConfig = field(
            default_factory=partial(
                OptimizerConfig,
                type="adam",
                lr=[3e-4],
                weight_decay=1e-4,
            )
        )

        # The scale of the gaussian random variable in the feedback loss calculation.
        noise: List[float] = field(default_factory=[0.4, 0.4, 0.2, 0.2, 0.08].copy)

        # Hyper-parameters for the forward optimizer
        f_optim: OptimizerConfig = field(
            default_factory=partial(
                OptimizerConfig,
                type="adam",
                lr=[3e-4],
                weight_decay=1e-4,
            )
        )

        # nudging parameter: Used when calculating the first target.
        beta: float = 0.7

    def __init__(
        self,
        datamodule: VisionDataModule,
        network: Network,
        hparams: ParallelDTP.HParams,
        config: Config,
    ):
        super().__init__(
            datamodule=datamodule,
            network=network,
            hparams=hparams,
            config=config,
        )
        # Here we can do automatic optimization, since we don't need to do multiple
        # sequential optimization steps per batch ourselves.
        self.automatic_optimization = True
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self._feedback_optimizer: Optional[Optimizer] = None

    def configure_sharded_model(self) -> None:
        return super().configure_sharded_model()

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
        phase: PhaseStr,
        optimizer_idx: Optional[int] = None,
    ) -> StepOutputDict:
        """Main step, used by the `[training/valid/test]_step` methods.

        NOTE: In the case of this Parallel model, we use the automatic optimization from PL.
        This means that we return a 'live' loss tensor here, rather than perform the optimzation
        manually.
        """
        x, y = batch

        dtype: torch.dtype = self.dtype if isinstance(self.dtype, torch.dtype) else torch.float
        # The total loss to be returned.
        loss: Tensor = torch.zeros(1, device=self.device, dtype=dtype)
        things_to_log: dict[str, Tensor] = {}
        if optimizer_idx in [None, 0]:
            # ----------- Optimize the feedback weights -------------
            feedback_loss = self.feedback_loss(x, y, phase=phase)
            loss += feedback_loss
            things_to_log["feedback_loss"] = feedback_loss

        if optimizer_idx in [None, 1]:
            # ----------- Optimize the forward weights -------------
            forward_outputs = self.forward_loss(x, y, phase=phase)
            forward_loss = forward_outputs["loss"]
            loss += forward_loss
            things_to_log["forward_loss"] = forward_loss
            logits: Tensor = forward_outputs["logits"]
            assert not logits.requires_grad
        else:
            # Only doing the feedback training, so we are missing the 'logits' output here!
            # TODO: Not ideal that we have to do this wasteful forward pass just to log the cross
            # entropy loss at each step!
            # with torch.no_grad():
            #     logits = self.forward_net(x)
            logits = None  # type: ignore
        return {"loss": loss, "logits": logits, "y": y, "log": things_to_log}

    def shared_step_end(self, step_output: StepOutputDict, phase: PhaseStr) -> StepOutputDict:
        for name, tensor in step_output.get("log", {}).items():
            self.log(f"{phase}/{name}", tensor, prog_bar=phase == "train")
        if step_output["logits"] is None:
            # Bypass the aggregation logic from the base class, let PL do its thing.
            return step_output
        return super().shared_step_end(step_output, phase)

    def forward_loss(self, x: Tensor, y: Tensor, phase: str) -> Dict[str, Tensor]:
        # NOTE: Could use the same exact forward loss as the sequential model, at the
        # moment.
        # TODO: The logs inside `DTP.forward_loss` seem to be lost when running with multiple GPUs.
        # They need to be done in the `shared_step_end` method.
        return super().forward_loss(x=x, y=y, phase=phase)

    def feedback_loss(self, x: Tensor, y: Tensor, phase: str) -> Tensor:
        feedback_optimizer = self.feedback_optimizer

        n_layers = len(self.backward_net)
        # Reverse the backward net, just for ease of readability.
        reversed_backward_net = self.backward_net[::-1]
        # Also reverse these values so they stay aligned with the net above.
        noise_scale_per_layer = list(reversed(self.feedback_noise_scales))
        # NOTE: Could also use a different number of samples per layer!
        noise_samples_per_layer = [self.hp.feedback_samples_per_iteration for _ in range(n_layers)]
        # NOTE: We can compute all the ys for all the layers up-front, because we don't
        # update the forward weights.
        # 1- Compute the forward activations (no grad).
        with torch.no_grad():
            ys: List[Tensor] = forward_all(self.forward_net, x, allow_grads_between_layers=False)

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
            # BUG: Can't discern a nn.Flatten layer and a layer that is trainable but we're in
            # validation phase.
            if self._is_trainable(reversed_backward_net[i])
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
                        compute_dist_angle(self.forward_net[i], reversed_backward_net[i])
                        for i in range(1, n_layers)
                        if self._is_trainable(reversed_backward_net[i])
                    ]
                )
            if self.trainer is not None:
                for layer_index, (distance, angle) in enumerate(zip(layer_distance, layer_angle)):
                    self.log(f"{phase}/B_distance[{layer_index}]", distance)
                    self.log(f"{phase}/B_angle[{layer_index}]", angle)

            # NOTE: Have to add the 'iterations' dimension, since it's used in the sequential model.
            layer_distances = torch.as_tensor(layer_distance).reshape([len(layer_distance), 1])
            layer_angles = torch.as_tensor(layer_angle).reshape([len(layer_distance), 1])
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
                fig.write_html(str(save_path.with_suffix(".html")), include_plotlyjs="cdn")

            if wandb.run:
                wandb.log({"feedback_training": fig})

        return loss.sum()

    def configure_optimizers(self):
        # Feedback optimizer:
        feedback_optimizer = self.hp.b_optim.make_optimizer(
            self.backward_net, lrs=self.feedback_lrs
        )
        feedback_optim_config = {"optimizer": feedback_optimizer}

        # Forward optimizer:
        forward_optimizer = self.hp.f_optim.make_optimizer(self.forward_net)
        forward_optim_config: Dict[str, Any] = {
            "optimizer": forward_optimizer,
        }
        if self.hp.use_scheduler:
            # Using the same LR scheduler as the original code:
            lr_scheduler = self.hp.lr_scheduler.make_scheduler(forward_optimizer)
            forward_optim_config["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "interval": self.hp.lr_scheduler.interval,
                "frequency": self.hp.lr_scheduler.frequency,
            }
        return [feedback_optim_config, forward_optim_config]

    @property
    def feedback_optimizers(self) -> List[Optional[Optimizer]]:
        """Returns the list of optimizers, one per layer of the feedback/backward net.

        For the first feedback layer, as well as all layers without trainable weights. the entry
        will be `None`.
        """
        raise NotImplementedError(f"There is only one feedback optimizer!")

    @property
    def feedback_optimizer(self) -> Optimizer:
        """Returns the list of optimizers, one per layer of the feedback/backward net.

        For the first feedback layer, as well as all layers without trainable weights. the entry
        will be `None`.
        """
        if self.trainer is None:
            return self._feedback_optimizer
        feedback_optimizer = self.optimizers()[0]
        return feedback_optimizer

    @property
    def forward_optimizer(self) -> Optimizer:
        """Returns The optimizer of the forward net."""
        if self.trainer is None:
            return self._forward_optimizer
        forward_optimizer = self.optimizers()[-1]
        return forward_optimizer
