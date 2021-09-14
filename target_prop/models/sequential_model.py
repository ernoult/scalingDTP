import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TypeVar, cast
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from plotly.subplots import make_subplots
from pytorch_lightning import Trainer
from simple_parsing.helpers import list_field
from target_prop.config import Config
from target_prop.feedback_loss import get_feedback_loss
from target_prop.metrics import compute_dist_angle
from target_prop.models.model import BaseModel
from target_prop.utils import get_list_of_values, is_trainable
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__file__)
T = TypeVar("T")


class SequentialModel(BaseModel):
    """ Model that trains the forward weights and feedback weights sequentially.

    NOTE: This is basically the same as @ernoult's implementation.
    """

    @dataclass
    class HParams(BaseModel.HParams):
        """ Hyper-Parameters of the model.

        TODO: Set these values as default (cifar10)
        ```console
        python main.py --batch-size 128 \
        --C 128 128 256 256 512 \
        --iter 20 30 35 55 20 \
        --epochs 90 \
        --lr_b 1e-4 3.5e-4 8e-3 8e-3 0.18 \
        --noise 0.4 0.4 0.2 0.2 0.08 \
        --lr_f 0.08 \
        --beta 0.7 \
        --path CIFAR-10 \
        --scheduler \
        --wdecay 1e-4 \
        ```
        """

        # Number of training steps for the feedback weights per batch. Can be a list of
        # integers, where each value represents the number of iterations for that layer.
        feedback_training_iterations: List[int] = list_field(20, 30, 35, 55, 20)

    def __init__(
        self,
        datamodule: VisionDataModule,
        hparams: "SequentialModel.HParams",
        config: Config,
    ):
        super().__init__(datamodule, hparams, config)
        # Can't do automatic optimization here, since we do multiple sequential updates
        # per batch.
        self.automatic_optimization = False
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        print("Hyper-Parameters:")
        print(self.hp.dumps_json(indent="\t"))
        # TODO: Use something like this:
        # self.supervised_metrics: List[Metrics]

    @property
    def phase(self) -> str:
        if self.trainer.training:
            return "train"
        if self.trainer.validating:
            return "val"
        if self.trainer.testing:
            return "test"
        if self.trainer.predicting:
            return "predict"
        # NOTE: This doesn't work when inside the sanity check!
        if self.trainer.state.stage.value == "sanity_check":
            return "val"
        # raise RuntimeError(f"HUH?", self.trainer.state,)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int,) -> float:
        return self.shared_step(batch, batch_idx=batch_idx, phase="train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int,) -> float:
        return self.shared_step(batch, batch_idx=batch_idx, phase="val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int,) -> float:
        return self.shared_step(batch, batch_idx=batch_idx, phase="test")

    def shared_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, phase: str,
    ):
        """ Main step, used by the `[training/valid/test]_step` methods.
        """

        x, y = batch
        # Setting this value just so we don't have to pass `phase=...` to `forward_loss`
        # and `feedback_loss` below.
        assert self.phase == phase, (self.phase, phase)
        
        dtype: Optional[torch.dtype] = self.dtype if isinstance(
            self.dtype, torch.dtype
        ) else None

        # ----------- Optimize the feedback weights -------------
        feedback_loss = self.feedback_loss(x, y)
        self.log(f"{phase}/B_loss", feedback_loss, prog_bar=phase=="train")

        # This is never a 'live' loss, since we do the optimization steps sequentially
        # inside `feedback_loss`.
        assert not feedback_loss.requires_grad

        # ----------- Optimize the forward weights -------------
        forward_loss = self.forward_loss(x, y)
        self.log(f"{phase}/F_loss", forward_loss, prog_bar=phase=="train")

        # During training, the forward loss will be a 'live' loss tensor, since we
        # gather the losses for each layer. Here we perform only one step.
        if forward_loss.requires_grad and not self.automatic_optimization:
            f_optimizer = self.forward_optimizer
            self.manual_backward(forward_loss)
            f_optimizer.step()
            f_optimizer.zero_grad()
            forward_loss = forward_loss.detach()
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()  # type: ignore

        return float(forward_loss + feedback_loss)

    def feedback_loss(self, x: Tensor, y: Tensor) -> Tensor:
        feedback_optimizer = self.feedback_optimizer

        n_layers = len(self.backward_net)
        # Reverse the backward net, just for ease of readability.
        reversed_backward_net = self.backward_net[::-1]
        # Also reverse these values so they stay aligned with the net above.
        noise_scale_per_layer = list(reversed(self.feedback_noise_scales))
        iterations_per_layer = list(reversed(self.feedback_iterations))

        # NOTE: We never train the last layer of the feedback net (G_0).
        assert iterations_per_layer[0] == 0
        assert noise_scale_per_layer[0] == 0

        # NOTE: We can compute all the ys for all the layers up-front, because we don't
        # update the forward weights.
        # 1- Compute the forward activations (no grad).
        with torch.no_grad():
            ys: List[Tensor] = self.forward_net.forward_all(
                x, allow_grads_between_layers=False
            )

        angles: List[List[float]] = []
        distances: List[List[float]] = []

        # Layer-wise autoencoder training begins:
        # NOTE: Skipping the first layer
        layer_losses: List[Tensor] = []
        for layer_index in range(1, n_layers):
            # Forward layer
            F_i = self.forward_net[layer_index]
            # Feedback layer
            G_i = reversed_backward_net[layer_index]
            x_i = ys[layer_index - 1]
            y_i = ys[layer_index]
            iterations_i = iterations_per_layer[layer_index]
            noise_scale_i = noise_scale_per_layer[layer_index]

            layer_angles: List[float] = []
            layer_distances: List[float] = []

            if iterations_i == 0:
                # Skip this layer, since it's not trainable.
                continue
            assert noise_scale_i > 0

            if self.phase != "train":
                # Only perform one iteration per layer when not training.
                iterations_i = 1

            # Train the current autoencoder:
            losses_per_iteration: List[Tensor] = []
            for iteration in range(iterations_i):
                # Get the loss (see `feedback_loss.py`)
                loss = get_feedback_loss(
                    G_i,
                    F_i,
                    input=x_i,
                    output=y_i,
                    noise_scale=noise_scale_i,
                    noise_samples=self.hp.feedback_samples_per_iteration,
                )

                # Compute the angle and distance for debugging the training of the
                # feedback weights:
                with torch.no_grad():
                    distance, angle = compute_dist_angle(F_i, G_i)

                logger.debug(
                    f"Layer {layer_index}, Iteration {iteration}, angle={angle}, distance={distance}"
                )
                layer_angles.append(angle)
                layer_distances.append(distance)

                if isinstance(loss, Tensor) and loss.requires_grad:
                    feedback_optimizer.zero_grad()
                    self.manual_backward(loss)
                    feedback_optimizer.step()
                    loss = loss.detach()
                    losses_per_iteration.append(loss)
                else:
                    if isinstance(loss, float):
                        loss = torch.as_tensor(loss, device=y.device)
                    losses_per_iteration.append(loss)

            iteration_losses = torch.stack(losses_per_iteration)
            # Do we want to report the average loss per iteration? or the total?
            total_loss_for_layer = iteration_losses.mean() # .sum()
            layer_losses.append(total_loss_for_layer)
            self.log(f"{self.phase}/B_loss[{layer_index}]", total_loss_for_layer)
            # IDEA: Could log this if we add some kind of early stopping for the feedback
            # training
            # self.log(f"{self.phase}/B_iterations[{layer_index}]", iterations)
            if layer_angles:
                # TODO: Logging all the distances and angles at once, which isn't ideal!
                # What would be nicer would be to log this as a small, light-weight plot.
                self.log(f"{self.phase}/B_angles[{layer_index}]", layer_angles)
                angles.append(layer_angles)
            if layer_distances:
                self.log(f"{self.phase}/B_distances[{layer_index}]", layer_distances)
                distances.append(layer_distances)

        if self.training and self.global_step % 100 == 0:
            if self.config.debug:
                fig = self.make_feedback_training_figure(angles=angles, distances=distances)
                # BUG: PL bug: tries to call len(figure).
                # self.log(f"{self.phase}/plot[{layer_index}]", fig)
                figures_dir = Path(self.trainer.log_dir or ".") / "figures"
                figures_dir.mkdir(exist_ok=True, parents=False)
                fig.write_image(str(figures_dir / f"feedback_training_{self.global_step}.png"))
                fig.write_html(str(figures_dir / f"feedback_training_{self.global_step}.html"), include_plotlyjs="cdn")

        return torch.stack(layer_losses).sum()

    def forward_loss(self, x: Tensor, y: Tensor) -> Tensor:
        # NOTE: Sanity check: Use standard backpropagation for training rather than TP.
        ## --------
        # return super().forward_loss(x=x, y=y)
        ## --------

        ys: List[Tensor] = self.forward_net.forward_all(
            x, allow_grads_between_layers=False,
        )
        logits = ys[-1]
        labels = y

        # Calculate the first target using the gradients of the loss w.r.t. the logits.
        # NOTE: Need to manually enable grad here so that we can also compute the first
        # target during validation / testing.
        with torch.set_grad_enabled(True):
            accuracy = self.accuracy(torch.softmax(logits, -1), labels)
            self.log(f"{self.phase}/accuracy", accuracy, prog_bar=True)
            
            temp_logits = logits.detach().clone()
            temp_logits.requires_grad_(True)
            ce_loss = F.cross_entropy(temp_logits, labels, reduction="sum")
            grads = torch.autograd.grad(
                ce_loss,
                temp_logits,
                only_inputs=True,  # Do not backpropagate further than the input tensor!
                create_graph=False,
            )
            assert len(grads) == 1

        y_n_grad = grads[0]

        delta = -self.hp.beta * y_n_grad

        self.log(f"{self.phase}/delta.norm()", delta.norm())
        # Compute the first target (for the last layer of the forward network):
        last_layer_target = logits.detach() + delta

        N = len(self.forward_net)
        # NOTE: Initialize the list of targets with Nones, and we'll replace all the
        # entries with tensors corresponding to the targets of each layer.
        targets: List[Optional[Tensor]] = [*(None for _ in range(N-1)), last_layer_target]

        # Reverse the ordering of the layers, just to make the indexing in the code below match
        # those of the math equations.
        reordered_feedback_net: Sequential = self.backward_net[::-1]  # type: ignore

        # Calculate the targets for each layer, moving backward through the forward net:
        # N-1, N-2, ..., 2, 1
        # NOTE: Starting from N-1 since we already have the target for the last layer).
        with torch.no_grad():
            for i in reversed(range(1, N)):

                G = reordered_feedback_net[i]
                # G = feedback_net[-1 - i]
                
                assert targets[i - 1] is None  # Make sure we're not overwriting anything.
                # NOTE: Shifted the indices by 1 compared to @ernoult's eq.
                # t^{n-1} = s^{n-1} + G(t^{n}; B) - G(s^{n} ; B).
                targets[i - 1] = ys[i - 1] + G(targets[i]) - G(ys[i])

                # NOTE: Alternatively, just target propagation:
                # targets[i - 1] = G(targets[i])

        # NOTE: targets[0] is the targets for the output of the first layer, not for x.
        # Make sure that all targets have been computed and that they are fixed (don't require grad)
        assert all(target is not None and not target.requires_grad for target in targets)
        target_tensors = cast(List[Tensor], targets) # Rename just for typing purposes.

        # Calculate the losses for each layer:
        forward_loss_per_layer = [
            # 0.5*((ys[i] - targets[i])**2).view(ys[i].size(0), -1).sum(1).sum()
            # NOTE: Equivalent to the following.
            0.5 * F.mse_loss(ys[i], target_tensors[i], reduction="sum")  # type: ignore
            for i in range(0, N)
        ]
        assert len(ys) == len(targets) == len(forward_loss_per_layer) == len(self.forward_net) == N

        for i, layer_loss in enumerate(forward_loss_per_layer):
            self.log(f"{self.phase}/F_loss[{i}]", layer_loss)

        loss_tensor = torch.stack(forward_loss_per_layer, -1)
        return loss_tensor.sum()

    def configure_optimizers(self):
        # NOTE: We pass the learning rates in the same order as the feedback net:
        lrs_per_feedback_layer = self._get_learning_rate_per_feedback_layer(
            forward_ordering=False
        )
        feedback_optimizer = self.hp.b_optim.make_optimizer(
            self.backward_net, learning_rates_per_layer=lrs_per_feedback_layer
        )
        forward_optimizer = self.hp.f_optim.make_optimizer(self.forward_net)

        feedback_optim_config = {"optimizer": feedback_optimizer}
        forward_optim_config = {
            "optimizer": forward_optimizer,
            "lr_scheduler": CosineAnnealingLR(
                forward_optimizer, T_max=85, eta_min=1e-5
            ),
        }
        assert self.hp.f_optim.use_lr_scheduler
        assert self.hp.feedback_before_forward
        return [
            feedback_optim_config,
            forward_optim_config,
        ]
    
    def make_feedback_training_figure(self, angles: List[List[float]], distances: List[List[float]]):
        n_plots = len(angles)
        # Create figure with secondary y-axis
        from plotly.graph_objects import Figure
        fig: Figure = make_subplots(
            rows=2,
            cols=n_plots,
            x_title="# of feedback training iterations",
            column_titles=[f"layer {i}" for i in range(n_plots)],
            row_titles=["Angle (degrees)", "Distance"],
        )
        # Add traces
        for i in range(n_plots):
            layer_angles = angles[i]
            layer_distances = distances[i]
            x = np.arange(len(layer_angles))

            fig.add_trace(
                go.Scatter(x=x, y=layer_angles), row=1, col=i + 1,
            )
            fig.add_trace(
                go.Scatter(x=x, y=layer_distances),
                row=2,
                col=i + 1,
            )

        # Add figure title
        fig.update_layout(
            title_text="Distance and angle between F and G during feedback weight training",
            showlegend=False,
        )

        # Set x-axis title
        # fig.update_xaxes(title_text="# of feedback training iterations", row=2)
        # Set y-axes titles (only for the first column)
        fig.update_yaxes(title_text="Angle (degrees)", row=1, col=1)
        fig.update_yaxes(title_text="Distance", row=2, col=1)

        return fig
