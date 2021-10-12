import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import plotly.graph_objects as go
import torch
import wandb
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
from simple_parsing.helpers import choice, list_field
from simple_parsing.helpers.hparams import log_uniform, uniform
from target_prop.config import Config
from target_prop.feedback_loss import get_feedback_loss
from target_prop.layers import forward_all
from target_prop.metrics import compute_dist_angle
from target_prop.models.model import BaseModel
from target_prop.optimizer_config import OptimizerConfig
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)
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

        # batch size
        batch_size: int = log_uniform(16, 512, default=128, base=2, discrete=True)

        # Channels per conv layer.
        channels: List[int] = list_field(128, 128, 256, 256, 512)

        # Number of training steps for the feedback weights per batch. Can be a list of
        # integers, where each value represents the number of iterations for that layer.
        feedback_training_iterations: List[int] = list_field(20, 30, 35, 55, 20)

        # Max number of training epochs in total.
        max_epochs: int = 90

        # Hyper-parameters for the "backward" optimizer
        # BUG: The default values of the arguments don't reflect the values that are
        # passed to `mutable_field`
        b_optim: OptimizerConfig = OptimizerConfig(
            type="sgd", lr=[1e-4, 3.5e-4, 8e-3, 8e-3, 0.18], momentum=0.9
        )
        # The scale of the gaussian random variable in the feedback loss calculation.
        noise: List[float] = uniform(
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

        # Number of noise samples to use to get the feedback loss in a single iteration.
        # NOTE: The loss used for each update is the average of these losses.
        feedback_samples_per_iteration: int = uniform(1, 20, default=1)

        # Max number of epochs to train for without an improvement to the validation
        # accuracy before the training is stopped. When 0, no early stopping is used.
        early_stopping_patience: int = 0

        # Sets symmetric weight initialization
        init_symetric_weights: bool = False

        # jacobian: bool = False  # compute jacobians

        activation: Type[nn.Module] = choice(
            {"relu": nn.ReLU, "elu": nn.ELU,}, default="elu"
        )

        # Step interval for creating and logging plots.
        plot_every: int = 10

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

        # ----------- Optimize the feedback weights -------------
        # NOTE: feedback_loss here returns a dict for now, since I think that makes things easier to
        # inspect.
        feedback_training_outputs: Dict = self.feedback_loss(x, y)

        feedback_loss: Tensor = feedback_training_outputs["loss"]
        self.log(f"{phase}/B_loss", feedback_loss, prog_bar=phase == "train")
        # This is never a 'live' loss, since we do the optimization steps sequentially
        # inside `feedback_loss`.
        assert not feedback_loss.requires_grad

        # ----------- Optimize the forward weights -------------
        forward_loss = self.forward_loss(x, y)
        self.log(f"{phase}/F_loss", forward_loss, prog_bar=phase == "train")

        # During training, the forward loss will be a 'live' loss tensor, since we
        # gather the losses for each layer. Here we perform only one step.
        if forward_loss.requires_grad and not self.automatic_optimization:
            f_optimizer = self.forward_optimizer
            self.manual_backward(forward_loss)
            f_optimizer.step()
            f_optimizer.zero_grad()
            forward_loss = forward_loss.detach()
            lr_scheduler = self.lr_schedulers()
            if lr_scheduler:
                assert not isinstance(lr_scheduler, list)
                lr_scheduler.step()

        # Since here we do manual optimization, we just return a float. This tells PL that we've
        # already performed the optimization steps.
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
            ys: List[Tensor] = forward_all(
                self.forward_net, x, allow_grads_between_layers=False
            )

        # List of losses, distances, and angles for each layer (with multiple iterations per layer).
        layer_losses: List[List[Tensor]] = []
        layer_angles: List[List[float]] = []
        layer_distances: List[List[float]] = []

        # Layer-wise autoencoder training begins:
        # NOTE: Skipping the first layer
        for layer_index in range(1, n_layers):
            # Forward layer
            F_i = self.forward_net[layer_index]
            # Feedback layer
            G_i = reversed_backward_net[layer_index]
            x_i = ys[layer_index - 1]
            y_i = ys[layer_index]
            # Number of feedback training iterations to perform for this layer.
            iterations_i = iterations_per_layer[layer_index]
            if iterations_i and self.phase != "train":
                # NOTE: Only perform one iteration per layer when not training.
                iterations_i = 1
            # The scale of noise to use for thist layer.
            noise_scale_i = noise_scale_per_layer[layer_index]

            # Collect the distances and angles between the forward and backward weights during this
            # iteratin.
            iteration_angles: List[float] = []
            iteration_distances: List[float] = []
            iteration_losses: List[Tensor] = []

            # NOTE: When a layer isn't trainable (e.g. layer is a Reshape or nn.ELU), then
            # iterations_i will be 0, so the for loop below won't be run.

            # Train the current autoencoder:
            for iteration in range(iterations_i):
                assert noise_scale_i > 0, (
                    layer_index,
                    iterations_i,
                )
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

                # perform the optimization step for that layer when training.
                if self.phase == "train":
                    assert isinstance(loss, Tensor) and loss.requires_grad
                    feedback_optimizer.zero_grad()
                    self.manual_backward(loss)
                    feedback_optimizer.step()
                    loss = loss.detach()
                else:
                    assert isinstance(loss, Tensor) and not loss.requires_grad
                    # When not training that layer,
                    loss = torch.as_tensor(loss, device=y.device)

                logger.debug(
                    f"Layer {layer_index}, Iteration {iteration}, angle={angle}, "
                    f"distance={distance}"
                )
                iteration_losses.append(loss)
                iteration_angles.append(angle)
                iteration_distances.append(distance)

                # IDEA: If we log these values once per iteration, will the plots look nice?
                # self.log(f"{self.phase}/B_loss[{layer_index}]", loss)
                # self.log(f"{self.phase}/B_angle[{layer_index}]", angle)
                # self.log(f"{self.phase}/B_distance[{layer_index}]", distance)

            layer_losses.append(iteration_losses)
            layer_angles.append(iteration_angles)
            layer_distances.append(iteration_distances)

            # IDEA: Logging the number of iterations could be useful if we add some kind of early
            # stopping for the feedback training, since the number of iterations might vary.
            self.log(f"{self.phase}/B_total_loss[{layer_index}]", sum(iteration_losses))
            self.log(f"{self.phase}/B_iterations[{layer_index}]", iterations_i)
            # NOTE: Logging all the distances and angles for each layer, which isn't ideal!
            # What would be nicer would be to log this as a small, light-weight plot showing the
            # evolution of the distances / angles for each layer.
            # self.log(f"{self.phase}/B_angles[{layer_index}]", iteration_angles)
            # self.log(f"{self.phase}/B_distances[{layer_index}]", iteration_distances)

        if self.training and self.global_step % self.hp.plot_every == 0:
            fig = make_stacked_feedback_training_figure(
                all_values=[layer_angles, layer_distances, layer_losses],
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

        # NOTE: Need to return something.
        total_b_loss = sum(sum(iteration_losses) for iteration_losses in layer_losses)
        return {
            "loss": total_b_loss,
            "layer_losses": layer_losses,
            "layer_angles": layer_angles,
            "layer_distances": layer_distances,
        }

    def forward_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """ Get the loss used to train the forward net. 

        NOTE: Unlike `feedback_loss`, this actually returns the 'live' loss tensor.
        """
        # NOTE: Sanity check: Use standard backpropagation for training rather than TP.
        ## --------
        # return super().forward_loss(x=x, y=y)
        ## --------
        step_outputs: Dict[str, Union[Tensor, Any]] = {}
        ys: List[Tensor] = forward_all(
            self.forward_net, x, allow_grads_between_layers=False,
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
        targets: List[Optional[Tensor]] = [
            *(None for _ in range(N - 1)),
            last_layer_target,
        ]

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

                assert (
                    targets[i - 1] is None
                )  # Make sure we're not overwriting anything.
                # NOTE: Shifted the indices by 1 compared to @ernoult's eq.
                # t^{n-1} = s^{n-1} + G(t^{n}; B) - G(s^{n} ; B).
                targets[i - 1] = ys[i - 1] + G(targets[i]) - G(ys[i])

                # NOTE: Alternatively, just target propagation:
                # targets[i - 1] = G(targets[i])

        # NOTE: targets[0] is the targets for the output of the first layer, not for x.
        # Make sure that all targets have been computed and that they are fixed (don't require grad)
        assert all(
            target is not None and not target.requires_grad for target in targets
        )
        target_tensors = cast(List[Tensor], targets)  # Rename just for typing purposes.

        # Calculate the losses for each layer:
        forward_loss_per_layer = [
            0.5 * ((ys[i] - targets[i]) ** 2).view(ys[i].size(0), -1).sum(1).mean()
            # NOTE: Apprently NOT Equivalent to the following!
            # 0.5 * F.mse_loss(ys[i], target_tensors[i], reduction="mean")
            for i in range(0, N)
        ]
        assert (
            len(ys)
            == len(targets)
            == len(forward_loss_per_layer)
            == len(self.forward_net)
            == N
        )

        for i, layer_loss in enumerate(forward_loss_per_layer):
            self.log(f"{self.phase}/F_loss[{i}]", layer_loss)

        loss_tensor = torch.stack(forward_loss_per_layer, dim=0)
        # TODO: Use 'sum' or 'mean' as the reduction between layers?
        return loss_tensor.sum(dim=0)

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
        }
        if self.hp.scheduler:
            # Using the same LR scheduler as the original code:
            lr_scheduler = CosineAnnealingLR(forward_optimizer, T_max=85, eta_min=1e-5)
            forward_optim_config["lr_scheduler"] = lr_scheduler
        return [
            feedback_optim_config,
            forward_optim_config,
        ]


def make_stacked_feedback_training_figure(
    all_values: Sequence[List[List[Union[Tensor, np.ndarray, float]]]],
    row_titles: Sequence[str],
    title_text: str,
    layer_names: List[str] = None,
) -> Figure:
    """Creates a stacked plot that shows the evolution of different values during a step of
    feedback training.
    
    `all_values` should contain a sequence of list of lists. (a list of "metric_values").
    Each "metric_values" should contain the value of a metric, for each layer, for each iteration.
    `row_titles` should contain the name associated with each item in `all_values`.
    `title_text` is the name of the overall figure.
    """
    all_values = [
        [
            [v.cpu().numpy() if isinstance(v, Tensor) else v for v in layer_values]
            for layer_values in values
        ]
        for values in all_values
    ]

    n_layers = len(all_values[0])
    n_plots = len(all_values)
    layer_names = layer_names or [f"layer {i}" for i in range(n_layers)]
    assert len(row_titles) == n_plots
    # Each list needs to have the same number of items (i.e. same number of layers)
    assert all(len(values) == n_layers for values in all_values)

    fig: Figure = make_subplots(
        rows=n_plots,
        cols=n_layers,
        x_title="# of feedback training iterations",
        column_titles=layer_names,
        row_titles=[row_title for row_title in row_titles],
    )

    # Add traces
    for plot_id, values in enumerate(all_values):
        for layer_id in range(n_layers):
            layer_values = values[layer_id]
            x = np.arange(len(layer_values))
            trace = go.Scatter(x=x, y=layer_values)
            fig.add_trace(
                trace, row=plot_id + 1, col=layer_id + 1,
            )

    # Add figure title
    fig.update_layout(
        title_text=title_text, showlegend=False,
    )

    for i, row_title in enumerate(row_titles):
        # Set y-axes titles (only for the first column)
        fig.update_yaxes(title_text=row_title, row=i + 1, col=1)
        # Set a fixed range on the y axis for that row:
        if "angle" in row_title.lower():
            fig.update_yaxes(row=i+1, range=[0, 90], fixedrange=True)

    return fig
