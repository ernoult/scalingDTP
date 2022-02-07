import dataclasses
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from simple_parsing.helpers import choice, field, list_field
from simple_parsing.helpers.hparams.hparam import log_uniform, uniform
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torchmetrics.classification.accuracy import Accuracy

import wandb
from target_prop._weight_operations import init_symetric_weights
from target_prop.backward_layers import invert, mark_as_invertible
from target_prop.callbacks import CompareToBackpropCallback
from target_prop.config import Config
from target_prop.feedback_loss import get_feedback_loss
from target_prop.layers import forward_all
from target_prop.metrics import compute_dist_angle
from target_prop.models.model import Model
from target_prop.networks.network import Network
from target_prop.optimizer_config import OptimizerConfig
from target_prop.scheduler_config import CosineAnnealingLRConfig, LRSchedulerConfig
from target_prop.utils.utils import is_trainable

from .utils import make_stacked_feedback_training_figure

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class ForwardOptimizerConfig(OptimizerConfig):
    """Configuration of the optimizer for the forward weights.

    NOTE: Creating a distinct class for this is currently the only way to specify different priors
    for the forward optimizer and the feedback optimizer.
    """

    # Type of Optimizer to use.
    type: str = choice(*OptimizerConfig.available_optimizers.keys(), default="sgd")

    # Learning rate of the optimizer.
    lr: float = log_uniform(1e-4, 1e-1, default=0.05)

    # Weight decay coefficient.
    weight_decay: Optional[float] = 1e-4

    # Momentum term to pass to SGD.
    # NOTE: This value is only used with SGD, not with Adam.
    momentum: float = 0.9


@dataclass
class FeedbackOptimizerConfig(OptimizerConfig):
    """Configuration of the optimizer for the forward weights.

    NOTE: Creating a distinct class for this is currently the only way to specify different priors
    for the forward optimizer and the feedback optimizer.
    """

    # Type of Optimizer to use.
    type: str = choice(*OptimizerConfig.available_optimizers.keys(), default="sgd")

    # Learning rate of the optimizer.
    lr: List[float] = log_uniform(
        1e-4, 1e-1, default_factory=[1e-4, 3.5e-4, 8e-3, 8e-3, 0.18].copy, shape=5
    )

    # Learning rate of the optimizer.
    # NOTE: IF we want to tune a single learning rate:
    # lr: float = log_uniform(1e-4, 1e-1, default=1e-3)

    # Weight decay coefficient.
    weight_decay: Optional[float] = None

    # Momentum term to pass to SGD.
    # NOTE: This value is only used with SGD, not with Adam.
    momentum: float = 0.9


class DTP(LightningModule, Model):
    """ Differential Target Propagation algorithm, implemented as a LightningModule.

    This is (as far as I know) exactly equivalent with @ernoult's implementation.
    The default values for the hyper-parameters are equivalent to what they would be when
    running the following command:

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

    In other words, to reproduce @ernoult's results on Cifar-10, there is no need to change
    anything here or pass any custom values from the command-line.
    """

    @dataclass
    class HParams(Model.HParams):
        """Hyper-Parameters of the model.

        The number of noise samples to use per iteration is set by `feedback_samples_per_iteration`.

        NOTE: By increasing the value of `feedback_samples_per_iteration` and setting the value of
        `feedback_training_iterations` to 1 for all layers, we could get something close to a
        "parallel" version of DTP, however the feedback layers still need to be updated in sequence.
        """

        # Arguments to be passed to the LR scheduler.
        lr_scheduler: LRSchedulerConfig = field(default_factory=CosineAnnealingLRConfig)
        # Use of a learning rate scheduler for the forward weights.
        use_scheduler: bool = True

        # batch size
        batch_size: int = log_uniform(16, 512, default=128, base=2, discrete=True)

        # Number of training steps for the feedback weights per batch. Can be a list of
        # integers, where each value represents the number of iterations for that layer.
        # NOTE: Not tuning these values:
        feedback_training_iterations: List[int] = list_field(20, 30, 35, 55, 20)
        # NOTE: tuning a single value for all layers:
        # feedback_training_iterations: int = uniform(1, 60, default=20, discrete=True)
        # NOTE: IF we want to tune each value independantly:
        # feedback_training_iterations: List[int] = uniform(
        #     1, 60, shape=5, default_factory=[20, 30, 35, 55, 20].copy, discrete=True
        # )

        # Max number of training epochs in total.
        max_epochs: int = 90

        # Hyper-parameters for the optimizer of the feedback weights (backward net).
        b_optim: FeedbackOptimizerConfig = FeedbackOptimizerConfig(
            type="sgd", lr=[1e-4, 3.5e-4, 8e-3, 8e-3, 0.18], momentum=0.9
        )

        # The scale of the gaussian random variable in the feedback loss calculation.
        # NOTE: Not tuning this parameter:
        noise: List[float] = list_field(0.4, 0.4, 0.2, 0.2, 0.08)
        # NOTE: tuning a value per layer:
        # noise: List[float] = uniform(  # type: ignore
        #     0.001, 0.5, default_factory=[0.4, 0.4, 0.2, 0.2, 0.08].copy, shape=5
        # )
        # NOTE: tuning a single value for all layers:
        # noise: float = uniform(0.001, 0.5, default=0.2)

        # Hyper-parameters for the forward optimizer
        f_optim: ForwardOptimizerConfig = ForwardOptimizerConfig(
            type="sgd", lr=0.05, weight_decay=1e-4, momentum=0.9
        )
        # nudging parameter: Used when calculating the first target.
        # beta: float = 0.7  # NOTE: not tuning this value
        beta: float = uniform(0.01, 1.0, default=0.7)  # Adding it to HPO space

        # Number of noise samples to use to get the feedback loss in a single iteration.
        # NOTE: The loss used for each update is the average of these losses.
        feedback_samples_per_iteration: int = 1  # Not tuning the value
        # feedback_samples_per_iteration: int = uniform(1, 20, default=1)  # tuning the value

        # Max number of epochs to train for without an improvement to the validation
        # accuracy before the training is stopped. When 0, no early stopping is used.
        early_stopping_patience: int = 0

        # Sets symmetric weight initialization. Useful for debugging.
        init_symetric_weights: bool = False

        # TODO: Add a Callback class to compute and plot jacobians, if that's interesting.
        # jacobian: bool = False  # compute jacobians

        # Step interval for creating and logging plots.
        plot_every: int = 1000

        def __post_init__(self):
            super().__post_init__()
            for field in dataclasses.fields(self):
                value = getattr(self, field.name)
                from simple_parsing.utils import is_list

                if is_list(field.type) and isinstance(value, np.ndarray):
                    # Convert to a list.
                    setattr(self, field.name, value.tolist())

    def __init__(
        self,
        datamodule: LightningDataModule,
        network: Network,
        hparams: "DTP.HParams",
        config: Config,
        network_hparams: HyperParameters = None,
    ):
        super().__init__()
        self.hp: DTP.HParams = hparams
        self.net_hp = network_hparams or network.hparams
        self.config = config
        if self.config.seed is not None:
            # NOTE: This is currently being done twice: Once in main_pl and once again here.
            seed_everything(seed=self.config.seed, workers=True)

        # NOTE: Setting this property allows PL to infer the shapes and number of params.
        self.example_input_array = torch.rand(  # type: ignore
            [datamodule.batch_size, *datamodule.dims], device=config.device
        )

        # Create the forward and backward nets.
        self.forward_net = network.to(config.device)
        self.backward_net = self.create_backward_net().to(config.device)

        if self.hp.init_symetric_weights:
            logger.info(f"Initializing the backward net with symetric weights.")
            init_symetric_weights(self.forward_net, self.backward_net)

        # The number of iterations to perform for each of the layers in `self.backward_net`.
        self.feedback_iterations = self._align_values_with_backward_net(
            self.hp.feedback_training_iterations, default=0, inputs_are_forward_ordered=True,
        )
        # The noise scale for each feedback layer.
        self.feedback_noise_scales = self._align_values_with_backward_net(
            self.hp.noise, default=0.0, inputs_are_forward_ordered=True,
        )
        # The learning rate for each feedback layer.
        lrs_per_layer = self.hp.b_optim.lr
        self.feedback_lrs = self._align_values_with_backward_net(
            lrs_per_layer, default=0.0, inputs_are_forward_ordered=True
        )

        if self.config.debug:
            print(f"Forward net: ")
            print(self.forward_net)
            print(f"Feedback net:")
            print(self.backward_net)

            N = len(self.backward_net)
            for i, (layer, lr, noise, iterations) in list(
                enumerate(
                    zip(
                        self.backward_net,
                        self.feedback_lrs,
                        self.feedback_noise_scales,
                        self.feedback_iterations,
                    )
                )
            ):
                logger.info(
                    f"self.backward_net[{i}]: (G[{N-i-1}]): Type {type(layer).__name__}, LR: {lr}, "
                    f"noise: {noise}, iterations: {iterations}"
                )
                if i == N - 1:
                    # The last layer of the backward_net (the layer closest to the input) is not
                    # currently being trained, so we expect it to not have these parameters.
                    assert lr == 0
                    assert noise == 0
                    assert iterations == 0
                    continue
                if any(p.requires_grad for p in layer.parameters()):
                    # For any of the trainable layers in the backward net (except the last one), we
                    # expect to have positive values:
                    assert lr > 0
                    assert noise > 0
                    assert iterations > 0
                else:
                    # Non-Trainable layers (e.g. Reshape) are not trained.
                    assert lr == 0
                    assert noise == 0
                    assert iterations == 0
        # Metrics:
        self.accuracy = Accuracy()
        self.top5_accuracy = Accuracy(top_k=5)
        self.save_hyperparameters(
            {
                "hp": self.hp.to_dict(),
                "config": self.config.to_dict(),
                "model_type": type(self).__name__,
                "net_hp": self.net_hp.to_dict(),
                "net_type": type(self.forward_net).__name__,
            }
        )

        # NOTE: These properties below are in the backward ordering, while those in the hparams are
        # in the forward order.

        self.trainer: Trainer  # type: ignore
        # Can't do automatic optimization here, since we do multiple sequential updates
        # per batch.
        self.automatic_optimization = False
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        print("Hyper-Parameters:")
        print(self.hp.dumps_json(indent="\t"))
        # TODO: Could use a list of metrics from torchmetrics instead of just accuracy:
        # self.supervised_metrics: List[Metrics]
        self._feedback_optimizers: Optional[List[Optional[Optimizer]]] = None
        self._forward_optimizer: Optional[Optimizer] = None

    def create_backward_net(self) -> nn.Sequential:
        # Pass an example input through the forward net so that we know the input/output shapes for
        # each layer. This makes it easier to then create the feedback (a.k.a backward) net.
        mark_as_invertible(self.forward_net)
        example_out: Tensor = self.forward_net(self.example_input_array)

        assert example_out.requires_grad
        # Get the "pseudo-inverse" of the forward network:
        backward_net: nn.Sequential = invert(self.forward_net).to(self.config.device)  # type: ignore

        # Pass the output of the forward net for the `example_input_array` through the
        # backward net, to check that the backward net is indeed able to recover the
        # inputs (at least in terms of their shape for now).
        example_in_hat: Tensor = backward_net(example_out)
        assert example_in_hat.requires_grad
        assert example_in_hat.shape == self.example_input_array.shape
        assert example_in_hat.dtype == self.example_input_array.dtype

        return backward_net

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore
        # Dummy forward pass, not used in practice. We just implement it so that PL can
        # display the input/output shapes of our networks.
        y = self.forward_net(input)
        r = self.backward_net(y)
        return y, r

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int,) -> float:  # type: ignore
        result = self.shared_step(batch, batch_idx=batch_idx, phase="train")
        if not self.automatic_optimization:
            return None
        return result

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int,
    ) -> float:  # type: ignore
        return self.shared_step(batch, batch_idx=batch_idx, phase="val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> float:  # type: ignore
        return self.shared_step(batch, batch_idx=batch_idx, phase="test")

    def shared_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, phase: str,
    ):
        """Main step, used by the `[training/valid/test]_step` methods."""
        x, y = batch

        # ----------- Optimize the feedback weights -------------
        # NOTE: feedback_loss here returns a dict for now, since I think that makes things easier to
        # inspect.
        feedback_training_outputs: Dict = self.feedback_loss(x, y, phase=phase)

        feedback_loss: Tensor = feedback_training_outputs["loss"]
        avg_feedback_loss: Tensor = feedback_training_outputs["avg_loss"]
        if self.trainer is not None:
            self.log(f"{phase}/B_loss", feedback_loss)
            self.log(f"{phase}/B_avg_loss", avg_feedback_loss, prog_bar=phase == "train")
        # This is never a 'live' loss, since we do the optimization steps sequentially
        # inside `feedback_loss`.
        assert not feedback_loss.requires_grad

        # ----------- Optimize the forward weights -------------
        forward_training_outputs: Dict = self.forward_loss(x, y, phase=phase)
        forward_loss: Tensor = forward_training_outputs["loss"]

        # During training, the forward loss will be a 'live' loss tensor, since we
        # gather the losses for each layer. Here we perform only one step.
        assert forward_loss.requires_grad == (phase == "train")
        # NOTE: If this is getting called from the `ParallelDTP`, then `self.automatic_optimization`
        # will be `True`, and we let PL do the update.
        if forward_loss.requires_grad and not self.automatic_optimization:
            forward_optimizer = self.forward_optimizer
            forward_optimizer.zero_grad()
            if self.trainer:
                self.manual_backward(forward_loss)
            else:
                # Unit testing.
                forward_loss.backward()
            forward_optimizer.step()
            self.log(f"F_lr", forward_optimizer.param_groups[0]["lr"])
            forward_loss = forward_loss.detach()

        last_layer_loss: Tensor = forward_training_outputs["layer_losses"][-1].detach()
        if self.trainer is not None:
            self.log(f"{phase}/F_loss", forward_loss)
            self.log(f"{phase}/Loss", last_layer_loss, prog_bar=phase == "train")

        # Since here we do manual optimization, we just return a float. This tells PL that we've
        # already performed the optimization steps, if needed.
        return float(forward_loss + feedback_loss)

    def feedback_loss(self, x: Tensor, y: Tensor, phase: str) -> Dict[str, Any]:

        n_layers = len(self.backward_net)
        # Reverse the backward net, just for ease of readability.
        reversed_backward_net = self.backward_net[::-1]
        reversed_feedback_optimizers = self.feedback_optimizers[::-1]
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
            ys: List[Tensor] = forward_all(self.forward_net, x, allow_grads_between_layers=False)

        # List of losses, distances, and angles for each layer (with multiple iterations per layer).
        layer_losses: List[List[Tensor]] = []
        layer_angles: List[List[float]] = []
        layer_distances: List[List[float]] = []
        layer_avg_losses: List[List[float]] = []

        # Layer-wise autoencoder training begins:
        # NOTE: Skipping the first layer
        for layer_index in range(1, n_layers):
            # Forward layer
            F_i = self.forward_net[layer_index]
            # Feedback layer
            G_i = reversed_backward_net[layer_index]
            layer_optimizer = reversed_feedback_optimizers[layer_index]
            assert (layer_optimizer is not None) == (is_trainable(G_i))

            x_i = ys[layer_index - 1]
            y_i = ys[layer_index]

            # Number of feedback training iterations to perform for this layer.
            iterations_i = iterations_per_layer[layer_index]
            if iterations_i and not self.training:
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
                loss = self.layer_feedback_loss(
                    feedback_layer=G_i,
                    forward_layer=F_i,
                    input=x_i,
                    output=y_i,
                    noise_scale=noise_scale_i,
                    noise_samples=self.hp.feedback_samples_per_iteration,
                )

                # Compute the angle and distance for debugging the training of the
                # feedback weights:
                with torch.no_grad():
                    metrics = compute_dist_angle(F_i, G_i)
                    if isinstance(metrics, dict):
                        # NOTE: When a block has more than one trainable layer, we only report the
                        # first non-zero value for now.
                        # TODO: Fix this later.
                        distance, angle = 0, 0
                        for k, v in metrics.items():
                            if v != (0, 0):
                                distance, angle = v
                                break
                    else:
                        distance, angle = metrics

                # perform the optimization step for that layer when training.
                if self.training and layer_optimizer:
                    assert isinstance(loss, Tensor) and loss.requires_grad
                    layer_optimizer.zero_grad()
                    # self.manual_backward(loss) won't work if self.trainer is None
                    # self.trainer is None in legacy unit tests
                    self.manual_backward(loss) if self.trainer is not None else loss.backward()
                    layer_optimizer.step()
                    if self.trainer is not None:
                        self.log(f"B_lr{layer_index}", layer_optimizer.param_groups[0]["lr"])
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
            total_iter_loss = sum(iteration_losses)
            if iterations_i > 0:
                avg_iter_loss = total_iter_loss / iterations_i
                layer_avg_losses.append(avg_iter_loss)

            if self.trainer is not None:
                self.log(f"{phase}/B_total_loss[{layer_index}]", total_iter_loss)
                if iterations_i > 0:
                    self.log(f"{phase}/B_avg_loss[{layer_index}]", avg_iter_loss)
                self.log(f"{phase}/B_iterations[{layer_index}]", float(iterations_i))
                self.log(f"{phase}/B_angle[{layer_index}]", iteration_angles[-1])
                self.log(f"{phase}/B_distance[{layer_index}]", iteration_distances[-1])

        if (
            self.training
            and self.global_step % self.hp.plot_every == 0
            and self.trainer is not None
        ):
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
                fig.write_html(str(save_path.with_suffix(".html")), include_plotlyjs="cdn")

            if wandb.run:
                wandb.log({"feedback_training": fig})

        # NOTE: Need to return something.
        total_b_loss = sum(sum(iteration_losses) for iteration_losses in layer_losses)
        avg_b_loss = sum(layer_avg_losses) / len(layer_avg_losses)
        return {
            "loss": total_b_loss,
            "avg_loss": avg_b_loss,
            "layer_losses": layer_losses,
            "layer_angles": layer_angles,
            "layer_distances": layer_distances,
        }

    def forward_loss(self, x: Tensor, y: Tensor, phase: str) -> Dict[str, Union[Tensor, Any]]:
        """Get the loss used to train the forward net.

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

            # self.trainer is None in some unit tests which only use PL module
            if self.trainer is not None:
                probs = torch.softmax(logits, -1)
                self.log(f"{phase}/accuracy", self.accuracy(probs, labels), prog_bar=True)
                self.log(f"{phase}/top5_accuracy", self.top5_accuracy(probs, labels))

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

        if self.trainer is not None:
            self.log(f"{phase}/delta.norm()", delta.norm())
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

                assert targets[i - 1] is None  # Make sure we're not overwriting anything.
                # NOTE: Shifted the indices by 1 compared to @ernoult's eq.
                # t^{n-1} = s^{n-1} + G(t^{n}; B) - G(s^{n} ; B).
                # targets[i - 1] = ys[i - 1] + G(targets[i]) - G(ys[i])
                prev_target = targets[i]
                assert prev_target is not None
                targets[i - 1] = self.compute_target(i=i, G=G, hs=ys, prev_target=prev_target)

                # NOTE: Alternatively, just target propagation:
                # targets[i - 1] = G(targets[i])

        # NOTE: targets[0] is the targets for the output of the first layer, not for x.
        # Make sure that all targets have been computed and that they are fixed (don't require grad)
        assert all(target is not None and not target.requires_grad for target in targets)
        target_tensors = cast(List[Tensor], targets)  # Rename just for typing purposes.

        # Calculate the losses for each layer:
        forward_loss_per_layer = []
        for i in range(0, N):
            if (
                ys[i].requires_grad or phase != "train"
            ):  # Removes duplicate reshape layer loss from total loss estimate
                layer_loss = 0.5 * ((ys[i] - targets[i]) ** 2).view(ys[i].size(0), -1).sum(1).mean()
                # NOTE: Apprently NOT Equivalent to the following!
                # 0.5 * F.mse_loss(ys[i], target_tensors[i], reduction="mean")
                forward_loss_per_layer.append(layer_loss)

        # self.trainer is None in some unit tests which only use PL module
        if self.trainer is not None:
            for i, layer_loss in enumerate(forward_loss_per_layer):
                self.log(f"{phase}/F_loss[{i}]", layer_loss)

        loss_tensor = torch.stack(forward_loss_per_layer, dim=0)
        # TODO: Use 'sum' or 'mean' as the reduction between layers?
        forward_loss = loss_tensor.sum(dim=0)
        return {
            "loss": forward_loss,
            "layer_losses": forward_loss_per_layer,
        }

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
        return hs[i - 1] + G(prev_target) - G(hs[i])

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
        return get_feedback_loss(
            feedback_layer=feedback_layer,
            forward_layer=forward_layer,
            input=input,
            output=output,
            noise_scale=noise_scale,
            noise_samples=noise_samples,
        )

    def on_train_epoch_end(self):
        lr_scheduler = self.lr_schedulers()
        if lr_scheduler and not self.automatic_optimization:
            assert not isinstance(lr_scheduler, list)
            lr_scheduler.step()

    def configure_optimizers(self) -> List[Dict]:
        """Returns the configurations for the optimizers.

        The entries are ordered like: [G_N, G_N-1, ..., G_2, G_1, F]

        The first items in the list are the configs for each trainable feedback layer.
        There is no entry for the first feedback layer (G_0)
        The last item in the list is for the forward optimizer.
        """
        # NOTE: We pass the learning rates in the same order as the feedback net:
        # NOTE: Here when we have one optimizer per feedback layer, we will put the forward optim at
        # the last index.
        configs: List[Dict] = []
        # NOTE: The last feedback layer (G_0) isn't trained, so it doesn't have an optimizer.
        assert len(self.backward_net) == len(self.feedback_lrs)
        for i, (feedback_layer, lr) in enumerate(zip(self.backward_net, self.feedback_lrs)):
            if i == (len(self.backward_net) - 1) or not is_trainable(feedback_layer):
                # NOTE: No learning rate for the first feedback layer atm, although we very well
                # could train it, it wouldn't change anything about the forward weights.
                # Non-trainable layers also don't have an optimizer.
                assert lr == 0.0, (i, lr, self.feedback_lrs, type(feedback_layer))
            else:
                assert lr != 0.0
                feedback_layer_optimizer = self.hp.b_optim.make_optimizer(feedback_layer, lrs=[lr])
                feedback_layer_optim_config = {"optimizer": feedback_layer_optimizer}
                configs.append(feedback_layer_optim_config)

        ## Forward optimizer:
        forward_optimizer = self.hp.f_optim.make_optimizer(self.forward_net)
        forward_optim_config: Dict[str, Any] = {
            "optimizer": forward_optimizer,
        }
        if self.hp.use_scheduler:
            # Using the same LR scheduler as the original code:
            lr_scheduler = self.hp.lr_scheduler.make_scheduler(forward_optimizer)
            lr_scheduler_config: Dict[str, Any] = {"scheduler": lr_scheduler}
            if self.automatic_optimization:
                lr_scheduler_config.update(
                    {
                        "interval": self.hp.lr_scheduler.interval,
                        "frequency": self.hp.lr_scheduler.frequency,
                    }
                )
            forward_optim_config["lr_scheduler"] = lr_scheduler_config
        configs.append(forward_optim_config)
        return configs

    @property
    def feedback_optimizers(self) -> List[Optional[Optimizer]]:
        """Returns the list of optimizers, one per layer of the feedback/backward net:
        [G_N, G_N-1, ..., G_2, G_1, None]

        For the "first" feedback layer (G_0), as well as all layers without trainable weights, the
        entry will be `None`.
        """
        # NOTE: self.trainer is None during unit testing
        if self.trainer is None:
            return self._feedback_optimizers
        _feedback_optimizers = []
        optimizers: List[Optimizer] = list(self.optimizers())
        for i, layer in enumerate(self.backward_net):
            layer_optimizer: Optional[Optimizer] = None
            # NOTE: First feedback layer is not currently trained (although it could eventually if
            # we wanted an end-to-end invertible network).
            if i != (len(self.backward_net) - 1) and is_trainable(layer):
                layer_optimizer = optimizers.pop(0)
            _feedback_optimizers.append(layer_optimizer)
        assert len(optimizers) == 1  # Only one left: The optimizer for the forward net.
        assert optimizers[-1] is self.forward_optimizer
        return _feedback_optimizers

    @property
    def forward_optimizer(self) -> Optimizer:
        """Returns The optimizer of the forward net."""
        # NOTE: self.trainer is None during unit testing
        if self.trainer is None:
            return self._forward_optimizer
        return self.optimizers()[-1]

    def _align_values_with_backward_net(
        self, values: List[T], default: T, inputs_are_forward_ordered: bool = False
    ) -> List[T]:
        """Aligns the values in `values` so that they are aligned with the trainable
        layers in the backward net.
        The last layer of the backward net (G_0) is also never trained.

        This assumes that `forward_ordering` is True, then `values` are forward-ordered.
        Otherwise, assumes that the input is given in the *backward* order Gn, Gn-1, ..., G0.

        NOTE: Outputs are *always* aligned with `self.backward_net` ([Gn, ..., G0]).

        Example: Using the default learning rate values for cifar10 as an example:

            `self.forward_net`: (conv, conv, conv, conv, reshape, linear)
            `self.backward_net`:   (linear, reshape, conv, conv, conv, conv)

            forward-aligned values: [1e-4, 3.5e-4, 8e-3, 8e-3, 0.18]

            `values` (backward-aligned): [0.18, 8e-3, 8e-3, 3.5e-4, 1e-4]  (note: backward order)


            `default`: 0

            Output:  [0.18, 0 (default), 8e-3, 8e-3, 3.5e-4, 1e-4, 0 (never trained)]

        Parameters
        ----------
        values : List[T]
            List of values for each trainable layer.
        default : T
            The value to set for non-trainable layers in the feedback network.
        inputs_are_forward_ordered : bool, optional
            Wether the inputs are given in a forward-aligned order or not. When they aren't, they
            are aligned with the trainable layers in the feedback network.
            Defaults to False.

        Returns
        -------
        List[T]
            List of values, one per layer in the backward net, with the non-trainable layers
            assigned the value of `default`.
        """
        n_layers_that_need_a_value = sum(map(is_trainable, self.backward_net))
        # Don't count the last layer of the backward net (i.e. G_0), since we don't
        # train it.
        n_layers_that_need_a_value -= 1

        if isinstance(values, (int, float)):
            values = [values for _ in range(n_layers_that_need_a_value)]  # type: ignore

        if len(values) > n_layers_that_need_a_value:
            truncated_values = values[:n_layers_that_need_a_value]
            # TODO: Eventually, either create a parameterized default value for the HParams for each
            # dataset, or switch to Hydra, if it's easy to use and solves this neatly.
            warnings.warn(
                RuntimeWarning(
                    f"There are {n_layers_that_need_a_value} layers that need a value, but we were "
                    f"given {len(values)} values! (values={values})\n"
                    f"Either pass a single value for all layers, or a value for each layer.\n"
                    f"WARNING: This will only use the first {n_layers_that_need_a_value} values: "
                    f"{truncated_values}"
                )
            )
            values = truncated_values
        elif len(values) < n_layers_that_need_a_value:
            # TODO: Same as above.
            last_value = values[-1]
            warnings.warn(
                RuntimeWarning(
                    f"There are {n_layers_that_need_a_value} layers that need a value, but we were "
                    f"only provided {len(values)} values! (values={values})\n"
                    f"Either pass a single value for all layers, or a value for each layer.\n"
                    f"WARNING: This will duplicate the first value ({last_value}) for all remaining "
                    f"layers."
                )
            )
            values = list(values) + [
                last_value for _ in range(n_layers_that_need_a_value - len(values))
            ]
            logger.warn(f"New values: {values}")
            assert len(values) == n_layers_that_need_a_value

        backward_ordered_input = list(reversed(values)) if inputs_are_forward_ordered else values
        # TODO: Once above TODO is addressed, re-instate this policy.
        # if len(values) != n_layers_that_need_a_value:
        #     raise ValueError(
        #         f"There are {n_layers_that_need_a_value} layers that need a value, but we were "
        #         f"given {len(values)} values! (values={values})\n "
        #     )

        values_left = backward_ordered_input.copy()
        values_per_layer: List[T] = []
        for layer in self.backward_net:
            if is_trainable(layer) and values_left:
                values_per_layer.append(values_left.pop(0))
            else:
                values_per_layer.append(default)
        assert values_per_layer[-1] == default

        backward_ordered_output = values_per_layer
        return backward_ordered_output

    def configure_callbacks(self):
        return [CompareToBackpropCallback()]
