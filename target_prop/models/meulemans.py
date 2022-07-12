from __future__ import annotations

import argparse
import dataclasses
from ast import literal_eval
from dataclasses import dataclass
from typing import Iterator, Optional, TypeVar

import numpy as np
import torch
from conditional_fields import conditional_field, set_conditionals
from pl_bolts.datamodules import CIFAR10DataModule
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import Serializable
from torch import Tensor, nn
from torch.nn import functional as F

import meulemans_dtp.main
from meulemans_dtp.final_configs.cifar10_DDTPConv import config as _config
from meulemans_dtp.lib import utils
from meulemans_dtp.lib.conv_layers import DDTPConvLayer
from meulemans_dtp.lib.conv_network import DDTPConvNetworkCIFAR
from meulemans_dtp.lib.direct_feedback_layers import DDTPMLPLayer
from meulemans_dtp.main import (
    AdamOptions,
    Args,
    DatasetOptions,
    LoggingOptions,
    MiscOptions,
    NetworkOptions,
    TrainOptions,
)
from target_prop.config import MiscConfig
from target_prop.models.model import Model, StepOutputDict
from target_prop.networks import Network

S = TypeVar("S", bound=Serializable)


def get_default_cifar10_args() -> Args:
    """Returns a typed version of the `args` Namespace that is used throughout the meulemans DTP
    codebase.

    This returns the *postprocessed* object.
    """
    raw_args = _get_default_raw_cifar10_args()
    # Note: This intenally calls `postprocess_args`, so it's exactly equivalent to using the raw
    # namespace.
    return meulemans_dtp.main.Args(**vars(raw_args))


def _get_default_raw_cifar10_args() -> argparse.Namespace:
    """Returns the `args` object that is used throughout the meulemans DTP codebase.

    This returns the *raw* object, before the values are post-processed.
    """
    parser = meulemans_dtp.main.add_command_line_args()
    cifar10_config = _config.copy()
    cifar10_config["epsilon"] = literal_eval(cifar10_config["epsilon"])
    for key, value in _config.items():
        if isinstance(value, np.ndarray):
            cifar10_config[key] = value.tolist()
    parser.set_defaults(**cifar10_config)
    args = parser.parse_args("")
    return args


# def _replace_ndarrays_with_lists(args: Args, inplace: bool = False):
#     cleaned_up_config = args if inplace else copy.deepcopy(args)
#     for key, value in vars(args).items():
#         if isinstance(value, np.ndarray):
#             setattr(cleaned_up_config, key, value.tolist())
#     return cleaned_up_config


def load_from_args(cls: type[S], raw_args: argparse.Namespace) -> S:
    # NOTE: The `args` should *NOT* already be postprocessed.
    kwargs = {}
    for field in dataclasses.fields(cls):
        name = field.name
        value = getattr(raw_args, name)
        kwargs[name] = value
        assert not isinstance(value, np.ndarray)
    return cls(**kwargs)


# Their default values for this 'args' object.

_CIFAR10_RAW_ARGS: argparse.Namespace = _get_default_raw_cifar10_args()
_CIFAR10_ARGS: Args = get_default_cifar10_args()


class MeulemansNetwork(DDTPConvNetworkCIFAR, Network):
    @dataclass
    class HParams(Network.HParams):
        """Hyper-parameters of the network used by the Meulamans model below.

        TODO: These are the values for CIFAR10. (DDTPConvNetworkCIFAR)
        TODO: Check how these values would be set from the Args object (the config dict) and set
        those as the defaults, instead of the values here.
        """

        bias: bool = True
        hidden_activation: str = "tanh"  # Default was tanh, set to 'elu' to match ours.
        feedback_activation: str = "linear"
        initialization: str = "xavier_normal"
        sigma: float = 0.1
        forward_requires_grad: bool = False
        plots: Optional[bool] = None
        nb_feedback_iterations: tuple[int, int, int, int] = (10, 20, 55, 20)

    def __init__(
        self, in_channels: int, n_classes: int, hparams: MeulemansNetwork.HParams | None = None
    ):
        assert in_channels == 3
        assert n_classes == 10
        if not hparams:
            hparams = self.HParams()
        self.hparams: MeulemansNetwork.HParams = hparams
        super().__init__(
            bias=hparams.bias,
            hidden_activation=hparams.hidden_activation,
            feedback_activation=hparams.feedback_activation,
            initialization=hparams.initialization,
            sigma=hparams.sigma,
            plots=hparams.plots,
            forward_requires_grad=hparams.forward_requires_grad,
            nb_feedback_iterations=hparams.nb_feedback_iterations,
        )

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._layers)

    def __len__(self) -> int:
        return len(self._layers)

    def compute_feedback_gradients(self, layer_index: int):
        return super().compute_feedback_gradients(layer_index=layer_index)
        # TODO: Adapt this so we just return a loss for that layer, if possible.

    def dummy_forward(self, h: Tensor, i: int) -> Tensor:
        """Propagate the activation of layer i forward through the network
        without saving the activations"""
        for layer in self.layers[i + 1 :]:
            if isinstance(layer, DDTPMLPLayer) and h.ndim == 4:
                h = h.flatten(1)
            h = layer.dummy_forward(h)
        return h


class Meulemans(Model[MeulemansNetwork]):
    @dataclass
    class HParams(Model.HParams, FlattenedAccess):
        """Arguments for our adapted version of the Meuleman's model.

        NOTE: Each field here corresponds to a group of arguments from the meulemans repo main
        script.
        Originally, the `args` object was a namespace, but now it is a dataclass, where we grouped
        the arguments from each group into a distinct dataclass (DatasetOptions, TrainOptions,
        ...).

        The same exact arguments are added here, but we only use a fraction of them, since we
        control the dataset / etc differently in our repo (via the DatasetConfig object).
        """

        dataset: meulemans_dtp.main.DatasetOptions = load_from_args(
            DatasetOptions, _CIFAR10_RAW_ARGS
        )
        training: meulemans_dtp.main.TrainOptions = load_from_args(TrainOptions, _CIFAR10_RAW_ARGS)
        adam: meulemans_dtp.main.AdamOptions = load_from_args(AdamOptions, _CIFAR10_RAW_ARGS)

        network: meulemans_dtp.main.NetworkOptions = load_from_args(
            NetworkOptions, _CIFAR10_RAW_ARGS
        )
        """ Options used to create the network.

        NOTE: There's some duplication here. They use the single 'Args' object will all arguments
        to create the network, not just the "network" group (which we grouped up into the
        `NetworkOptions`).

        TODO: It might be better to move the relevant options to MeulemansNetwork.HParams, so it's
        more consistent: The `Meulemans` model would take in a `MeulemansNetwork` as an argument,
        and that network would be created using its own hparams.
        """

        misc: meulemans_dtp.main.MiscOptions = load_from_args(MiscOptions, _CIFAR10_RAW_ARGS)
        """Other miscelaneous options (cuda, etc). """

        logging: meulemans_dtp.main.LoggingOptions = load_from_args(
            LoggingOptions, _CIFAR10_RAW_ARGS
        )
        """ Logging options. """

        # NOTE: Fields below are just initialized based on other values.

        save_angle: bool = conditional_field(
            lambda logging: (
                logging.save_GN_activations_angle
                or logging.save_BP_activations_angle
                or logging.save_BP_angle
                or logging.save_GN_angle
                or logging.save_GNT_angle
            )
        )
        classification: bool = conditional_field(
            lambda dataset: dataset.dataset in ["mnist", "fashion_mnist", "cifar10"]
        )
        regression: bool = conditional_field(
            lambda dataset: dataset.dataset in ["student_teacher", "boston"]
        )
        diff_rec_loss: bool = conditional_field(lambda network: network.network_type in ["DTPDR"])
        direct_fb: bool = conditional_field(
            lambda network: network.network_type
            in [
                "DKDTP",
                "DKDTP2",
                "DMLPDTP",
                "DMLPDTP2",
                "DDTPControl",
                "DDTPConv",
                "DDTPConvCIFAR",
                "DDTPConvControlCIFAR",
            ]
        )

        def __post_init__(self):
            """Do the postprocessing of the arguments. This is largely copied from their repo.

            TODO: Remove the parts of this that we don't need, once we're confident that we've
            replicated their implementation correctly (once the repro tests pass).
            """
            # Initialize the conditional fields.
            set_conditionals(self)

            # NOTE: The following is copied over and adapted from the `postprocess_args` fn.

            ### Create summary log writer
            self.logging.setup_out_dir()
            if not self.classification and not self.regression:
                raise ValueError(f"Dataset is not supported.")

            # initializing command line arguments if None
            if self.network.output_activation is None:
                self.network.output_activation = "softmax" if self.classification else "linear"
            if self.network.fb_activation is None:
                self.network.fb_activation = self.network.hidden_activation
            if self.network.hidden_fb_activation is None:
                self.network.hidden_fb_activation = self.network.hidden_activation

            if self.training.optimizer_fb is None:
                self.training.optimizer_fb = self.training.optimizer
            if isinstance(self.network.size_hidden, str):
                _hdim = utils.process_hdim(self.network.size_hidden)
                assert isinstance(_hdim, int)
                self.network.size_hidden = _hdim
            if self.network.size_mlp_fb == "None":
                self.network.size_mlp_fb = None
            elif isinstance(self.network.size_mlp_fb, str):
                _size_mlp_fb = utils.process_hdim_fb(self.network.size_mlp_fb)
                assert isinstance(_size_mlp_fb, int)
                self.network.size_mlp_fb = _size_mlp_fb

            # Manipulating command line arguments if asked
            # TODO: They were manipulating these if they were strings, which isn't necessary anymore,
            # however, it might be important to be able to load from their configs in the long run.
            self.misc.random_seed = int(self.misc.random_seed)

            if self.training.normalize_lr:
                self.training.lr = (np.array(self.lr) / self.training.target_stepsize).tolist()

            if self.network.network_type in ["GN", "GN2"]:
                # if the GN variant of the network is used, the fb weights do not need
                # to be trained
                self.freeze_fb_weights = True

            elif self.network.network_type == "DFA":
                # manipulate cmd arguments such that we use a DMLPDTP2 network with
                # linear MLP's with fixed weights
                self.training.freeze_fb_weights = True
                self.network.network_type = "DMLPDTP2"
                self.network.size_mlp_fb = None
                self.network.fb_activation = "linear"
                self.training.train_randomized = False

            elif self.network.network_type == "DFAConv":
                self.training.freeze_fb_weights = True
                self.network.network_type = "DDTPConv"
                self.network.fb_activation = "linear"
                self.training.train_randomized = False

            elif self.network_type == "DFAConvCIFAR":
                self.training.freeze_fb_weights = True
                self.network.network_type = "DDTPConvCIFAR"
                self.network.fb_activation = "linear"
                self.training.train_randomized = False

            if isinstance(self.logging.gn_damping, str) and "," in self.logging.gn_damping:
                self.logging.gn_damping = utils.str_to_list(self.logging.gn_damping, type=float)  # type: ignore
            else:
                self.logging.gn_damping = float(self.logging.gn_damping)

            # Checking valid combinations of command line arguments
            if self.training.shallow_training:
                if not self.network.network_type == "BP":
                    raise ValueError(
                        "The shallow_training method is only implemented"
                        "in combination with BP. Make sure to set "
                        "the network_type argument on BP."
                    )

    def __init__(
        self,
        datamodule: CIFAR10DataModule,
        network: MeulemansNetwork,
        hparams: Meulemans.HParams | None = None,
        config: MiscConfig | None = None,
    ):
        if not isinstance(network, MeulemansNetwork):
            raise RuntimeError(
                f"Meulemans DTP only works with a specific network architecture. "
                f"Can't yet use networks of type {type(network)}."
            )
        super().__init__(datamodule=datamodule, network=network, hparams=hparams, config=config)
        self.hp: Meulemans.HParams
        self.automatic_optimization = False

        temp_forward_optimizer_list, temp_feedback_optimizer_list = utils.choose_optimizer(
            self.hp, self.network
        )
        self.n_forward_optimizers = len(temp_forward_optimizer_list._optimizer_list)
        self.n_feedback_optimizers = len(temp_feedback_optimizer_list._optimizer_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def configure_optimizers(self):
        forward_optimizer_list, feedback_optimizer_list = utils.choose_optimizer(
            self.hp, self.network
        )
        assert len(forward_optimizer_list._optimizer_list) == self.n_forward_optimizers
        assert len(feedback_optimizer_list._optimizer_list) == self.n_feedback_optimizers

        return [*forward_optimizer_list._optimizer_list, *feedback_optimizer_list._optimizer_list]

    @property
    def feedback_optimizers(self) -> list[torch.optim.Optimizer]:
        return self.optimizers()[self.n_forward_optimizers :]  # type: ignore

    @property
    def forward_optimizers(self) -> list[torch.optim.Optimizer]:
        return self.optimizers()[: self.n_forward_optimizers]  # type: ignore

    def shared_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, phase: str
    ) -> StepOutputDict:
        x, y = batch
        # TODO: Not currently doing any of the pretraining stuff from their repo.
        assert not self.hp.freeze_fb_weights

        predictions = self.network(x)
        if phase == "train":
            # TODO: Make sure this is equivalent to `train_parallel`.
            # TODO: Use `self.current_epoch` in combination with the relevant hparam from `Args`
            # to recreate the "only train feedback weights for a given number of epochs" behaviour.
            for optim in self.forward_optimizers:
                optim.zero_grad()
            for optim in self.feedback_optimizers:
                optim.zero_grad()

            self.train_forward_parameters(inputs=x, predictions=predictions, targets=y)
            self.train_feedback_parameters()

            for optim in self.forward_optimizers:
                optim.step()
            # forward_optimizer.step()

        return {"logits": predictions, "y": y}

    def train_feedback_parameters(self):
        """Train the feedback parameters on the current mini-batch.

        Adapted from the meulemans codebase.

        TODO: Extract the loss calculation logic of the layers, and instead of having all the loss
        calculations inside the layers (as it currently is in the meulemans code) use the same kind
        of structure as in DTP, where we sum the reconstruction losses of all the layers.
        """
        feedback_optimizers = self.feedback_optimizers

        def _optimizer_step():
            for optimizer in feedback_optimizers:
                optimizer.step()

        def _zero_grad(set_to_none: bool = False):
            for optimizer in feedback_optimizers:
                optimizer.zero_grad(set_to_none=set_to_none)

        _zero_grad()
        # TODO: Assuming these for now, to simplify the code a bit.
        assert self.hp.direct_fb
        assert not self.hp.train_randomized_fb
        assert not self.hp.diff_rec_loss

        # TODO: Make sure that the last layer is trained properly as well. (The [:-1] here comes
        # from the meulemans code)
        for layer_index, layer in enumerate(self.network.layers[:-1]):
            n_iter = layer._nb_feedback_iterations
            for iteration in range(n_iter):
                # TODO: Double-check if they are zeroing the gradients in each layer properly.
                # So far it seems like they aren't!
                self.network.compute_feedback_gradients(layer_index)
                # NOTE: @lebrice: Is it really necessary to step all optimizers for each
                # iteration, for each layer? Isn't this O(n^3) with n the number of layers?
                # Maybe it's necessary because of the weird direct feedback connections?
                _optimizer_step()

    def train_forward_parameters(
        self,
        inputs: Tensor,
        predictions: Tensor,
        targets: Tensor,
    ):
        """Train the forward parameters on the current mini-batch.

        NOTE: This method is an adaptation of the code from the meulemans codebase.
        """
        output_activation_to_loss_fn = {"softmax": F.cross_entropy, "sigmoid": F.mse_loss}
        loss_function = output_activation_to_loss_fn[self.hp.network.output_activation]
        # NOTE: Assuming these for now.
        assert not self.hp.training.train_randomized

        forward_optimizers = self.forward_optimizers

        if not predictions.requires_grad:
            # we need the gradient of the loss with respect to the network
            # output. If a LeeDTPNetwork is used, this is already the case.
            # The gradient will also be saved in the activations attribute of the
            # output layer of the network
            predictions.requires_grad = True
            # NOTE: (@lebrice) This might be "safer", to make sure that we don't backpropagate into
            # the weights that created the predictions:
            # predictions = predictions.clone().detach().requires_grad_(True)

        # NOTE: Simplifying the code a bit by assuming that this is False, for now.
        # save_target = args.save_GN_activations_angle or args.save_BP_activations_angle

        # forward_optimizer.zero_grad()
        for optimizer in forward_optimizers:
            optimizer.zero_grad()

        loss = loss_function(predictions, targets)

        # Get the target using one backprop step with lr of beta.
        # NOTE: target_lr := beta in our paper.
        output_target = self.network.compute_output_target(
            loss, target_lr=self.hp.training.target_stepsize
        )

        # Computes and saves the gradients for the forward parameters for that layer.
        self.network.layers[-1].compute_forward_gradients(
            h_target=output_target, h_previous=self.network.layers[-2].activations
        )
        # if save_target:
        #     self.layers[-1].target = output_target

        for i in range(self.network.depth - 1):
            h_target = self.network.propagate_backward(output_target, i)
            layer = self.network.layers[i]
            # if save_target:
            #     self.layers[i]._target = h_target
            if i == 0:
                assert isinstance(layer, DDTPConvLayer)
                layer.compute_forward_gradients(
                    h_target=h_target,
                    h_previous=inputs,
                    forward_requires_grad=self.network.forward_requires_grad,
                )
            else:
                previous_layer = self.network.layers[i - 1]
                previous_activations: Tensor | None = previous_layer.activations
                if i == self.network.nb_conv:  # flatten conv layer
                    assert previous_activations is not None
                    previous_activations = previous_activations.flatten(1)
                layer.compute_forward_gradients(
                    h_target, previous_activations, self.network.forward_requires_grad
                )

        # TODO: In the meulemans `train_parallel` code, they seem to do an extra `f_optimizer.step`
        # at the end of each batch. Either it's an extra step, or it's the only step, in which case
        # we'd have to also do it only after the feedback weights have been trained, as they do it.
        # for f_optimizer in forward_optimizers:
        #     f_optimizer.step()
