from __future__ import annotations

import argparse
import dataclasses
from ast import literal_eval
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
import torch
from pl_bolts.datamodules import CIFAR10DataModule
from simple_parsing.helpers.serialization.serializable import Serializable
from torch import Tensor
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

import meulemans_dtp.main
from meulemans_dtp.final_configs.cifar10_DDTPConv import config as _config
from meulemans_dtp.lib import utils
from meulemans_dtp.lib.conv_layers import DDTPConvLayer
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
from target_prop.networks.meulemans_convnet import MeulemansConvNet

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


class Meulemans(Model[MeulemansConvNet]):
    @dataclass
    class HParams(Model.HParams):
        """Arguments for our adapted version of the Meuleman's model.

        NOTE: Each field here corresponds to a group of arguments from the meulemans repo main
        script. Originally, the `args` object was a namespace, but now it is a dataclass, where we
        grouped the arguments from each group into a distinct dataclass (DatasetOptions,
        TrainOptions, etc). All these classes are combined to create the `Args` class.

        Here, we have the same exact arguments, but they are structured a bit differently.
        Each group is set on a different property. We also only actually use a fraction of them,
        since we control the dataset / etc differently in our repo (via the DatasetConfig object).

        This is kind of an in-between solution, until we re-implement the network/training logic
        from their codebase.
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
        # NOTE: Simplifying these, since we only care about img classification atm.
        classification: bool = field(default=True, init=False)
        regression: bool = field(default=False, init=False)
        diff_rec_loss: bool = field(default=False, init=False)
        direct_fb: bool = field(default=True, init=False)
        save_angle: bool = field(default=False, init=False)

        def __post_init__(self):
            """Do the postprocessing of the arguments. This is largely copied from their repo.

            TODO: Remove the parts of this that we don't need, once we're confident that we've
            replicated their implementation correctly (once the repro tests pass).
            """
            self.save_angle = (
                self.logging.save_GN_activations_angle
                or self.logging.save_BP_activations_angle
                or self.logging.save_BP_angle
                or self.logging.save_GN_angle
                or self.logging.save_GNT_angle
            )

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
            if isinstance(self.network.size_hidden, str):
                _hdim = utils.process_hdim(self.network.size_hidden)
                assert isinstance(_hdim, int)
                self.network.size_hidden = _hdim
            if self.network.size_mlp_fb == "None":
                self.network.size_mlp_fb = None

            if self.training.optimizer_fb is None:
                self.training.optimizer_fb = self.training.optimizer
            elif isinstance(self.network.size_mlp_fb, str):
                _size_mlp_fb = utils.process_hdim_fb(self.network.size_mlp_fb)
                assert isinstance(_size_mlp_fb, int)
                self.network.size_mlp_fb = _size_mlp_fb

            # Manipulating command line arguments if asked
            # TODO: They were manipulating these if they were strings, which isn't necessary anymore,
            # however, it might be important to be able to load from their configs in the long run.
            self.misc.random_seed = int(self.misc.random_seed)

            if self.training.normalize_lr:
                self.training.lr = (
                    np.array(self.training.lr) / self.training.target_stepsize
                ).tolist()

            # NOTE: The following lines are not currently used, since we currently only have
            # "DDTPConvCifar" as a network.

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

            elif self.network.network_type == "DFAConvCIFAR":
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
        network: MeulemansConvNet,
        hparams: Meulemans.HParams | None = None,
        config: MiscConfig | None = None,
    ):
        if not isinstance(network, MeulemansConvNet):
            raise RuntimeError(
                f"Meulemans DTP only works with a specific network architecture. "
                f"Can't yet use networks of type {type(network)}."
            )
        super().__init__(datamodule=datamodule, network=network, hparams=hparams, config=config)
        self.hp: Meulemans.HParams
        self.automatic_optimization = False

        self.n_forward_optimizers = len(network)
        self.n_feedback_optimizers = 1

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def configure_optimizers(self):
        """Create the optimizers."""
        # NOTE: Extracted and adapted these few lines in order to make it possible for us to
        # change the structure and contents of the hparams class of this model later.

        # forward_optimizer_list, feedback_optimizer_list = utils.choose_optimizer(
        #     self.hp, self.network
        # )
        # assert len(forward_optimizer_list._optimizer_list) == self.n_forward_optimizers
        # assert len(feedback_optimizer_list._optimizer_list) == self.n_feedback_optimizers
        # return [*forward_optimizer_list._optimizer_list, *feedback_optimizer_list._optimizer_list]

        forward_optimizers: list[Optimizer] = []
        assert len(self.hp.training.lr) == len(self.network)
        for i, (lr, layer) in enumerate(zip(self.hp.training.lr, self.network)):
            assert isinstance(layer, (DDTPConvLayer, DDTPMLPLayer))

            if self.hp.network.no_bias:
                assert not hasattr(layer, "bias") or layer.bias is None
                parameters = [layer.weights]
            else:
                parameters = [layer.weights, layer.bias]

            if self.hp.training.optimizer == "SGD":
                optimizer = torch.optim.SGD(
                    parameters,
                    lr=lr,
                    momentum=self.hp.training.momentum,
                    weight_decay=self.hp.training.forward_wd,
                )
            else:
                assert self.hp.training.optimizer == "Adam"
                eps = self.hp.adam.epsilon[i]
                optimizer = torch.optim.Adam(
                    parameters,
                    lr=lr,
                    betas=(self.hp.adam.beta1, self.hp.adam.beta2),
                    eps=eps,
                    weight_decay=self.hp.training.forward_wd,
                )
            forward_optimizers.append(optimizer)

        feedback_params = self.network.get_feedback_parameter_list()
        if isinstance(self.hp.training.lr_fb, float):
            if self.hp.training.optimizer_fb == "SGD":
                feedback_optimizer = torch.optim.SGD(
                    feedback_params,
                    lr=self.hp.training.lr_fb,
                    weight_decay=self.hp.training.feedback_wd,
                )
            elif self.hp.training.optimizer_fb == "RMSprop":
                feedback_optimizer = torch.optim.RMSprop(
                    feedback_params,
                    lr=self.hp.training.lr_fb,
                    momentum=self.hp.training.momentum,
                    alpha=0.95,
                    eps=0.03,
                    weight_decay=self.hp.training.feedback_wd,
                    centered=True,
                )
            else:
                assert self.hp.training.optimizer_fb == "Adam"
                feedback_optimizer = torch.optim.Adam(
                    feedback_params,
                    lr=self.hp.training.lr_fb,
                    betas=(self.hp.adam.beta1_fb, self.hp.adam.beta2_fb),
                    eps=self.hp.adam.epsilon_fb,
                    weight_decay=self.hp.training.feedback_wd,
                )
            feedback_optimizers = [feedback_optimizer]
        else:
            assert self.hp.network.network_type == "DDTPConv"
            assert isinstance(self.hp.training.lr_fb, (list, np.ndarray))
            assert len(self.hp.training.lr_fb) == 2
            assert self.hp.training.optimizer == "Adam"

            epsilon_fb: list[float] = []
            if isinstance(self.hp.adam.epsilon_fb, float):
                epsilon_fb = [self.hp.adam.epsilon_fb, self.hp.adam.epsilon_fb]
            else:
                assert len(self.hp.adam.epsilon_fb) == 2
                epsilon_fb = self.hp.adam.epsilon_fb
            conv_fb_parameters = self.network.get_conv_feedback_parameter_list()
            fc_fb_parameters = self.network.get_fc_feedback_parameter_list()
            conv_fb_optimizer = torch.optim.Adam(
                conv_fb_parameters,
                lr=self.hp.training.lr_fb[0],
                betas=(self.hp.adam.beta1_fb, self.hp.adam.beta2_fb),
                eps=epsilon_fb[0],
                weight_decay=self.hp.training.feedback_wd,
            )
            fc_fb_optimizer = torch.optim.Adam(
                fc_fb_parameters,
                lr=self.hp.training.lr_fb[1],
                betas=(self.hp.adam.beta1_fb, self.hp.adam.beta2_fb),
                eps=epsilon_fb[1],
                weight_decay=self.hp.training.feedback_wd,
            )

            feedback_optimizers = [conv_fb_optimizer, fc_fb_optimizer]
        return [*forward_optimizers, *feedback_optimizers]

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

        # TODO: Use `self.current_epoch` in combination with the relevant hparam from `Args`
        # to recreate the "only train feedback weights for a given number of epochs" behaviour.
        # TODO: Currently doing the same thing as in their codebase (step all the optimizers),
        # however it might not make sense, since we don't want to step more than once!
        train_forward = self.current_epoch >= self.hp.training.epochs_fb
        train_feedback = not self.hp.training.freeze_fb_weights
        # FIXME: Remove this.
        train_forward = train_feedback = True
        predictions = self.network(x)
        outputs: StepOutputDict = {"logits": predictions, "y": y}
        if phase != "train":
            return outputs

        if train_forward and train_feedback:
            # Training both.
            for optim in self.feedback_optimizers:
                optim.zero_grad(set_to_none=True)
            for optim in self.forward_optimizers:
                optim.zero_grad(set_to_none=True)

            self.train_forward_parameters(inputs=x, predictions=predictions, targets=y)
            self.train_feedback_parameters()

            for optim in self.forward_optimizers:
                optim.step()
        elif train_feedback:
            raise NotImplementedError
        elif train_forward:
            raise NotImplementedError

        return outputs

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
        assert not self.hp.training.train_randomized_fb
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
