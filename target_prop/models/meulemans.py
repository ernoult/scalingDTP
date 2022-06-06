from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch
from pl_bolts.datamodules import CIFAR10DataModule
from torch import Tensor, nn
from torch.nn import functional as F

from meulemans_dtp.final_configs.cifar10_DDTPConv import config as _config
from meulemans_dtp.lib import utils
from meulemans_dtp.lib.conv_layers import DDTPConvLayer
from meulemans_dtp.lib.conv_network import DDTPConvNetworkCIFAR
from meulemans_dtp.main import Args
from target_prop.config import MiscConfig
from target_prop.models.model import Model, StepOutputDict
from target_prop.networks import Network


def clean_up_config_dict(config: dict):
    cleaned_up_config = config.copy()
    cleaned_up_config["epsilon"] = literal_eval(cleaned_up_config["epsilon"])
    return cleaned_up_config


import copy


def _replace_ndarrays_with_lists(args: Args):
    cleaned_up_config = copy.deepcopy(args)
    for key, value in vars(args).items():
        if isinstance(value, np.ndarray):
            setattr(cleaned_up_config, key, value.tolist())
    return cleaned_up_config


DEFAULT_ARGS = _replace_ndarrays_with_lists(Args.from_dict(clean_up_config_dict(_config)))


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
        hparams = hparams or self.HParams()
        self.hparams = hparams
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


class Meulemans(Model):
    @dataclass
    class HParams(Model.HParams):
        """Arguments for our adapted version of the Meuleman's model."""

        args: Args = DEFAULT_ARGS
        """ The arguments from the Meulemans codebase. """

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
        self.config = config
        del self.forward_net
        self.network = network

        self.automatic_optimization = False

        temp_forward_optimizer_list, temp_feedback_optimizer_list = utils.choose_optimizer(
            self.hp.args, self.network
        )
        self.n_forward_optimizers = len(temp_forward_optimizer_list._optimizer_list)
        self.n_feedback_optimizers = len(temp_feedback_optimizer_list._optimizer_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def configure_optimizers(self):
        forward_optimizer_list, feedback_optimizer_list = utils.choose_optimizer(
            self.hp.args, self.network
        )
        assert len(forward_optimizer_list._optimizer_list) == self.n_forward_optimizers
        assert len(feedback_optimizer_list._optimizer_list) == self.n_feedback_optimizers

        # TODO: For PL, we need to return the list of optimizers, but the code for Meulemans
        # expects
        # to get an OptimizerList object...
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

        predictions = self.network(x)
        if phase == "train":
            self.train_feedback_parameters()

            # NOTE: Not using the outputs of this method at the moment.
            self.train_forward_parameters(inputs=x, predictions=predictions, targets=y)
        return {"logits": predictions, "y": y}

    def train_feedback_parameters(self):
        """Train the feedback parameters on the current mini-batch.

        Adapted from the meulemans codebase.

        TODO: Extract the loss calculation logic of the layers, and instead of having all the loss
        calculations inside the layers (as it currently is in the meulemans code) use the same kind
        of structure as in DTP, where we sum the reconstruction losses of all the layers.
        """
        args = self.hp.args
        feedback_optimizers = self.feedback_optimizers
        net = self.network

        def _optimizer_step():
            for optimizer in feedback_optimizers:
                optimizer.step()

        def _zero_grad(set_to_none: bool = False):
            for optimizer in feedback_optimizers:
                optimizer.zero_grad(set_to_none=set_to_none)

        _zero_grad()
        # TODO: Assuming these for now, to simplify the code a bit.
        assert args.direct_fb
        assert not args.train_randomized_fb
        assert not args.diff_rec_loss

        # TODO: Make sure that the last layer is trained properly as well. (The [:-1] here comes
        # from the meulemans code)
        for layer_index, layer in enumerate(net.layers[:-1]):
            n_iter = layer._nb_feedback_iterations
            for iteration in range(n_iter):
                # TODO: Double-check if they are zeroing the gradients in each layer properly.
                # So far it seems like they aren't!
                net.compute_feedback_gradients(layer_index)
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
        loss_function = output_activation_to_loss_fn[self.hp.args.output_activation]
        # NOTE:
        assert not self.hp.args.train_randomized

        # net = self.network
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
        output_target = self.network.compute_output_target(loss, target_lr=args.target_stepsize)

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

        # NOTE: Removing those, since they aren't even used atm.
        # if self.hp.args.classification:
        #     if self.hp.args.output_activation == "sigmoid":
        #         batch_accuracy = utils.accuracy(predictions, utils.one_hot_to_int(targets))
        #     else:  # softmax
        #         batch_accuracy = utils.accuracy(predictions, targets)
        # else:
        #     batch_accuracy = None
        # batch_loss = loss.detach()

        # return batch_accuracy, batch_loss
