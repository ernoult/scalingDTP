import pdb
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from simple_parsing.helpers import choice, list_field
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from target_prop._weight_operations import init_symetric_weights
from target_prop.config import Config
from target_prop.metrics import compute_dist_angle
from target_prop.models import DTP
from target_prop.models.dtp import FeedbackOptimizerConfig, ForwardOptimizerConfig
from target_prop.networks import LeNet
from target_prop.utils import is_trainable

# import logging
# from collections import OrderedDict
# from dataclasses import dataclass
# from typing import ClassVar, Iterable, List, Optional, Tuple, Type

# import pytest
# import torch
# from pytorch_lightning import Trainer
# from pytorch_lightning.utilities.seed import seed_everything
# from simple_parsing.helpers import choice, list_field
# from simple_parsing.helpers.hparams import log_uniform, uniform
# from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
# from target_prop._weight_operations import init_symetric_weights
# from target_prop.backward_layers import mark_as_invertible
# from target_prop.config import Config
# from target_prop.layers import Reshape, forward_all, invert
# from target_prop.legacy import (
#     VGG,
#     createDataset,
#     createOptimizers,
#     train_backward,
#     train_batch,
#     train_forward,
# )
# from target_prop.metrics import compute_dist_angle
# from target_prop.models import DTP
# from target_prop.networks.simple_vgg import SimpleVGG
# from target_prop.utils import is_trainable, named_trainable_parameters
# from torch import Tensor, nn
# from torch.nn import functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.fixture
def dtp_hparams():
    @dataclass
    class DTPLeNetHParams(DTP.HParams):
        """DTP defaults for LeNet."""

        feedback_training_iterations: List[int] = list_field(41, 51, 24)
        batch_size: int = 256
        noise: List[float] = list_field(0.41640228838517584, 0.3826261146623929, 0.1395382069358601)
        beta: float = 0.4655
        b_optim: FeedbackOptimizerConfig = FeedbackOptimizerConfig(
            type="sgd",
            lr=[0.0007188427494432325, 0.00012510321884615596, 0.03541466958291287],
            momentum=0.9,
        )
        f_optim: ForwardOptimizerConfig = ForwardOptimizerConfig(
            type="sgd", lr=0.03618, weight_decay=1e-4, momentum=0.9
        )

        max_epochs: int = 90
        init_symetric_weights: bool = False

    hparams = DTPLeNetHParams()
    return hparams


@pytest.fixture
def dtp_model(dtp_hparams: HyperParameters):
    config = Config(dataset="cifar10", num_workers=0, debug=False)
    datamodule = config.make_datamodule(batch_size=dtp_hparams.batch_size)
    network = LeNet(in_channels=datamodule.dims[0], n_classes=datamodule.num_classes)
    dtp_model = DTP(datamodule=datamodule, hparams=dtp_hparams, config=config, network=network)
    return dtp_model


@pytest.fixture
def dtp_no_bias_model(dtp_hparams: HyperParameters):
    config = Config(dataset="cifar10", num_workers=0, debug=False)
    datamodule = config.make_datamodule(batch_size=dtp_hparams.batch_size)
    network_hparams = LeNet.HParams()
    network_hparams.bias = False
    network = LeNet(
        in_channels=datamodule.dims[0], n_classes=datamodule.num_classes, hparams=network_hparams
    )
    dtp_model = DTP(datamodule=datamodule, hparams=dtp_hparams, config=config, network=network)
    return dtp_model


class TestLeNet:
    # @pytest.mark.skip("skip for now")
    @pytest.mark.parametrize("noise", [0.1])
    @pytest.mark.parametrize("lr", [0.03])
    def test_feedback_weight_training(
        self, dtp_hparams: DTP.HParams, dtp_model: DTP, noise: float, lr: float
    ):
        # Fix seed
        seed_everything(seed=123, workers=True)
        if torch.cuda.is_available():
            device = torch.device("cuda")

        # Setup CIFAR10 datamodule with batch size 1
        config = Config(dataset="cifar10", num_workers=0, debug=False)
        datamodule = config.make_datamodule(batch_size=dtp_hparams.batch_size)
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        # Get a batch
        data, label = next(iter(datamodule.train_dataloader()))
        data = data.to(device)
        label = label.to(device)

        # Setup DTP model with LeNet hyperparams
        # NOTE: we just train output layer feedback weights to align with forward on
        # a single batch, so we set all the other layer iterations to 0 for now
        num_iters = 5000
        dtp_model = dtp_model.to(device)
        dtp_model.feedback_iterations = [num_iters, 0, 0, 0]
        dtp_model.feedback_noise_scales[0] = noise
        dtp_model.feedback_lrs[0] = lr
        # following gets 2.8 degree
        # dtp_model.feedback_noise_scales[0] = 0.08
        # dtp_model.feedback_lrs[0] = 0.01
        print("number of params = ", count_parameters(dtp_model.forward_net))

        # Setup feedback optimizers
        self._setup_feedback_optimizers(dtp_model)

        # Run feedback weight training
        # pdb.set_trace()
        outputs = dtp_model.feedback_loss(data, label, phase="train")

        # Check angles and distances
        layer_angles = outputs["layer_angles"][-1]  # get last layer angles
        print(
            f"noise={noise}, lr={lr}, angles={layer_angles[-10:]}"
        )  # print last 10 iteration angles
        self._line_plot(np.arange(num_iters), np.array(layer_angles), "foo")

        # Make sure that converged output layer angle is less than 8 degrees
        assert layer_angles[-1] < 8.0

    # @pytest.mark.parametrize("noise", [0.1])
    # @pytest.mark.parametrize("dtp_model", [dtp_no_bias_model])
    def test_dtp_forward_updates_match_backprop_with_symmetric_init(
        self, dtp_hparams: DTP.HParams, dtp_no_bias_model: DTP
    ):
        # Fix seed
        dtp_model = dtp_no_bias_model  # rename for ease
        seed_everything(seed=123, workers=True)
        # if torch.cuda.is_available():
        #     device = torch.device("cuda")
        device = torch.device("cpu")

        # Setup CIFAR10 datamodule with batch size 1
        config = Config(dataset="cifar10", num_workers=0, debug=True)
        datamodule = config.make_datamodule(batch_size=dtp_hparams.batch_size)
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        # Get a batch
        data, label = next(iter(datamodule.train_dataloader()))
        data = data.to(device)
        label = label.to(device)

        # Setup DTP model with symmetric weights
        init_symetric_weights(dtp_model.forward_net, dtp_model.backward_net)
        dtp_model = dtp_model.to(device)

        # Get backprop and DTP grads
        from target_prop.callbacks import get_backprop_grads, get_dtp_grads

        backprop_grads = get_backprop_grads(dtp_model, data, label)
        dtp_grads = get_dtp_grads(dtp_model, data, label, temp_beta=dtp_model.hp.beta)

        # Compare gradients
        distances: Dict[str, float] = {}
        angles: Dict[str, float] = {}
        for (bp_param, bp_grad), (dtp_param, dtp_grad) in zip(
            backprop_grads.items(), dtp_grads.items()
        ):
            assert bp_param == dtp_param
            distance, angle = compute_dist_angle(bp_grad, dtp_grad)
            distances[bp_param] = distance
            angles[bp_param] = angle

        # Print angles for each layer
        for key, value in angles.items():
            print(key, value)
        return True

    def _line_plot(self, x, y, name, x_label=None, y_label=None):
        sns.set_theme(style="darkgrid")
        plt.plot(x, y)
        plt.savefig(f"{name}.png", bbox_inches="tight")

    def _setup_feedback_optimizers(self, dtp_model):
        feedback_optimizers = []
        for i, (feedback_layer, lr) in enumerate(
            zip(dtp_model.backward_net, dtp_model.feedback_lrs)
        ):
            layer_optimizer = None
            if i == (len(dtp_model.backward_net) - 1) or not is_trainable(feedback_layer):
                assert lr == 0.0, (i, lr, dtp_model.feedback_lrs, type(feedback_layer))
            else:
                assert lr != 0.0
                layer_optimizer = dtp_model.hp.b_optim.make_optimizer(feedback_layer, lrs=[lr])
            feedback_optimizers.append(layer_optimizer)
        dtp_model._feedback_optimizers = feedback_optimizers
