"""
Numerical tests for Theorems 4.2 and 4.3.

Run the following from project root to execute tests:
pytest target_prop/networks/lenet_test.py -s
"""

import os
import pdb
from collections import defaultdict
from dataclasses import dataclass
from email.policy import default
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
import pandas as pd
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
from target_prop.utils import is_trainable, named_trainable_parameters


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
    angles_data = {}

    @pytest.mark.skip("skip for now")
    @pytest.mark.parametrize("num_iters", [5000])
    @pytest.mark.parametrize("noise", [0.1])
    @pytest.mark.parametrize("lr", [0.03])
    @pytest.mark.parametrize("seed", [123])
    def test_feedback_weight_training(
        self,
        dtp_hparams: DTP.HParams,
        dtp_model: DTP,
        num_iters: int,
        noise: float,
        lr: float,
        seed: int,
    ):
        # Fix seed
        seed_everything(seed=seed, workers=True)
        if torch.cuda.is_available():
            device = torch.device("cuda")

        # Setup CIFAR10 datamodule
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
        outputs = dtp_model.feedback_loss(data, label, phase="train")

        # Check angles and distances
        layer_angles = outputs["layer_angles"][-1]  # get last layer angles
        print(
            f"noise={noise}, lr={lr}, angles={layer_angles[-10:]}"
        )  # print last 10 iteration angles
        self._line_plot(np.arange(num_iters), np.array(layer_angles), "foo")

        # Make sure that converged output layer angle is less than 8 degrees
        assert layer_angles[-1] < 8.0

    # @pytest.mark.skip("skip for now")
    @pytest.mark.parametrize("seed", [123, 124, 125, 126, 127])
    def test_dtp_forward_updates_are_orthogonal_to_backprop_with_random_init(
        self, dtp_hparams: DTP.HParams, dtp_no_bias_model: DTP, seed: int
    ):
        # Fix seed
        dtp_model = dtp_no_bias_model  # rename for ease
        seed_everything(seed=seed, workers=True)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        # device = torch.device("cpu")

        # Setup CIFAR10 datamodule
        config = Config(dataset="cifar10", num_workers=0, debug=True)
        datamodule = config.make_datamodule(batch_size=dtp_hparams.batch_size)
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        # Get a batch
        data, label = next(iter(datamodule.train_dataloader()))
        data = data.to(device)
        label = label.to(device)

        # Setup DTP model with random weights
        dtp_model = dtp_model.to(device)

        # Get backprop and DTP grads for a batch
        from target_prop.callbacks import get_backprop_grads, get_dtp_grads

        backprop_grads = get_backprop_grads(dtp_model, data, label)
        dtp_grads = get_dtp_grads(dtp_model, data, label, temp_beta=dtp_model.hp.beta)

        # Compare gradients by angles and distances
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
        self.angles_data[f"random_init_{seed}"] = angles
        for key, value in angles.items():
            print(key, value)

    # @pytest.mark.skip("skip for now")
    @pytest.mark.parametrize("seed", [123, 124, 125, 126, 127])
    def test_dtp_forward_updates_match_backprop_with_symmetric_init(
        self, dtp_hparams: DTP.HParams, dtp_no_bias_model: DTP, seed: int
    ):
        # Fix seed
        dtp_model = dtp_no_bias_model  # rename for ease
        seed_everything(seed=seed, workers=True)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        # device = torch.device("cpu")

        # Setup CIFAR10 datamodule
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

        # Get backprop and DTP grads for a batch
        from target_prop.callbacks import get_backprop_grads, get_dtp_grads

        backprop_grads = get_backprop_grads(dtp_model, data, label)
        dtp_grads = get_dtp_grads(dtp_model, data, label, temp_beta=dtp_model.hp.beta)

        # Compare gradients by angles and distances
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
        self.angles_data[f"symmetric_init_{seed}"] = angles
        for key, value in angles.items():
            print(key, value)
        return True

    @pytest.mark.parametrize("num_iters", [[5000, 5000, 5000]])
    @pytest.mark.parametrize("noise", [[0.2, 0.2, 0.1]])  # layer order: [first, ..., last]
    @pytest.mark.parametrize("lrs", [[1e-4, 2.5e-3, 0.01]])  # layer order: [first, ..., last]
    @pytest.mark.parametrize("seed", [123, 124, 125, 126, 127])
    def test_dtp_forward_updates_match_backprop_with_ldrl_init(
        self,
        dtp_hparams: DTP.HParams,
        dtp_no_bias_model: DTP,
        num_iters: List[int],
        noise: List[float],
        lrs: List[float],
        seed: int,
    ):
        # Fix seed
        print("noise: ", noise)
        print("lrs: ", lrs)
        print("num_iters: ", num_iters)
        dtp_model = dtp_no_bias_model  # rename for ease
        seed_everything(seed=seed, workers=True)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        # device = torch.device("cpu")

        # Setup CIFAR10 datamodule
        config = Config(dataset="cifar10", num_workers=0, debug=True)
        datamodule = config.make_datamodule(batch_size=dtp_hparams.batch_size)
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        # Get a batch
        data, label = next(iter(datamodule.train_dataloader()))
        data = data.to(device)
        label = label.to(device)

        # Setup DTP model with LeNet hyperparams
        dtp_model = dtp_model.to(device)

        # Assert lengths for LeNet
        assert len(num_iters) == 3
        assert len(noise) == 3
        assert len(lrs) == 3

        # Do feedback weight training (L-DRL)
        # Add [0] for first feedback layer which is never trained
        dtp_model.feedback_iterations = num_iters[::-1] + [0]  # reverse order for feedback training
        dtp_model.feedback_noise_scales = noise[::-1] + [0]
        dtp_model.feedback_lrs = lrs[::-1] + [0]
        self._setup_feedback_optimizers(dtp_model)
        outputs = dtp_model.feedback_loss(data, label, phase="train")

        # Inspect angles between forward and backward layers after L-DRL
        layer_angles = outputs["layer_angles"]
        layer_names = [name for name, _ in named_trainable_parameters(dtp_model.forward_net)]
        layer_names = layer_names[1:]  # skip first layer since it's never trained
        assert len(layer_names) == len(layer_angles)
        for i in range(len(layer_angles)):
            name = layer_names[i]
            angle = layer_angles[i][-1]  # pick last iteration angle
            print(f"[{name}] angle between F & G after L-DRL: {angle}")

        # Get backprop and DTP grads for a batch
        from target_prop.callbacks import get_backprop_grads, get_dtp_grads

        backprop_grads = get_backprop_grads(dtp_model, data, label)
        dtp_grads = get_dtp_grads(dtp_model, data, label, temp_beta=dtp_model.hp.beta)

        # Compare gradients by angles and distances
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
        self.angles_data[f"l-drl_init_{seed}"] = angles
        for key, value in angles.items():
            print(key, value)
        return True

    # @pytest.mark.skip("skip for now")
    @pytest.mark.parametrize("path", ["./data"])  # skip this test if you don't want to plot
    def test_angle_plot(self, path: str):
        # Build pandas dataframe for plotting
        pd_angles_data = defaultdict(list)
        for init_scheme_seed in self.angles_data.keys():
            init_scheme = " ".join(init_scheme_seed.split("_")[:-1])
            seed = init_scheme_seed.split("_")[-1]
            for param_name, angle in self.angles_data[init_scheme_seed].items():
                pd_angles_data["seed"].append(seed)
                pd_angles_data["init_scheme"].append(init_scheme)
                pd_angles_data["param"].append(param_name)
                pd_angles_data["angle"].append(angle)
        df = pd.DataFrame(data=pd_angles_data)

        # Save data
        print("angle data:")
        print(df)
        df.to_csv(
            os.path.join(path, "angle_data.csv"),
            encoding="utf-8",
            index=False,
        )

        # Save figure
        file_path = os.path.join(path, "angle_plot.pdf")
        sns.set_theme(style="whitegrid")  # "darkgrid" also looks nice
        fig = sns.catplot(
            data=df,
            x="init_scheme",
            y="angle",
            hue="param",
            kind="bar",
            height=5,
            aspect=1.5,
            alpha=0.9,
            legend_out=False,
            errwidth=2,
            capsize=0.05,
        )
        fig.set_axis_labels("", "angle")
        fig.legend.set_title("params")
        plt.savefig(file_path, format="pdf", bbox_inches="tight")

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
