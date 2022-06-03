"""Numerical tests for Theorems 4.2 and 4.3.

Run the following from project root to execute tests: pytest target_prop/networks/lenet_test.py -s
"""

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, OrderedDict, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.seed import seed_everything
from torch import Tensor
from torch.utils.data import DataLoader

from target_prop._weight_operations import init_symetric_weights
from target_prop.callbacks import get_backprop_grads, get_dtp_grads
from target_prop.config import MiscConfig
from target_prop.config.optimizer_config import OptimizerConfig
from target_prop.datasets.dataset_config import get_datamodule
from target_prop.metrics import compute_dist_angle
from target_prop.models import DTP
from target_prop.networks import LeNet
from target_prop.utils.utils import is_trainable, named_trainable_parameters

pytestmark = pytest.mark.skipif("-vv" not in sys.argv, reason="These tests take a while to run.")


@pytest.fixture(scope="module", params=[123, 124, 125, 126, 127])
def seed(request):
    """Fixture that seeds all the randomness, and provides the random seed used.

    When another fixture uses this one, it is also re-run for every seed here, and has its
    randomness seeded.

    Starting from this fixture, a "graph" of fixtures is created, where every node that follows is
    using the same seed as its parent, etc.
    """
    seed = request.param
    print(f"Runnign tests with seed {seed}")
    seed_everything(seed, workers=True)
    yield seed


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.fixture(scope="module", params=[[0.2, 0.2, 0.1]])
def noise(request):
    """NOTE: Creating a fixture for this so that all tests get the values."""
    # NOTE: layer order: [first, ..., last]
    return request.param


@pytest.fixture(scope="module", params=[[1e-4, 2.5e-3, 0.01]])
def feedback_lrs(request):
    # NOTE: layer order: [first, ..., last]
    return request.param


@pytest.fixture(scope="module")
def dtp_hparams():
    """Fixture that returns the hyper-parameters of L-DRL algo (DTP)"""
    return DTP.HParams(
        feedback_training_iterations=[41, 51, 24],
        batch_size=256,
        noise=[0.41640228838517584, 0.3826261146623929, 0.1395382069358601],
        beta=0.4655,
        b_optim=OptimizerConfig(
            type="sgd",
            lr=[0.0007188427494432325, 0.00012510321884615596, 0.03541466958291287],
            momentum=0.9,
        ),
        f_optim=OptimizerConfig(type="sgd", lr=[0.03618], weight_decay=1e-4, momentum=0.9),
        max_epochs=90,
        init_symetric_weights=False,
    )


@pytest.fixture(scope="module")
def config(seed: int):
    return MiscConfig(debug=True, seed=seed)


@pytest.fixture(scope="module")
def datamodule(dtp_hparams: DTP.HParams):
    return get_datamodule(dataset="cifar10", num_workers=0, batch_size=dtp_hparams.batch_size)


@pytest.fixture(scope="module")
def x_and_y(seed: int, config: MiscConfig, datamodule: LightningDataModule):
    """Yields a batch of data, for the given seed."""
    # Setup CIFAR10 datamodule
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    train_loader: DataLoader[Tuple[Tensor, Tensor]] = datamodule.train_dataloader()  # type: ignore
    # batch_index = np.random.randint(len(train_loader))
    # Get a batch
    data: Tensor
    label: Tensor
    data, label = next(iter(train_loader))
    data = data.to(config.device)
    label = label.to(config.device)
    print(f"Seed: {seed}: first 10 labels: {label[:10]}")
    return data, label


@pytest.fixture(scope="module")
def initial_weights(seed: int, datamodule: VisionDataModule, x_and_y: Tuple[Tensor, Tensor]):
    """Module-wide fixture that gives initial weights for a given seed."""
    net = LeNet(
        in_channels=datamodule.dims[0],
        n_classes=datamodule.num_classes,  # type: ignore
        hparams=LeNet.HParams(bias=True),
    )
    x, y = x_and_y
    net.to(device=x.device)
    _ = net(x)  # initialize lazy weights.
    state_dict = net.state_dict()
    yield state_dict


@pytest.fixture(scope="function")
def network(
    datamodule: VisionDataModule,
    config: MiscConfig,
    initial_weights: OrderedDict[str, Tensor],
    x_and_y: Tuple[Tensor, Tensor],
):
    # NOTE: The network / model fixtures have to be function-scoped, so that training the network
    # doesn't affect other runs. This was correctly done before, no worries.
    net = LeNet(
        in_channels=datamodule.dims[0],
        n_classes=datamodule.num_classes,  # type: ignore
        hparams=LeNet.HParams(bias=True),
    )
    net.load_state_dict(initial_weights, strict=True)
    net.to(device=torch.device(config.device))
    x, _ = x_and_y
    _ = net(x)  # dummy forward pass just to initialize the lazy layers.
    return net


@pytest.fixture(scope="function")
def no_bias_network(
    datamodule: VisionDataModule,
    config: MiscConfig,
    initial_weights: OrderedDict[str, Tensor],
    x_and_y: Tuple[Tensor, Tensor],
):
    net = LeNet(
        in_channels=datamodule.dims[0],
        n_classes=datamodule.num_classes,  # type: ignore
        hparams=LeNet.HParams(bias=False),
    )
    # Load, but ignore the biases.
    missing, unexpected = net.load_state_dict(initial_weights, strict=False)
    assert not missing
    assert all("bias" in param_name for param_name in unexpected)
    net.to(device=torch.device(config.device))
    # mark_as_invertible(net)
    x, _ = x_and_y
    _ = net(x)  # dummy forward pass just to initialize the lazy layers, if any.
    return net


@pytest.fixture(scope="function")
def dtp_model(
    network: LeNet,
    config: MiscConfig,
    datamodule: VisionDataModule,
    dtp_hparams: DTP.HParams,
):
    # config = Config(dataset="cifar10", num_workers=0, debug=False)
    # datamodule = config.make_datamodule(batch_size=dtp_hparams.batch_size)
    # network = LeNet(in_channels=datamodule.dims[0], n_classes=datamodule.num_classes)  # type: ignore
    dtp_model = DTP(datamodule=datamodule, hparams=dtp_hparams, config=config, network=network)
    dtp_model.to(device=config.device)
    return dtp_model


@pytest.fixture(scope="function")
def dtp_no_bias_model(
    no_bias_network: LeNet,
    datamodule: VisionDataModule,
    config: MiscConfig,
    dtp_hparams: DTP.HParams,
):
    model = DTP(
        datamodule=datamodule,
        network=no_bias_network,
        hparams=dtp_hparams,
        config=config,
    )
    model.to(device=config.device)
    return model


@pytest.fixture(scope="module")
def backprop_grads(
    x_and_y: Tuple[Tensor, Tensor],
    datamodule: VisionDataModule,
    initial_weights: OrderedDict[str, Tensor],
):
    """Returns the backprop gradients for a given network (for a given seed)."""
    # NOTE: We want to compute these backprop gradients only once per seed and reuse them in all
    # tests. Therefore, we need to create a network here that is used only for those.
    x, y = x_and_y
    network = LeNet(
        in_channels=x.shape[1],
        n_classes=datamodule.num_classes,  # type: ignore
        hparams=LeNet.HParams(bias=True),
    )
    network.load_state_dict(initial_weights, strict=True)
    network.to(x.device)
    grads = get_backprop_grads(network, x=x, y=y)
    # del network  # Safe to do, since grads are detached/cloned, but might be unnecessary.
    return grads


@pytest.fixture(scope="module")
def backprop_grads_no_bias(
    x_and_y: Tuple[Tensor, Tensor],
    datamodule: VisionDataModule,
    initial_weights: OrderedDict[str, Tensor],
):
    """Returns the backprop gradients for a given network without biases (for a given seed)."""
    # NOTE: We want to compute these backprop gradients only once per seed and reuse them in all
    # tests. Therefore, we need to create a network here that is used only for those.
    x, y = x_and_y
    network = LeNet(in_channels=x.shape[1], n_classes=10, hparams=LeNet.HParams(bias=False))
    missing, unexpected = network.load_state_dict(initial_weights, strict=False)
    assert not missing
    assert all("bias" in param_name for param_name in unexpected)
    network.to(x.device)
    grads = get_backprop_grads(network, x=x, y=y)
    # del network  # Safe to do, since grads are detached/cloned, but might be unnecessary.
    return grads


@pytest.fixture(scope="module", params=[[5000, 5000, 5000]])
def num_iters(request):
    iters: List[int] = request.param
    return iters


class TestLeNet:
    angles_data: Dict[str, Any] = {}
    distance_data: Dict[str, Any] = {}

    @pytest.mark.skip("skip for now")
    @pytest.mark.parametrize("num_iters", [5000])
    @pytest.mark.parametrize("noise", [0.1])
    @pytest.mark.parametrize("lr", [0.03])
    def test_feedback_weight_training(
        self,
        dtp_hparams: DTP.HParams,
        dtp_model: DTP,
        num_iters: int,
        noise: float,
        lr: float,
        seed: int,
    ):
        """Figure 4.2."""
        # TODO: Once this is checked to be working correctly, and we have results etc, replace all
        # this stuff below with the fixtures above.
        # Fix seed
        # seed_everything(seed=seed, workers=True)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Setup CIFAR10 datamodule
        datamodule = get_datamodule(
            dataset="cifar10", batch_size=dtp_hparams.batch_size, num_workers=0
        )
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        # Get a batch
        data: Tensor
        label: Tensor
        data, label = next(iter(datamodule.train_dataloader()))  # type: ignore
        data = data.to(device)
        label = label.to(device)

        # Setup DTP model with LeNet hyperparams
        # NOTE: we just train output layer feedback weights to align with forward on
        # a single batch, so we set all the other layer iterations to 0 for now
        dtp_model.to(device)
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

    def test_dtp_forward_updates_are_orthogonal_to_backprop_with_random_init(
        self, dtp_hparams: DTP.HParams, dtp_no_bias_model: DTP, config: MiscConfig
    ):
        # TODO: Once this is checked to be working correctly, and we have results etc, replace all
        # this stuff below with the fixtures above.
        dtp_model = dtp_no_bias_model  # rename for ease
        # seed_everything(seed=seed, workers=True)

        # Setup CIFAR10 datamodule
        config = MiscConfig(debug=True)
        datamodule = get_datamodule(
            dataset="cifar10", batch_size=dtp_hparams.batch_size, num_workers=0
        )
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        # Get a batch
        data: Tensor
        label: Tensor
        data, label = next(iter(datamodule.train_dataloader()))  # type: ignore
        data = data.to(config.device)
        label = label.to(config.device)

        # Setup DTP model with random weights
        dtp_model.to(config.device)

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
        self.distance_data[f"random_init_{seed}"] = distances
        for key, value in angles.items():
            print(key, value)

    def test_dtp_forward_updates_match_backprop_with_symmetric_init(
        self, dtp_hparams: DTP.HParams, dtp_no_bias_model: DTP, config: MiscConfig
    ):
        # TODO: Once this is checked to be working correctly, and we have results etc, replace all
        # this stuff below with the fixtures above.
        # Fix seed
        dtp_model = dtp_no_bias_model  # rename for ease
        # seed_everything(seed=seed, workers=True)

        # Setup CIFAR10 datamodule
        datamodule = get_datamodule(
            dataset="cifar10", num_workers=0, batch_size=dtp_hparams.batch_size
        )
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        # Get a batch
        data: Tensor
        label: Tensor
        data, label = next(iter(datamodule.train_dataloader()))  # type: ignore
        data = data.to(config.device)
        label = label.to(config.device)

        # Setup DTP model with symmetric weights
        init_symetric_weights(dtp_model.forward_net, dtp_model.backward_net)
        dtp_model.to(config.device)

        # Get backprop and DTP grads for a batch

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
        self.distance_data[f"symmetric_init_{seed}"] = distances
        for key, value in angles.items():
            print(key, value)
        return True

    def test_dtp_forward_updates_match_backprop_with_ldrl_init(
        self,
        dtp_hparams: DTP.HParams,
        dtp_no_bias_model: DTP,
        seed: int,
        num_iters: List[int],
        noise: List[float],
        feedback_lrs: List[float],
        config: MiscConfig,
    ):
        # TODO: Once this is checked to be working correctly, and we have results etc, replace all
        # this stuff below with the fixtures above.
        # Fix seed
        lrs = feedback_lrs
        print("noise: ", noise)
        print("lrs: ", lrs)
        print("num_iters: ", num_iters)
        dtp_model = dtp_no_bias_model  # rename for ease
        # seed_everything(seed=seed, workers=True)

        # Setup CIFAR10 datamodule
        datamodule = get_datamodule(
            dataset="cifar10", num_workers=0, batch_size=dtp_hparams.batch_size
        )
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        # Get a batch
        data: Tensor
        label: Tensor
        data, label = next(iter(datamodule.train_dataloader()))  # type: ignore
        data = data.to(config.device)
        label = label.to(config.device)

        # Setup DTP model with LeNet hyperparams
        dtp_model.to(config.device)

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
        self.distance_data[f"l-drl_init_{seed}"] = distances
        for key, value in angles.items():
            print(key, value)
        return True

    def test_meulemans_random_init(
        self,
        x_and_y: Tuple[Tensor, Tensor],
        no_bias_network: LeNet,
        initial_weights: Dict[str, Tensor],
        dtp_hparams: DTP.HParams,
        backprop_grads_no_bias: Dict[str, Tensor],
        seed: int,
    ):
        x, y = x_and_y
        distances, angles = meulemans(
            x=x,
            y=y,
            network=no_bias_network,
            initial_weights=initial_weights,
            network_hparams=no_bias_network.hparams,  # type: ignore
            backprop_gradients=backprop_grads_no_bias,
            beta=dtp_hparams.beta,
            n_pretraining_iterations=0,
            seed=seed,
        )

        self.angles_data[f"drl_init_{seed}"] = angles
        self.distance_data[f"drl_init_{seed}"] = distances

    def test_meulemans_trained(
        self,
        x_and_y: Tuple[Tensor, Tensor],
        no_bias_network: LeNet,
        initial_weights: Dict[str, Tensor],
        dtp_hparams: DTP.HParams,
        backprop_grads_no_bias: Dict[str, Tensor],
        seed: int,
    ):
        x, y = x_and_y
        distances, angles = meulemans(
            x=x,
            y=y,
            network=no_bias_network,
            initial_weights=initial_weights,
            network_hparams=no_bias_network.hparams,  # type: ignore
            backprop_gradients=backprop_grads_no_bias,
            beta=dtp_hparams.beta,
            n_pretraining_iterations=5000,
            seed=seed,
        )

        self.angles_data[f"drl_{seed}"] = angles
        self.distance_data[f"drl_{seed}"] = distances

    # @pytest.mark.skip("skip for now")
    @pytest.mark.parametrize("path", ["./data"])  # skip this test if you don't want to plot
    def test_angle_plot(self, path: str):
        # Build pandas dataframe for plotting
        pd_data = defaultdict(list)
        for init_scheme_seed in self.angles_data.keys():
            init_scheme = " ".join(init_scheme_seed.split("_")[:-1])
            seed = init_scheme_seed.split("_")[-1]
            for param_name, angle in self.angles_data[init_scheme_seed].items():
                distance = self.distance_data[init_scheme_seed][param_name]
                pd_data["seed"].append(seed)
                pd_data["init_scheme"].append(init_scheme)
                pd_data["param"].append(param_name)
                pd_data["angle"].append(angle)
                pd_data["distance"].append(distance)
        df = pd.DataFrame(data=pd_data)

        # Save data
        print("data:")
        print(df)
        df.to_csv(
            os.path.join(path, "data.csv"),
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


import contextlib
import io
from collections import OrderedDict
from typing import Dict, Tuple, TypeVar

from torch import Tensor, nn
from torch.nn import functional as F

from meulemans_dtp.lib.conv_network import DDTPConvNetworkCIFAR
from target_prop.networks.lenet import LeNet

T = TypeVar("T")


@contextlib.contextmanager
def disable_prints():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# from meulemans_dtp.main import Args
from target_prop.networks import Network

our_network_param_names_to_theirs = {
    LeNet: {
        "conv_0.conv.bias": "_layers.0._conv_layer.bias",
        "conv_0.conv.weight": "_layers.0._conv_layer.weight",
        "conv_1.conv.bias": "_layers.1._conv_layer.bias",
        "conv_1.conv.weight": "_layers.1._conv_layer.weight",
        "fc1.linear1.bias": "_layers.2._bias",
        "fc1.linear1.weight": "_layers.2._weights",
        "fc2.linear1.bias": "_layers.3._bias",
        "fc2.linear1.weight": "_layers.3._weights",
    }
}
their_network_param_names_to_ours: Dict[Type[Network], Dict[str, str]] = {
    model_type: {v: k for k, v in param_name_mapping.items()}
    for model_type, param_name_mapping in our_network_param_names_to_theirs.items()
}


def meulemans(
    *,
    x: Tensor,
    y: Tensor,
    network: LeNet,
    initial_weights: Dict[str, Tensor],
    network_hparams: LeNet.HParams,
    backprop_gradients: Dict[str, Tensor],
    beta: float,
    n_pretraining_iterations: int,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Minimal script to get the data of Figure 4.3 for Meuleman's DTP.

    This is to be added in the test file of `lenet_test.py`.
    """
    x = x.cuda()
    y = y.cuda()

    from meulemans_dtp import main
    from meulemans_dtp.final_configs.cifar10_DDTPConv import config as _config
    from meulemans_dtp.lib import builders, train, utils
    from meulemans_dtp.lib.conv_network import DDTPConvNetworkCIFAR

    # Double-check that the network is at it's initial state.
    _initial_weights = network.state_dict()
    for k, param in _initial_weights.items():
        assert torch.allclose(param, initial_weights[k])
    initial_network_weights = network.state_dict()

    # -- Get the arguments.
    # TODO: Eventually, use this typed class for the arg parsing instead of theirs. But for now,
    # we just use their code, but duck-typed using that Args class, to make it easier to use.
    # parser = main.add_command_line_args_v2()

    parser = main.add_command_line_args()
    # NOTE: They seem to want those to be strings, and then they convert stuff back to lists.
    config_with_strings = {k: str(v) if not isinstance(v, bool) else v for k, v in _config.items()}
    parser.set_defaults(**config_with_strings)
    with disable_prints():
        args = parser.parse_args("")
        args = main.postprocess_args(args)

    # args = typing.cast(Args, args)  # Fake cast: doesnt do anything, it's just for the type checker.

    # NOTE: Setting this, just in case they use this value somewhere I haven't seen yet.
    args.random_seed = seed
    # NOTE: Setting this to False for the LeNet equivalent network to work.
    args.freeze_BPlayers = False
    args.freeze_forward_weights = False
    # NOTE: Modifying these values so their architecture matches ours perfectly.
    args.hidden_activation = "elu"
    # NOTE: Setting beta to the same value as ours:
    args.target_stepsize = beta
    # Disable bias in their architecture if we also disable bias in ours.
    args.no_bias = not network_hparams.bias
    # Set the padding in exactly the same way as well.
    # TODO: Set this to 1 once tune-lenet is merged into master.
    DDTPConvNetworkCIFAR.pool_padding = 1
    # Set the number of iterations to match ours.
    args.nb_feedback_iterations = [n_pretraining_iterations for _ in args.nb_feedback_iterations]

    # Create their LeNet-equivalent network.
    meulemans_network = builders.build_network(args).cuda()
    assert isinstance(meulemans_network, DDTPConvNetworkCIFAR)

    # Copy over the state from our network to theirs, translating the weight names.
    missing, unexpected = meulemans_network.load_state_dict(
        translate(initial_network_weights), strict=False
    )
    assert not unexpected, f"Weights should match exactly, but got extra keys: {unexpected}."
    print(f"Arguments {missing} were be randomly initialized.")

    # Check that the two networks still give the same output for the same input, therefore that the
    # forward parameters haven't changed.
    _check_outputs_are_identical(network, meulemans_net=meulemans_network, x=x)

    meulemans_backprop_grads = get_meulemans_backprop_grads(
        meulemans_net=meulemans_network, x=x, y=y
    )
    _check_bp_grads_are_identical(
        our_backprop_grads=backprop_gradients,
        meulemans_backprop_grads=meulemans_backprop_grads,
    )

    # Q: the lrs have to be the same between the different models?
    # TODO: The network I'm using for LeNet-equivalent doesn't actually allow this to work:
    # Says "frozen blabla isn't supported with OptimizerList"
    forward_optimizer, feedback_optimizer = utils.choose_optimizer(args, meulemans_network)

    if n_pretraining_iterations > 0:
        # NOTE: Need to do the forward pass to store the activations, which are then used for feedback
        # training
        predictions = meulemans_network(x)
        train.train_feedback_parameters(
            args=args, net=meulemans_network, feedback_optimizer=feedback_optimizer
        )

    # Double-check that the forward parameters have not been updated:
    _check_forward_params_havent_moved(
        meulemans_net=meulemans_network, initial_network_weights=initial_network_weights
    )
    loss_function: nn.Module
    # Get the loss function to use (extracted from their code, was saved on train_var).
    if args.output_activation == "softmax":
        loss_function = nn.CrossEntropyLoss()
    else:
        assert args.output_activation == "sigmoid"
        loss_function = nn.MSELoss()

    # Make sure that there is nothing in the grads: delete all of them.
    meulemans_network.zero_grad(set_to_none=True)
    predictions = meulemans_network(x)
    # This propagates the targets backward, computes local forward losses, and sets the gradients
    # in the forward parameters' `grad` attribute.
    _, _ = train.train_forward_parameters(
        args,
        net=meulemans_network,
        predictions=predictions,
        targets=y,
        loss_function=loss_function,
        forward_optimizer=forward_optimizer,
    )
    assert all(p.grad is not None for _, p in _get_forward_parameters(meulemans_network).items())

    # NOTE: the values in `p.grad` are the gradients from their DTP algorithm.
    meulemans_dtp_grads = {
        # NOTE: safe to ignore, from the check above.
        name: p.grad.detach()  # type: ignore
        for name, p in _get_forward_parameters(meulemans_network).items()
    }

    # Need to rescale these by 1 / beta as well.
    scaled_meulemans_dtp_grads = {
        key: (1 / beta) * grad for key, grad in meulemans_dtp_grads.items()
    }

    distances: Dict[str, float] = {}
    angles: Dict[str, float] = {}
    with torch.no_grad():
        for name, meulemans_backprop_grad in meulemans_backprop_grads.items():
            # TODO: Do we need to scale the DRL grads like we do ours DTP?
            meulemans_dtp_grad = scaled_meulemans_dtp_grads[name]
            distance, angle = compute_dist_angle(meulemans_dtp_grad, meulemans_backprop_grad)

            distances[name] = distance
            angles[name] = angle
        # NOTE: We can actually find the parameter for these:

    return (
        translate_back(distances, network_type=LeNet),
        translate_back(angles, network_type=LeNet),
    )


def get_meulemans_backprop_grads(
    meulemans_net: DDTPConvNetworkCIFAR,
    x: Tensor,
    y: Tensor,
) -> Dict[str, Tensor]:
    """Returns the backprop gradients for the meulemans network."""
    # NOTE: Need to unfreeze the forward parameters of their network, since they apprear to be fixed

    meulemans_net_forward_params = _get_forward_parameters(meulemans_net)
    for forward_param in meulemans_net_forward_params.values():
        forward_param.requires_grad_(True)
    # NOTE: The forward pass through their network is a "regular" forward pass: Grads can flow
    # between all layers.
    predictions = meulemans_net(x)
    loss = F.cross_entropy(predictions, y)
    names = list(meulemans_net_forward_params.keys())
    parameters = [meulemans_net_forward_params[name] for name in names]
    meulemans_backprop_grads = dict(zip(names, torch.autograd.grad(loss, parameters)))
    return {k: v.detach() for k, v in meulemans_backprop_grads.items()}


def translate(dtp_values: Dict[str, T]) -> "OrderedDict[str, T]":
    """Translate our network param names to theirs."""
    return OrderedDict(
        [(our_network_param_names_to_theirs[LeNet][k], v) for k, v in dtp_values.items()]
    )


def translate_back(
    meulemans_values: Dict[str, T], network_type: Type[Network] = LeNet
) -> Dict[str, T]:
    """Translate thir network param names back to ours."""
    return {
        their_network_param_names_to_ours[network_type][k]: v for k, v in meulemans_values.items()
    }


def _get_forward_parameters(meulemans_net) -> Dict[str, Tensor]:
    # NOTE: Gets the forward weights dict programmatically.
    # The networks only return them as a list normally.
    meulemans_net_forward_params_list = meulemans_net.get_forward_parameter_list()
    # Create a dictionary of the forward parameters by finding the matching entries:
    return {
        name: param
        for name, param in meulemans_net.named_parameters()
        for forward_param in meulemans_net_forward_params_list
        if param is forward_param
    }


def _check_forward_params_havent_moved(
    meulemans_net: DDTPConvNetworkCIFAR, initial_network_weights: Dict[str, Tensor]
):
    # Translate the keys of the initial parameter dict, so we can load state dict with it:
    meulemans_net_initial_parameters = translate(initial_network_weights)
    meulemans_net_forward_params = _get_forward_parameters(meulemans_net)
    for name, parameter in meulemans_net_forward_params.items():
        assert name in meulemans_net_initial_parameters
        initial_value = meulemans_net_initial_parameters[name]
        # Make sure that the initial value wasn't somehow moved into the model, and then modified
        # by the model.
        assert parameter is not initial_value
        # Check that both are equal:
        if not torch.allclose(parameter, initial_value):
            raise RuntimeError(
                f"The forward parameter {name} was affected by the feedback training?!",
                (parameter - initial_value).mean(),
            )


def _check_outputs_are_identical(
    our_network: Network, meulemans_net: DDTPConvNetworkCIFAR, x: Tensor
):
    # Check that their network gives the same output for the same input as ours.
    # NOTE: No need to run this atm, because they use tanh, and we don't. So this won't match
    # anyway.
    rng_state = torch.random.get_rng_state()
    our_output: Tensor = our_network(x)
    torch.random.set_rng_state(rng_state)
    their_output: Tensor = meulemans_net(x)
    meulemans_forward_params = translate_back(
        _get_forward_parameters(meulemans_net), network_type=type(our_network)
    )
    assert isinstance(our_network, nn.Module)
    for param_name, param in our_network.named_parameters():
        their_param = meulemans_forward_params[param_name]
        if not torch.allclose(param, their_param):
            raise RuntimeError(
                f"Weights for param {param_name} aren't the same between our model and Meulemans', "
                f" so the output won't be the same!"
            )

    our_x = x
    their_x = x.clone()  # just to be 200% safe.

    assert len(our_network) == len(meulemans_net.layers)

    for layer_index, (our_layer, their_layer) in enumerate(zip(our_network, meulemans_net.layers)):
        assert isinstance(our_layer, nn.Module)
        our_x = our_layer(our_x)
        # In their case they also need to flatten here.
        if layer_index == meulemans_net.nb_conv:
            their_x = their_x.flatten(1)
        their_x = their_layer(their_x)

        # their_x = their_layer(their_x)
        if our_x.shape != their_x.shape:
            raise RuntimeError(
                f"Output shapes for layer {layer_index} don't match! {our_x.shape=}, "
                f"{their_x.shape=}!"
            )

        if not torch.allclose(our_x, their_x):
            # breakpoint()
            raise RuntimeError(f"Output of layers at index {layer_index} don't match!")

    if not torch.allclose(our_output, their_output):
        raise RuntimeError(
            f"The Meulamans network doesn't produce the same output as ours!\n"
            f"\t{our_output=}\n"
            f"\t{their_output=}\n"
            f"\t{(our_output - their_output).abs().sum()=}\n"
        )


def _check_bp_grads_are_identical(
    our_backprop_grads: Dict[str, Tensor],
    meulemans_backprop_grads: Dict[str, Tensor],
):
    # Compares our BP gradients and theirs: they should be identical.
    # If not, then there's probably a difference between our network and theirs (e.g. activation).
    # NOTE: Since the activation is different, we don't actually run this check.

    their_backprop_grads = translate_back(meulemans_backprop_grads)
    assert set(our_backprop_grads.keys()) == set(their_backprop_grads.keys())
    for param_name, dtp_bp_grad in our_backprop_grads.items():
        meulemans_bp_grad = their_backprop_grads[param_name]

        if meulemans_bp_grad is None:
            if dtp_bp_grad is not None:
                raise RuntimeError(
                    f"Meulemans DTP doesn't have a backprop grad for param {param_name}, but our "
                    f"DTP model has one!"
                )
            continue

        assert meulemans_bp_grad.shape == dtp_bp_grad.shape
        if not torch.allclose(dtp_bp_grad, meulemans_bp_grad):
            raise RuntimeError(
                f"Backprop gradients for parameter {param_name} aren't the same as ours!\n"
                f"\tTheir backprop gradient:\n"
                f"\t{dtp_bp_grad}\n"
                f"\tTheir backprop gradient:\n"
                f"\t{meulemans_bp_grad}\n"
                f"\tTotal absolute difference:\n"
                f"\t{(dtp_bp_grad - meulemans_bp_grad).abs().sum()=}\n"
            )
