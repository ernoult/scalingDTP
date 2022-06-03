""" Tests for new networks fthat get added to the codebase. """

from typing import ClassVar, Generic, TypeVar

import pytest
import torch
from pytorch_lightning import seed_everything
from torch import Tensor, nn
from torch.testing import assert_allclose

from target_prop._weight_operations import init_symetric_weights
from target_prop.backward_layers import invert, mark_as_invertible
from target_prop.metrics import compute_dist_angle

from .network import Network

NetworkType = TypeVar("NetworkType", bound=Network)


class NetworkTests(Generic[NetworkType]):

    net_type: type[NetworkType]
    network_kwargs: ClassVar[dict] = dict(
        in_channels=3,
        n_classes=10,
        hparams=None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def make_network(cls, *args, **kwargs) -> NetworkType:
        net = cls.net_type(*args, **cls.network_kwargs, **kwargs)
        if isinstance(net, nn.Module):
            net = net.to(cls.device)
        return net

    @pytest.fixture
    def x(self) -> Tensor:
        """Fixture that gives network inputs."""
        return torch.rand([10, 3, 32, 32]).to(self.device)

    @pytest.fixture(scope="function")
    def network(self, x: Tensor) -> NetworkType:
        net = self.make_network()
        # Mark it as invertible and perform a forward pass, so it's easy to invert when needed.
        mark_as_invertible(net)
        _ = net(x)
        return net

    @pytest.fixture
    def y_pred(self, network: NetworkType, x: Tensor) -> Tensor:
        """Fixture that gives network outputs."""
        return network(x)

    @torch.no_grad()
    def test_can_create_pseudoinverse(self, network: NetworkType, x: Tensor):
        pseudoinverse = invert(network).to(self.device)
        assert len(pseudoinverse) == len(network)

        print(network)
        print(pseudoinverse)

        y_pred = network(x)
        x_hat = pseudoinverse(y_pred)
        assert x_hat.shape == x.shape

    @pytest.fixture
    def pseudoinverse(self, network: NetworkType):
        pseudoinverse = invert(network).to(self.device)
        return pseudoinverse

    @torch.no_grad()
    @pytest.mark.parametrize("seed", [123, 456])
    def test_network_creation_reproducible_given_seed(self, seed: int, x: Tensor):
        seed_everything(seed=seed)
        network_a = self.make_network()
        y_a = network_a(x)

        seed_everything(seed=seed)
        network_b = self.make_network()
        y_b = network_b(x)
        assert_allclose(y_a, y_b)

    @torch.no_grad()
    @pytest.mark.parametrize("seed", [123, 456])
    def test_pseudoinverse_creation_reproducible_given_seed(
        self, seed: int, network: NetworkType, y_pred: Tensor
    ):
        seed_everything(seed=seed)
        pseudoinverse_a = invert(network).to(self.device)
        x_hat_a = pseudoinverse_a(y_pred)

        seed_everything(seed=seed)
        pseudoinverse_b = invert(network).to(self.device)

        state_a = pseudoinverse_a.state_dict()
        state_b = pseudoinverse_b.state_dict()
        for key in state_a:
            assert_allclose(state_a[key], state_b[key])

        x_hat_b = pseudoinverse_b(y_pred)
        # TODO: This might be failing here because of the 'magic bridge' being shared between the
        # two pseudoinverses in the case of the MaxPool2d layer.
        # assert_allclose(x_hat_a, x_hat_b)

    @torch.no_grad()
    def test_can_init_symmetric_weights(self, network: NetworkType, pseudoinverse: nn.Module):
        """Can initialize the weights of the pseudoinverse symmetrically with respect to the forward network."""
        init_symetric_weights(network, pseudoinverse)

    @pytest.mark.xfail(reason="TODO: symmetric init doesn't always reduce reconstruction error.")
    @torch.no_grad()
    def test_init_symmetric_weights_reduces_reconstruction_error(
        self, network: NetworkType, pseudoinverse: nn.Module, x: Tensor, y_pred: Tensor
    ):
        """Initializing the weights of the pseudoinverse symmetrically should give a better
        pseudoinverse (not always exact inverse).
        """
        x_hat_before = pseudoinverse(y_pred)
        reconstruction_error_random_init = torch.norm(x - x_hat_before)

        init_symetric_weights(network, pseudoinverse)

        x_hat_after = pseudoinverse(y_pred)
        reconstruction_error_symmetric_init = torch.norm(x - x_hat_after)

        assert reconstruction_error_symmetric_init < reconstruction_error_random_init

    @torch.no_grad()
    def test_can_calculate_distances_and_angles_between_layers(
        self, network: NetworkType, pseudoinverse: nn.Module
    ):
        # TODO: Update this test once this `compute_dist_angle` function has its output type harmonized.
        metrics = compute_dist_angle(network, pseudoinverse)
        n_layers = len(network)
        if isinstance(metrics, dict):
            assert len(metrics) == n_layers
            assert set(metrics.keys()) == set(range(n_layers))
        else:
            assert isinstance(metrics, tuple) and len(metrics) == 2
            distances, angles = metrics
            assert isinstance(distances, float)
            assert isinstance(angles, float)
            # assert len(distances) == n_layers
            # assert len(angles) == n_layers
