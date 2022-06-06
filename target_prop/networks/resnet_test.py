from typing import TypeVar

import pytest
import torch
from torch import Tensor, nn

from .network_test import NetworkTests
from .resnet import ResNet, ResNet18, ResNet34

ResNetType = TypeVar("ResNetType", bound=ResNet)


class ResNetTests(NetworkTests[ResNetType]):
    @torch.no_grad()
    @pytest.mark.xfail(reason="TODO: Symetric init doesn't reduce resnets reconstruction loss atm.")
    def test_init_symmetric_weights_reduces_reconstruction_error(
        self, network: ResNetType, pseudoinverse: nn.Module, x: Tensor, y_pred: Tensor
    ):
        super().test_init_symmetric_weights_reduces_reconstruction_error(
            network, pseudoinverse, x, y_pred
        )


class TestResNet18(ResNetTests[ResNet18]):
    net_type = ResNet18


class TestResNet34(ResNetTests[ResNet34]):
    net_type = ResNet34
