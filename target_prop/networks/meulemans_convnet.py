from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from torch import Tensor, nn

from meulemans_dtp.lib.conv_network import DDTPConvNetworkCIFAR
from meulemans_dtp.lib.direct_feedback_layers import DDTPMLPLayer

from .network import Network


class MeulemansConvNet(DDTPConvNetworkCIFAR, Network):
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
        self, in_channels: int, n_classes: int, hparams: MeulemansConvNet.HParams | None = None
    ):
        assert in_channels == 3
        assert n_classes == 10
        if not hparams:
            hparams = self.HParams()
        self.hparams: MeulemansConvNet.HParams = hparams
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
