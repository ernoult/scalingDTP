from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

from simple_parsing.helpers import choice, list_field
from simple_parsing.helpers.hparams.hparam import categorical, log_uniform, uniform
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from target_prop.layers import MaxPool2d, Reshape
from torch import nn


class SimpleVGG(nn.Sequential):
    @dataclass
    class HParams(HyperParameters):
        channels: List[int] = list_field(128, 128, 256, 256, 512)
        activation: Type[nn.Module] = choice(
            {"relu": nn.ReLU, "elu": nn.ELU,}, default=nn.ELU,
        )

    def __init__(self, in_channels: int, n_classes: int, hparams: "SimpleVGG.HParams" = None):
        hparams = hparams or self.HParams()
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        activation: Type[nn.Module] = hparams.activation

        channels = [in_channels] + hparams.channels
        # NOTE: Can use [0:] and [1:] below because zip will stop when the shortest
        # iterable is exhausted. This gives us the right number of blocks.
        for i, (in_channels, out_channels) in enumerate(zip(channels[0:], channels[1:])):
            block = nn.Sequential(
                OrderedDict(
                    conv=nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,),
                    rho=activation(),
                    # NOTE: Even though `return_indices` is `False` here, we're actually passing
                    # the indices to the backward net for this layer through a "magic bridge".
                    # We use `return_indices=False` here just so the layer doesn't also return
                    # the indices in its forward pass.
                    pool=MaxPool2d(kernel_size=2, stride=2, return_indices=False),
                    # NOTE: Would be nice to use AvgPool, seems more "plausible" and less hacky.
                    # pool=nn.AvgPool2d(kernel_size=2),
                )
            )
            layers[f"conv_{i}"] = block
        layers["fc"] = nn.Sequential(
            OrderedDict(
                reshape=Reshape(target_shape=(-1,)),
                linear=nn.LazyLinear(out_features=n_classes, bias=True),
            )
        )
        super().__init__(layers)
        self.hparams = hparams


simple_vgg = SimpleVGG
SimpleVGGHparams = SimpleVGG.HParams


class SimpleVGG2(nn.Sequential):
    """
    Simple VGG without pooling layer in the first block
    """
    @dataclass
    class HParams(HyperParameters):
        channels: List[int] = list_field(128, 128, 256, 256, 512)
        activation: Type[nn.Module] = choice(
            {"relu": nn.ReLU, "elu": nn.ELU,}, default=nn.ELU,
        )

    def __init__(self, in_channels: int, n_classes: int, hparams: "SimpleVGG.HParams" = None):
        hparams = hparams or self.HParams()
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        activation: Type[nn.Module] = hparams.activation

        channels = [in_channels] + hparams.channels
        # NOTE: Can use [0:] and [1:] below because zip will stop when the shortest
        # iterable is exhausted. This gives us the right number of blocks.
        for i, (in_channels, out_channels) in enumerate(zip(channels[0:], channels[1:])):
            block = nn.Sequential(
                OrderedDict(
                    conv=nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,),
                    rho=activation(),
                    # NOTE: Even though `return_indices` is `False` here, we're actually passing
                    # the indices to the backward net for this layer through a "magic bridge".
                    # We use `return_indices=False` here just so the layer doesn't also return
                    # the indices in its forward pass.
                    pool=MaxPool2d(kernel_size=2, stride=2, return_indices=False) if i > 0 else nn.Identity(),
                    # NOTE: Would be nice to use AvgPool, seems more "plausible" and less hacky.
                    # pool=nn.AvgPool2d(kernel_size=2),
                )
            )
            layers[f"conv_{i}"] = block
        layers["fc"] = nn.Sequential(
            OrderedDict(
                reshape=Reshape(target_shape=(-1,)),
                linear=nn.LazyLinear(out_features=n_classes, bias=True),
            )
        )
        super().__init__(layers)
        self.hparams = hparams

simple_vgg2 = SimpleVGG2
SimpleVGG2Hparams = SimpleVGG2.HParams