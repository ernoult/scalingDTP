from collections import OrderedDict

from target_prop.layers import MaxPool2d, Reshape
from torch import nn

available_activations = {"elu": nn.ELU, "relu": nn.ReLU}


def SimpleVGG(
    in_channels=3, n_classes=10, channels=[128, 128, 256, 256, 512], activation_type="elu"
):
    layers: OrderedDict[str, nn.Module] = OrderedDict()
    activation = available_activations[activation_type]

    channels = [in_channels] + channels
    # NOTE: Can use [0:] and [1:] below because zip will stop when the shortest
    # iterable is exhausted. This gives us the right number of blocks.
    for i, (in_channels, out_channels) in enumerate(zip(channels[0:], channels[1:])):
        block = nn.Sequential(
            OrderedDict(
                conv=nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
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
    return nn.Sequential(layers)
