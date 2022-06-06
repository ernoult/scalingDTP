from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple, Type

import torch.nn.functional as F
from simple_parsing.helpers import list_field
from torch import Tensor, nn

from target_prop.backward_layers import invert
from target_prop.layers import Reshape

from .network import Network


class Residual(nn.Module):
    def __init__(self, fn: nn.Module, shortcut: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.fn = fn
        self.shortcut = shortcut or nn.Sequential()

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + self.shortcut(x)


class BasicBlock(nn.Module):
    """
    Basic residual block with optional BatchNorm.
    """

    expansion: ClassVar[int] = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, use_batchnorm: bool = False):
        super().__init__()
        # Save hyperparams relevant for inversion
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.use_batchnorm = use_batchnorm

        # Initialize layers
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes) if self.use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_batchnorm else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                OrderedDict(
                    conv=nn.Conv2d(
                        in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                    ),
                    bn=(
                        nn.BatchNorm2d(self.expansion * planes)
                        if self.use_batchnorm
                        else nn.Identity()
                    ),
                )
            )

        # IDEA: Could maybe simplify the code a little bit? However it would make accessing the
        # weights a bit more indirect..
        # self.residual = nn.Sequential(
        #     Residual(
        #         nn.Sequential(self.conv1, self.bn1, nn.ReLU(), self.conv2, self.bn2),
        #         shortcut=self.shortcut,
        #     ),
        #     nn.ReLU(),
        # )

    def forward(self, x: Tensor) -> Tensor:
        # out = F.relu(self.residual(x))

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class InvertedBasicBlock(nn.Module):
    """
    Implements basic block that mimics residual operation in inverted manner.

    Original residual forward pass:
    x -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> + -> relu -> out
    |                                            |
    |------------ conv -> bn --------------------|

    Inverted residual forward pass:
    x -> relu -> bn2 -> conv2_t -> relu -> bn1 -> conv1_t -> + -> out
           |                                                 |
           |--------------- bn -> conv_t --------------------|
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=False):
        super().__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.use_batchnorm = use_batchnorm

        # Create inverted layers
        self.conv1 = invert(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        )
        self.bn1 = invert(nn.BatchNorm2d(planes) if self.use_batchnorm else nn.Identity())
        self.conv2 = invert(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.bn2 = invert(nn.BatchNorm2d(planes) if self.use_batchnorm else nn.Identity())

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                OrderedDict(
                    conv=nn.Conv2d(
                        in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                    ),
                    bn=nn.BatchNorm2d(self.expansion * planes)
                    if self.use_batchnorm
                    else nn.Identity(),
                )
            )
        self.shortcut = invert(self.shortcut)

    def forward(self, x):
        x = F.relu(x)
        out = F.relu(self.conv2(self.bn2(x)))
        out = self.conv1(self.bn1(out))
        out += self.shortcut(x)
        return out


@invert.register(BasicBlock)
def invert_basic(module: BasicBlock) -> InvertedBasicBlock:
    backward = InvertedBasicBlock(
        in_planes=module.in_planes,
        planes=module.planes,
        stride=module.stride,
        use_batchnorm=module.use_batchnorm,
    )
    return backward


def make_layer(
    block: Type[BasicBlock],
    planes: int,
    num_blocks: int,
    stride: int,
    in_planes: int,
    use_batchnorm: bool,
) -> Tuple[nn.Sequential, int]:
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(block(in_planes, planes, stride, use_batchnorm))
        in_planes = planes * block.expansion
    return nn.Sequential(*layers), in_planes


class ResNet(nn.Sequential, Network):
    """
    ResNet with optional BatchNorm.
    Adapted from PyTorch ResNet:
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

    NOTE: This ResNet differs from standard ResNet in input layer. Here, we adopt ResNet
    from pytorch-cifar repo linked above that uses conv layer with kernel size 3 to process
    32x32 input whereas standard ResNet uses conv layer with kernel size 7 followed by maxpool
    which is better suited for larger image sizes like 224x224.
    """

    @dataclass
    class HParams(Network.HParams):
        use_batchnorm: bool = False
        num_blocks: List[int] = list_field(2, 2, 2, 2)

    def __init__(self, in_channels: int, n_classes: int, hparams: ResNet.HParams | None = None):
        # Catch hparams
        hparams = hparams or self.HParams()
        use_batchnorm = hparams.use_batchnorm
        block_type = BasicBlock
        num_blocks = hparams.num_blocks

        # Build ResNet
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        layers["layer_0"] = nn.Sequential(
            OrderedDict(
                conv=nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
                bn=nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
                rho=nn.ReLU(),
            )
        )

        in_planes = 64
        layers["layer_1"], in_planes = make_layer(
            block_type,
            64,
            num_blocks[0],
            stride=1,
            in_planes=in_planes,
            use_batchnorm=use_batchnorm,
        )
        layers["layer_2"], in_planes = make_layer(
            block_type,
            128,
            num_blocks[1],
            stride=2,
            in_planes=in_planes,
            use_batchnorm=use_batchnorm,
        )
        layers["layer_3"], in_planes = make_layer(
            block_type,
            256,
            num_blocks[2],
            stride=2,
            in_planes=in_planes,
            use_batchnorm=use_batchnorm,
        )
        layers["layer_4"], in_planes = make_layer(
            block_type,
            512,
            num_blocks[3],
            stride=2,
            in_planes=in_planes,
            use_batchnorm=use_batchnorm,
        )
        layers["fc"] = nn.Sequential(
            OrderedDict(
                pool=nn.AdaptiveAvgPool2d(
                    output_size=(1, 1)
                ),  # NOTE: This is specific for 32x32 input!
                reshape=Reshape(target_shape=(-1,)),
                linear=nn.LazyLinear(out_features=n_classes, bias=True),
            )
        )
        super().__init__(layers)
        self.hparams = hparams


class ResNet18(ResNet):
    @dataclass
    class HParams(ResNet.HParams):
        use_batchnorm: bool = False
        num_blocks: List[int] = list_field(2, 2, 2, 2)


class ResNet34(ResNet):
    @dataclass
    class HParams(ResNet.HParams):
        use_batchnorm: bool = False
        num_blocks: List[int] = list_field(3, 4, 6, 3)


resnet = ResNet
ResNet18Hparams = ResNet18.HParams
ResNet34Hparams = ResNet34.HParams
