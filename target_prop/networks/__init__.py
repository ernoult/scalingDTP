from .resnet import ResNet18Hparams, ResNet34Hparams, resnet, ResNet, ResNet18, ResNet34
from .simple_vgg import SimpleVGGHparams, simple_vgg, SimpleVGG
from typing import Union

Network = Union[ResNet18, ResNet34, SimpleVGG]
