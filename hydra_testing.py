import importlib
from typing import Any, ClassVar, Dict, Optional, Type
import hydra
from omegaconf import DictConfig, OmegaConf
from simple_parsing.helpers import choice
from simple_parsing.helpers.serialization.serializable import Serializable
from target_prop.models import Model
from target_prop.networks import Network
import os
from hydra.utils import get_class
from logging import getLogger as get_logger

from target_prop.scheduler_config import (
    CosineAnnealingLRConfig,
    LRSchedulerConfig,
    StepLRConfig,
)

logger = get_logger(__name__)

from dataclasses import dataclass, field


import hydra
from hydra.core.config_store import ConfigStore
from target_prop.models import *
from target_prop.networks import *


# NOTE: Doesn't work with Union types, it feels a bit wonky tbh since it's just for duck-typing
# purposes.

from abc import ABC
from dataclasses import MISSING


# NOTE: Not entirely sure I understand what this is for.
from target_prop.utils.hydra_utils import LoadableFromHydra


@dataclass
class Options(LoadableFromHydra):
    # The model used.
    model: Model.HParams = field(default_factory=DTP.HParams)
    # The network to be used.
    network: Network.HParams = field(default_factory=SimpleVGG.HParams)
    # Type of learning rate scheduler.
    # lr_scheduler: Optional[LRSchedulerConfig] = field(default_factory=CosineAnnealingLRConfig)
    dataset: str = choice("mnist", "cifar10", default="cifar10")


# TODO: Use the __subclasses__ of the types to find all the "structured configs" we know of.


cs = ConfigStore.instance()
cs.store(name="base_config", node=Options)

Model.HParams.cs_store(group="model", name="base_model")

DTP.HParams.cs_store(group="model", name="dtp")
ParallelDTP.HParams.cs_store(group="model", name="parallel_dtp")
VanillaDTP.HParams.cs_store(group="model", name="vanilla_dtp")
TargetProp.HParams.cs_store(group="model", name="target_prop")

SimpleVGG.HParams.cs_store(group="network", name="simple_vgg")
LeNet.HParams.cs_store(group="network", name="lenet")
ResNet18.HParams.cs_store(group="network", name="resnet18")
ResNet34.HParams.cs_store(group="network", name="resnet34")

LRSchedulerConfig.cs_store(group="lr_scheduler", name="scheduler")
StepLRConfig.cs_store(group="lr_scheduler", name="step")
CosineAnnealingLRConfig.cs_store(group="lr_scheduler", name="cosine")


# DTP.HParams().save_yaml("conf/model/dtp.yaml")
# ParallelDTP.HParams().save_yaml("conf/model/parallel_dtp.yaml")
# SimpleVGG.HParams().save_yaml("conf/network/simple_vgg.yaml")
# CosineAnnealingLRConfig().save_yaml("conf/lr_scheduler/cosine.yaml")

# TODO: Figure out a way to save / extract the type that was passed to `node`, so that we can
# actually construct the right type of dataclass after-the-fact.
# IDEA: add a "_type" key in to_yaml, that we then eval to get the value!
# assert False, cs.repo["network"]


@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    print(os.getcwd())
    options = Options.from_dictconfig(config)
    print(options.model)
    return
    options: Options = Options.from_dict(args_dict)
    print(options)
    model_hparams = options.model
    network_hparams = options.network
    print(model_hparams)
    print(network_hparams)
    # lr_scheduler
    # model_hparams: Model.HParams = Model.HParams.from_dict(args_dict["model"])
    # network_hparams: Network.HParams = Network.HParams.from_dict(args_dict["network"])
    # lr_scheduler: LRSchedulerConfig = LRSchedulerConfig.from_dict(args_dict["lr_scheduler"])
    # logger.info(f"model hparams: {model_hparams}")
    # logger.info(f"network hparams: {network_hparams}")


if __name__ == "__main__":
    main()
