from typing import Type
import hydra
from omegaconf import DictConfig
from target_prop.models.model import Model
from target_prop.networks.network import Network
import os
from logging import getLogger as get_logger

from target_prop.utils.hydra_utils import get_outer_class
from target_prop.scheduler_config import (
    CosineAnnealingLRConfig,
    StepLRConfig,
)
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from target_prop.models import *
from target_prop.networks import *
from target_prop.utils.hydra_utils import LoadableFromHydra
from target_prop.config import Config

logger = get_logger(__name__)


@dataclass
class Options(LoadableFromHydra):
    """ All the options required for a run. This dataclass acts as a schema for the Hydra configs.
    
    For more info, see https://hydra.cc/docs/tutorials/structured_config/schema/
    """

    # The model used.
    model: Model.HParams = field(default_factory=DTP.HParams)
    # The network to be used.
    network: Network.HParams = field(default_factory=SimpleVGG.HParams)

    # Other configuration options.
    config: Config = field(default_factory=Config)


# TODO: Use the __subclasses__ of the types to find all the "structured configs" we know of.

cs = ConfigStore.instance()
cs.store(name="base_options", node=Options)

DTP.HParams.cs_store(group="model", name="dtp")
ParallelDTP.HParams.cs_store(group="model", name="parallel_dtp")
VanillaDTP.HParams.cs_store(group="model", name="vanilla_dtp")
TargetProp.HParams.cs_store(group="model", name="target_prop")

SimpleVGG.HParams.cs_store(group="network", name="simple_vgg")
LeNet.HParams.cs_store(group="network", name="lenet")
ResNet18.HParams.cs_store(group="network", name="resnet18")
ResNet34.HParams.cs_store(group="network", name="resnet34")

StepLRConfig.cs_store(group="lr_scheduler", name="step")
CosineAnnealingLRConfig.cs_store(group="lr_scheduler", name="cosine")

# Config.cs_store(group="config", name="base_config", default=Config())

# TODO: Figure out a way to save / extract the type that was passed to `node`, so that we can
# actually construct the right type of dataclass after-the-fact.
# IDEA: add a "_type" key in to_yaml, that we then eval to get the value!
# assert False, cs.repo["network"]


@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    print(os.getcwd())
    options = Options.from_dictconfig(config)
    # print(options.dumps_json(indent="\t"))
    print(options.model)
    print(options.network)
    print(options.config)
    print(options.model.lr_scheduler)

    from main_pl import run

    model_hparams: Model.HParams = options.model
    model_type: Type[Model] = get_outer_class(type(options.model))
    network_hparams: Network.HParams = options.network
    network_type: Type[Network] = get_outer_class(type(options.network))

    top1, top5 = run(
        config=options.config,
        model_type=model_type,
        hparams=model_hparams,
        network_type=network_type,
        network_hparams=network_hparams,
    )


if __name__ == "__main__":
    main()
