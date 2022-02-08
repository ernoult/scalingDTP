import os
from dataclasses import dataclass, field
from logging import getLogger as get_logger
from typing import Any, Dict, List, Optional, Type
import logging
import wandb

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything

from target_prop.config import Config
from target_prop.datasets.dataset_config import DatasetConfig
from target_prop.models import *
from target_prop.models.model import Model
from target_prop.networks import *
from target_prop.networks.network import Network
from target_prop.scheduler_config import CosineAnnealingLRConfig, StepLRConfig
from target_prop.utils.hydra_utils import LoadableFromHydra, get_outer_class

logger = get_logger(__name__)


@dataclass
class Options(LoadableFromHydra):
    """ All the options required for a run. This dataclass acts as a schema for the Hydra configs.
    
    For more info, see https://hydra.cc/docs/tutorials/structured_config/schema/
    """

    # Configuration for the dataset + transforms.
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # The model used.
    model: Model.HParams = field(default_factory=DTP.HParams)
    # The network to be used.
    network: Network.HParams = field(default_factory=SimpleVGG.HParams)

    # Keyword arguments for the Trainer constructor.
    trainer: Dict = field(default_factory=dict)

    # Configs for the callbacks.
    callbacks: Dict = field(default_factory=dict)

    # Config(s) for the logger(s).
    logger: Dict = field(default_factory=dict)

    # Wether to run in debug mode or not.
    debug: bool = False

    verbose: bool = False

    # Random seed.
    seed: Optional[int] = None

    name: str = ""


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
def main(raw_options: DictConfig) -> None:
    print(os.getcwd())
    print(OmegaConf.to_yaml(raw_options))

    options = Options.from_dictconfig(raw_options)

    model_hparams: Model.HParams = options.model
    model_type: Type[Model] = get_outer_class(type(options.model))
    network_hparams: Network.HParams = options.network
    network_type: Type[Network] = get_outer_class(type(options.network))

    dataset: DatasetConfig = options.dataset

    # NOTE: In the process of moving the args from `Config` to this top-level `Options` class.
    config: Config = Config(seed=options.seed, debug=options.debug)

    if options.seed is not None:
        seed_everything(seed=options.seed, workers=True)
    model_hparams = model_hparams

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    assert isinstance(model_hparams, model_type.HParams), "HParams type should match model type"

    # Create the datamodule:
    datamodule = dataset.make_datamodule(batch_size=model_hparams.batch_size)

    # Create the network
    network: Network = network_type(
        in_channels=datamodule.dims[0],
        n_classes=datamodule.num_classes,  # type: ignore
        hparams=network_hparams,
    )

    # Create the model
    model: Model = model_type(
        network=network,
        datamodule=datamodule,
        hparams=model_hparams,
        network_hparams=network_hparams,
        config=config,
    )
    assert isinstance(model, LightningModule)

    # Create the callbacks
    callbacks: List[Callback] = [
        hydra.utils.instantiate(callback_dict) for callback_dict in options.callbacks.values()
    ]
    from pytorch_lightning.loggers import LightningLoggerBase

    # --- Create the trainer.. ---

    trainer_kwargs = options.trainer

    if options.debug:
        logger.info(f"Setting the max_epochs to 1, since the 'debug' flag was passed.")
        trainer_kwargs["max_epochs"] = 1

    # Create the logger(s):
    loggers: List[LightningLoggerBase] = [
        hydra.utils.instantiate(logger_dict) for logger_dict in options.logger.values()
    ]

    if options.verbose:
        root_logger.setLevel(logging.DEBUG)

    trainer: Trainer = hydra.utils.instantiate(trainer_kwargs, callbacks=callbacks, logger=loggers)

    # --- Run the experiment. ---
    trainer.fit(model, datamodule=datamodule)

    val_results = trainer.validate(model=model, datamodule=datamodule)
    assert len(val_results) == 1
    top1_accuracy: float = val_results[0]["val/accuracy"]
    top5_accuracy: float = val_results[0]["val/top5_accuracy"]
    print(f"Validation top1 accuracy: {top1_accuracy:.1%}")
    print(f"Validation top5 accuracy: {top5_accuracy:.1%}")

    from orion.client import report_objective

    test_error = 1 - top1_accuracy
    report_objective(test_error)

    if wandb.run:
        wandb.finish()

    # Run on the test set:
    # test_results = trainer.test(model, datamodule=datamodule, verbose=True)
    # top1_accuracy: float = test_results[0]["test/accuracy"]
    # top5_accuracy: float = test_results[0]["test/top5_accuracy"]
    # print(f"Test top1 accuracy: {top1_accuracy:.1%}")
    # print(f"Test top5 accuracy: {top5_accuracy:.1%}")


if __name__ == "__main__":
    main()
