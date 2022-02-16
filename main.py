import os
from dataclasses import dataclass, field
from logging import getLogger as get_logger
from typing import Any, Dict, List, Optional, Type
import logging
from simple_parsing.helpers.serialization.serializable import Serializable
import wandb
from pytorch_lightning.loggers import LightningLoggerBase

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
from target_prop.utils.hydra_utils import get_outer_class

logger = get_logger(__name__)


@dataclass
class Options(Serializable):
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
    trainer: Any = field(default_factory=dict)  # type: ignore

    # Configs for the callbacks.
    callbacks: Any = field(default_factory=dict)  # type: ignore

    # Config(s) for the logger(s).
    logger: Any = field(default_factory=dict)  # type: ignore

    # Wether to run in debug mode or not.
    debug: bool = False

    verbose: bool = False

    # Random seed.
    seed: Optional[int] = None

    # Name for the experiment.
    name: str = ""

    def __post_init__(self):
        actual_callbacks: Dict[str, Callback] = {}

        # NOTE: Need to do a bit of sneaky type tricks to convince the outside world that these
        # fields have the right type.

        # Create the callbacks
        callbacks = self.callbacks
        if isinstance(self.callbacks, dict):
            for name, callback in self.callbacks.items():
                if isinstance(callback, dict):
                    callback = hydra.utils.instantiate(callback)
                elif not isinstance(callback, Callback):
                    raise ValueError(f"Invalid callback value {callback}")
                actual_callbacks[name] = callback
            callbacks = list(actual_callbacks.values())
        self.callbacks: List[Callback] = callbacks

        # Create the loggers, if any.
        loggers = self.logger
        if isinstance(self.logger, dict):
            actual_loggers: Dict[str, LightningLoggerBase] = {}
            for name, lightning_logger in self.logger.items():
                if isinstance(lightning_logger, dict):
                    lightning_logger = hydra.utils.instantiate(lightning_logger)
                elif not isinstance(lightning_logger, LightningLoggerBase):
                    raise ValueError(f"Invalid logger value {lightning_logger}")
            loggers = list(actual_loggers.values())
        self.logger: List[LightningLoggerBase] = loggers

        # Create the Trainer.
        trainer = self.trainer
        if isinstance(self.trainer, dict):
            if self.debug:
                logger.info(f"Setting the max_epochs to 1, since the 'debug' flag was passed.")
                self.trainer["max_epochs"] = 1
            trainer = hydra.utils.instantiate(
                self.trainer, callbacks=self.callbacks, logger=self.logger,
            )
        elif not isinstance(self.trainer, Trainer):
            raise ValueError(f"Invalid Trainer value {self.trainer}")
        self.trainer: Trainer = trainer


cs = ConfigStore.instance()
cs.store(name="base_options", node=Options)

cs.store(group="model", name="dtp", node=DTP.HParams())
cs.store(group="model", name="parallel_dtp", node=ParallelDTP.HParams())
cs.store(group="model", name="vanilla_dtp", node=VanillaDTP.HParams())
cs.store(group="model", name="target_prop", node=TargetProp.HParams())

cs.store(group="network", name="simple_vgg", node=SimpleVGG.HParams())
cs.store(group="network", name="lenet", node=LeNet.HParams())
cs.store(group="network", name="resnet18", node=ResNet18.HParams())
cs.store(group="network", name="resnet34", node=ResNet34.HParams())

cs.store(group="lr_scheduler", name="step", node=StepLRConfig)
cs.store(group="lr_scheduler", name="cosine", node=CosineAnnealingLRConfig)


@hydra.main(config_path="conf", config_name="config")
def main(raw_options: DictConfig) -> None:
    print(os.getcwd())
    options = OmegaConf.to_object(raw_options)
    assert isinstance(options, Options)
    run(options=options)


def run(options: Options):
    if options.seed is not None:
        seed_everything(seed=options.seed, workers=True)

    root_logger = logging.getLogger()
    if options.debug:
        root_logger.setLevel(logging.INFO)
    elif options.verbose:
        root_logger.setLevel(logging.DEBUG)

    # Create the datamodule:
    dataset: DatasetConfig = options.dataset
    datamodule = dataset.make_datamodule(batch_size=options.model.batch_size)

    # Create the network
    network_hparams: Network.HParams = options.network
    network_type: Type[Network] = get_outer_class(type(options.network))
    assert isinstance(
        network_hparams, network_type.HParams
    ), "HParams type should match parent type"
    network: Network = network_type(
        in_channels=datamodule.dims[0],
        n_classes=datamodule.num_classes,  # type: ignore
        hparams=network_hparams,
    )

    # Create the model
    model_hparams: Model.HParams = options.model
    model_type: Type[Model] = get_outer_class(type(options.model))
    assert isinstance(model_hparams, model_type.HParams), "HParams type should match model type"
    model: Model = model_type(
        network=network,
        datamodule=datamodule,
        hparams=model_hparams,
        network_hparams=network_hparams,
        config=Config(seed=options.seed, debug=options.debug),
    )
    assert isinstance(model, LightningModule)
    trainer = options.trainer

    root_logger = logging.getLogger()
    # --- Run the experiment. ---
    trainer.fit(model, datamodule=datamodule)

    val_results = trainer.validate(model=model, datamodule=datamodule)
    assert len(val_results) == 1
    top1_accuracy: float = val_results[0]["val/accuracy"]
    top5_accuracy: float = val_results[0]["val/top5_accuracy"]
    print(f"Validation top1 accuracy: {top1_accuracy:.1%}")
    print(f"Validation top5 accuracy: {top5_accuracy:.1%}")

    if not options.debug:
        from orion.client import report_objective

        test_error = 1 - top1_accuracy
        report_objective(test_error)

    if wandb.run:
        wandb.finish()

    # TODO: Enable this later.
    # Run on the test set:
    # test_results = trainer.test(model, datamodule=datamodule, verbose=True)
    # top1_accuracy: float = test_results[0]["test/accuracy"]
    # top5_accuracy: float = test_results[0]["test/top5_accuracy"]
    # print(f"Test top1 accuracy: {top1_accuracy:.1%}")
    # print(f"Test top5 accuracy: {top5_accuracy:.1%}")


if __name__ == "__main__":
    main()
