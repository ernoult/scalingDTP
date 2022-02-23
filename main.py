import os
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Any, Dict, List, Optional, Type
import logging
from hydra_zen import get_target
from simple_parsing.helpers.serialization.serializable import Serializable
from simple_parsing.helpers import field
import wandb
from pytorch_lightning.loggers import LightningLoggerBase
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
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
    dataset: DatasetConfig  # = field(default_factory=DatasetConfig)

    # The model used.
    model: Model.HParams  # = field(default_factory=Model.HParams)
    # The network to be used.
    network: Network.HParams  # = field(default_factory=SimpleVGG.HParams)

    # Keyword arguments for the Trainer constructor.
    trainer: Dict = field(default_factory=dict)  # type: ignore

    # Configs for the callbacks.
    callbacks: Dict = field(default_factory=dict)  # type: ignore

    # Config(s) for the logger(s).
    logger: Dict = field(default_factory=dict)  # type: ignore

    # Wether to run in debug mode or not.
    debug: bool = False

    verbose: bool = False

    # Random seed.
    seed: Optional[int] = None

    # Name for the experiment.
    name: str = ""


cs = ConfigStore.instance()
cs.store(name="base_options", node=Options)

cs.store(group="model", name="model", node=Model.HParams())
cs.store(group="model", name="dtp", node=DTP.HParams())
cs.store(group="model", name="parallel_dtp", node=ParallelDTP.HParams())
cs.store(group="model", name="vanilla_dtp", node=VanillaDTP.HParams())
cs.store(group="model", name="target_prop", node=TargetProp.HParams())
cs.store(group="model", name="backprop", node=BaselineModel.HParams())

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
    experiment = Experiment(options)
    assert isinstance(options, Options)
    # run(options=options)

    experiment.run()


from pl_bolts.datamodules.vision_datamodule import VisionDataModule


@dataclass
class Experiment(Serializable):
    """ Experiment class. Created from the Options that are parsed from Hydra. Can be used to run
    the experiment.
    """

    options: Options
    trainer: Trainer = field(init=False, to_dict=False)
    model: Model = field(init=False, to_dict=False)
    network: Network = field(init=False, to_dict=False)
    datamodule: VisionDataModule = field(init=False, to_dict=False)

    callbacks: List[Callback] = field(init=False, default_factory=list, to_dict=False)
    loggers: List[LightningLoggerBase] = field(init=False, default_factory=list, to_dict=False)

    def __post_init__(self) -> None:
        """ Actually creates the callbacks and trainers etc from the options in `options`.
        They are then stored in `self`. 
        """
        actual_callbacks: Dict[str, Callback] = {}

        # NOTE: Need to do a bit of sneaky type tricks to convince the outside world that these
        # fields have the right type.

        # Create the callbacks
        assert isinstance(self.options.callbacks, dict)
        for name, callback in self.options.callbacks.items():
            if isinstance(callback, dict):
                callback = hydra.utils.instantiate(callback)
            elif not isinstance(callback, Callback):
                raise ValueError(f"Invalid callback value {callback}")
            actual_callbacks[name] = callback
        self.callbacks = list(actual_callbacks.values())

        # Create the loggers, if any.
        assert isinstance(self.options.logger, dict)
        actual_loggers: Dict[str, LightningLoggerBase] = {}
        for name, lightning_logger in self.options.logger.items():
            if isinstance(lightning_logger, dict):
                lightning_logger = hydra.utils.instantiate(lightning_logger)
            elif not isinstance(lightning_logger, LightningLoggerBase):
                raise ValueError(f"Invalid logger value {lightning_logger}")
        self.logger = list(actual_loggers.values())

        # Create the Trainer.
        assert isinstance(self.options.trainer, dict)
        if self.options.debug:
            logger.info(f"Setting the max_epochs to 1, since the 'debug' flag was passed.")
            self.options.trainer["max_epochs"] = 1
        trainer = hydra.utils.instantiate(
            self.options.trainer, callbacks=self.callbacks, logger=self.logger,
        )
        assert isinstance(trainer, Trainer)
        self.trainer = trainer

        # Create the datamodule:
        dataset: DatasetConfig = self.options.dataset
        self.datamodule = dataset.make_datamodule(batch_size=self.options.model.batch_size)

        # Create the network
        network_hparams: Network.HParams = self.options.network
        network_type: Type[Network] = get_outer_class(type(network_hparams))
        assert isinstance(
            network_hparams, network_type.HParams
        ), "HParams type should match net type"
        self.network = network_type(
            in_channels=self.datamodule.dims[0],
            n_classes=self.datamodule.num_classes,  # type: ignore
            hparams=network_hparams,
        )

        # Create the model
        model_hparams: Model.HParams = self.options.model
        model_type: Type[Model] = get_outer_class(type(model_hparams))
        assert isinstance(model_hparams, model_type.HParams), "HParams type should match model type"
        self.model = model_type(
            network=self.network,
            datamodule=self.datamodule,
            hparams=model_hparams,
            network_hparams=network_hparams,
            config=Config(seed=self.options.seed, debug=self.options.debug),
        )
        assert isinstance(self.model, LightningModule)

    def run(self) -> float:
        if self.options.seed is not None:
            seed_everything(seed=self.options.seed, workers=True)

        root_logger = logging.getLogger()
        if self.options.debug:
            root_logger.setLevel(logging.INFO)
        elif self.options.verbose:
            root_logger.setLevel(logging.DEBUG)

        root_logger = logging.getLogger()
        # --- Run the experiment. ---
        self.trainer.fit(self.model, datamodule=self.datamodule)

        val_results = self.trainer.validate(model=self.model, datamodule=self.datamodule)
        assert len(val_results) == 1
        top1_accuracy: float = val_results[0]["val/accuracy"]
        top5_accuracy: float = val_results[0]["val/top5_accuracy"]
        print(f"Validation top1 accuracy: {top1_accuracy:.1%}")
        print(f"Validation top5 accuracy: {top5_accuracy:.1%}")

        if not self.options.debug:
            from orion.client import report_objective

            test_error = 1 - top1_accuracy
            report_objective(test_error)

        if wandb.run:
            wandb.finish()

        return top1_accuracy
        # TODO: Enable this later.
        # Run on the test set:
        # test_results = trainer.test(model, datamodule=datamodule, verbose=True)
        # top1_accuracy: float = test_results[0]["test/accuracy"]
        # top5_accuracy: float = test_results[0]["test/top5_accuracy"]
        # print(f"Test top1 accuracy: {top1_accuracy:.1%}")
        # print(f"Test top5 accuracy: {top5_accuracy:.1%}")


def run(options: Options):
    if options.seed is not None:
        seed_everything(seed=options.seed, workers=True)

    root_logger = logging.getLogger()
    if options.debug:
        root_logger.setLevel(logging.INFO)
    elif options.verbose:
        root_logger.setLevel(logging.DEBUG)

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

    return top1_accuracy
    # TODO: Enable this later.
    # Run on the test set:
    # test_results = trainer.test(model, datamodule=datamodule, verbose=True)
    # top1_accuracy: float = test_results[0]["test/accuracy"]
    # top5_accuracy: float = test_results[0]["test/top5_accuracy"]
    # print(f"Test top1 accuracy: {top1_accuracy:.1%}")
    # print(f"Test top5 accuracy: {top5_accuracy:.1%}")


if __name__ == "__main__":
    main()
