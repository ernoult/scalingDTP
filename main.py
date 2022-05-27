from __future__ import annotations

import logging
import os
import random
import warnings
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Dict, Optional, Type

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase
from simple_parsing.helpers import field
from simple_parsing.helpers.serialization.serializable import Serializable
from hydra.utils import instantiate

from target_prop.config import Config
from target_prop.datasets.dataset_config import (
    DatasetConfig,
    remove_normalization_from_transforms,
    validate_datamodule,
)
from target_prop.models import (
    DTP,
    BaselineModel,
    Model,
    ParallelDTP,
    TargetProp,
    VanillaDTP,
)
from target_prop.models.model import Model
from target_prop.networks import LeNet, Network, ResNet18, ResNet34, SimpleVGG
from target_prop.networks.network import Network
from target_prop.scheduler_config import CosineAnnealingLRConfig, StepLRConfig
from target_prop.utils.hydra_utils import get_outer_class

logger = get_logger(__name__)


@dataclass
class Options(Serializable):
    """All the options required for a run. This dataclass acts as a schema for the Hydra configs.

    For more info, see https://hydra.cc/docs/tutorials/structured_config/schema/
    """

    # Configuration for the dataset + transforms.
    # TODO: might replace this with a simpler config with just a _target_ set to the pl_bolts
    # datamodules. The only thing is, this class also contains config for the transforms.
    dataset: DatasetConfig

    # The hyper-parameters of the model to use.
    model: Model.HParams

    # The hyper-parameters of the network to use.
    network: Network.HParams

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


# TODO: Learn more about how variable interpolation works in Hydra, so that we can do this sort of
# stuff:
# from hydra_zen import builds
# cs.store(
#     group="network",
#     name="simple_vgg",
#     node=builds(
#         SimpleVGG,
#         hparams=builds(SimpleVGG.HParams),
#         in_channels="${/dataset.dims[0]}",
#         n_classes="${/dataset.num_classes}",
#     ),
# )

cs.store(group="network", name="simple_vgg", node=SimpleVGG.HParams())
cs.store(group="network", name="lenet", node=LeNet.HParams())
cs.store(group="network", name="resnet18", node=ResNet18.HParams())
cs.store(group="network", name="resnet34", node=ResNet34.HParams())

cs.store(group="lr_scheduler", name="step", node=StepLRConfig)
cs.store(group="lr_scheduler", name="cosine", node=CosineAnnealingLRConfig)


@hydra.main(
    config_path="conf", config_name="config", version_base=None,
)
def main(config: DictConfig | Options) -> float:
    print(os.getcwd())
    experiment = Experiment.from_options(config)
    return experiment.run()


@dataclass
class Experiment:
    """Experiment class.

    Created from the Options that are parsed from Hydra. Can be used to run the experiment.
    """

    model: Model
    network: Network
    datamodule: VisionDataModule
    trainer: Trainer

    @classmethod
    def from_options(cls, options: DictConfig | Options) -> Experiment:
        """Creates all the components of an experiment from the DictConfig coming from Hydra."""
        if isinstance(options, DictConfig):
            converted_options = OmegaConf.to_object(options)
            assert isinstance(converted_options, Options)
            options = converted_options
        exp = instantiate_experiment_components(options)
        return exp

    def run(self) -> float:
        return run_experiment(self)


def run_experiment(exp: Experiment | DictConfig | Options) -> float:
    """Run the experiment, and return the classification error.

    By default, if validation is performed, returns the validation error. Returns the training error
    when `trainer.overfit_batches != 0` (e.g. when debugging or testing). Otherwise, if
    `trainer.limit_val_batches == 0`, returns the test error.
    """
    if not isinstance(exp, Experiment):
        exp = Experiment.from_options(exp)

    # TODO Probably log the hydra config with something like this:
    # exp.trainer.logger.log_hyperparams()
    exp.trainer.fit(exp.model, datamodule=exp.datamodule)

    if (exp.trainer.limit_val_batches == exp.trainer.limit_test_batches == 0) or (
        exp.trainer.overfit_batches == 1
    ):
        # We want to report the training error.
        metrics = {
            **exp.trainer.logged_metrics,
            **exp.trainer.callback_metrics,
            **exp.trainer.progress_bar_metrics,
        }
        if "train/accuracy" not in metrics:
            raise RuntimeError(
                f"Unable to find the train/accuracy key in the training metrics:\n"
                f"{metrics.keys()}"
            )
        train_acc = metrics["train/accuracy"]
        train_error = 1 - train_acc

        # Probably not want to upload this to wandb, I'd assume.
        return train_error

    if exp.trainer.limit_val_batches != 0:
        results = exp.trainer.validate(model=exp.model, datamodule=exp.datamodule)
        results_type = "val"
    else:
        warnings.warn(
            RuntimeWarning(
                "About to use the test set for evaluation! This should be done in a sweep!"
            )
        )
        results = exp.trainer.test(model=exp.model, datamodule=exp.datamodule)
        results_type = "test"

    top1_accuracy: float = results[0][f"{results_type}/accuracy"]
    top5_accuracy: float = results[0][f"{results_type}/top5_accuracy"]
    print(f"{results_type} top1 accuracy: {top1_accuracy:.1%}")
    print(f"{results_type} top5 accuracy: {top5_accuracy:.1%}")

    error = 1 - top1_accuracy

    if wandb.run:
        wandb.finish()
    return error


def instantiate_experiment_components(options: Options) -> Experiment:
    """Do all the postprocessing necessary (e.g., create the network, Model, datamodule, callbacks,
    Trainer, etc) to go from the options that come from Hydra, into all required components for the
    experiment, which is stored as a namedtuple-like class called `Experiment`.

    NOTE: This also has the effect of seeding the random number generators, so the weights that are
    constructed are always deterministic.
    """

    root_logger = logging.getLogger()
    if options.debug:
        root_logger.setLevel(logging.INFO)
    elif options.verbose:
        root_logger.setLevel(logging.DEBUG)

    if options.seed is not None:
        seed = options.seed
        print(f"seed manually set to {options.seed}")
    else:
        seed = random.randint(0, int(1e5))
        print(f"Randomly selected seed: {seed}")
    seed_everything(seed=seed, workers=True)

    # NOTE: Need to do a bit of sneaky type tricks to convince the outside world that these
    # fields have the right type.

    # instantiate all the callbacks
    callbacks = list(instantiate(options.callbacks).values())

    # Create the loggers, if any.
    pl_logger = list(instantiate(options.logger).values())

    # Create the Trainer.
    assert isinstance(options.trainer, dict)
    if options.debug:
        logger.info(f"Setting the max_epochs to 1, since the 'debug' flag was passed.")
        options.trainer["max_epochs"] = 1
    if "_target_" not in options.trainer:
        options.trainer["_target_"] = Trainer
    trainer = instantiate(options.trainer, callbacks=callbacks, logger=pl_logger,)
    assert isinstance(trainer, Trainer)
    trainer = trainer

    # Create the datamodule:
    dataset: DatasetConfig = options.dataset

    datamodule = instantiate(dataset, batch_size=options.model.batch_size)
    datamodule = validate_datamodule(datamodule)

    # Create the network
    network_hparams: Network.HParams = options.network
    # TODO: Convert the network into configs, with some value interpolation of some sort, or just
    # by passing the in_channels and n_classes to the instantiate function:
    # network = instantiate(
    #     options.network, in_channels=datamodule.dims[0], n_classes=datamodule.num_classes
    # )

    network_type: Type[Network] = get_outer_class(type(network_hparams))
    assert isinstance(network_hparams, network_type.HParams), "HParams type should match net type"
    network = network_type(
        in_channels=datamodule.dims[0],
        n_classes=datamodule.num_classes,  # type: ignore
        hparams=network_hparams,
    )
    assert network.hparams is network_hparams

    # Create the model
    model_hparams: Model.HParams = options.model
    model_type: Type[Model] = get_outer_class(type(model_hparams))
    assert isinstance(model_hparams, model_type.HParams), "HParams type should match model type"
    model = model_type(
        datamodule=datamodule,
        hparams=model_hparams,
        config=Config(seed=options.seed, debug=options.debug),
        network=network,
    )
    assert isinstance(model, LightningModule)

    return Experiment(trainer=trainer, model=model, network=network, datamodule=datamodule,)


if __name__ == "__main__":
    main()
