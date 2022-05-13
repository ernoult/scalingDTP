import ray
import math
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
import torch
import pytorch_lightning as pl
from filelock import FileLock
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import os
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
import os
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Dict, List, Optional, Type
import logging
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
from target_prop.models import Model, DTP, VanillaDTP, TargetProp, ParallelDTP, BaselineModel
from target_prop.models.model import Model
from target_prop.networks import  Network, ResNet18, ResNet34, SimpleVGG, LeNet, ViT
from target_prop.networks.network import Network
from target_prop.scheduler_config import CosineAnnealingLRConfig, StepLRConfig
from target_prop.utils.hydra_utils import get_outer_class

import os
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Dict, List, Optional, Type
import logging
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
from target_prop.models import Model, DTP, VanillaDTP, TargetProp, ParallelDTP, BaselineModel
from target_prop.models.model import Model
from target_prop.networks import  Network, ResNet18, ResNet34, SimpleVGG, LeNet, ViT
from target_prop.networks.network import Network
from target_prop.scheduler_config import CosineAnnealingLRConfig, StepLRConfig
from target_prop.utils.hydra_utils import get_outer_class
def train_mnist(tuneconfig):


    logger = get_logger(__name__)
    raw_options = OmegaConf.create({'dataset': {'dataset': 'cifar10', 'data_dir': '/home/ono/Dev/scalingDTP/data', 'num_workers': 12, 'shuffle': True, 'normalize': True, 'image_crop_pad': 4, 'val_split': 0.1, 'use_legacy_std': False}, 'model': {'lr_scheduler': {'interval': 'epoch', 'frequency': 1, 'T_max': 85, 'eta_min': 1e-05}, 'batch_size': 128, 'use_scheduler': True, 'feedback_training_iterations': [20, 25, 30, 35, 15], 'max_epochs': 90, 'b_optim': {'type': 'sgd', 'lr': [0.0001, 0.00035, 0.001, 0.002, 0.08], 'weight_decay': None, 'momentum': 0.9}, 'noise': [0.4, 0.4, 0.2, 0.2, 0.08], 'f_optim': {'type': 'sgd', 'lr': [0.04], 'weight_decay': 0.0001, 'momentum': 0.9}, 'beta': 0.7, 'feedback_samples_per_iteration': 1, 'early_stopping_patience': 0, 'init_symetric_weights': False, 'plot_every': 1000}, 'network': {'activation': 'elu', 'batch_size': 128, 'channels': [128, 128, 256, 256, 512], 'bias': True}, 'trainer': {'_target_': 'pytorch_lightning.Trainer', 'gpus': -1, 'strategy': 'dp', 'min_epochs': 1, 'max_epochs': 90, 'resume_from_checkpoint': None}, 'callbacks': {'model_checkpoint': {'_target_': 'pytorch_lightning.callbacks.ModelCheckpoint', 'monitor': 'val/accuracy', 'mode': 'max', 'save_top_k': 1, 'save_last': True, 'verbose': False, 'dirpath': 'checkpoints/', 'filename': 'epoch_{epoch:03d}', 'auto_insert_metric_name': False}, 'early_stopping': {'_target_': 'pytorch_lightning.callbacks.EarlyStopping', 'monitor': 'val/accuracy', 'mode': 'max', 'patience': 100, 'min_delta': 0}, 'model_summary': {'_target_': 'pytorch_lightning.callbacks.RichModelSummary', 'max_depth': 1}, 'rich_progress_bar': {'_target_': 'pytorch_lightning.callbacks.RichProgressBar'}}, 'logger': {'wandb': {'_target_': 'pytorch_lightning.loggers.wandb.WandbLogger', 'project': 'scalingDTP', 'name': '${name}', 'save_dir': tune.get_trial_dir(), 'offline': False, 'id': None, 'log_model': False, 'prefix': '', 'job_type': 'train', 'group': '', 'tags': []}}, 'debug': False, 'verbose': False, 'seed': 4248715256, 'name': ''}
                      )
    trainer =Trainer = field(init=False, to_dict=False)
    model: Model = field(init=False, to_dict=False)
    network: Network = field(init=False, to_dict=False)
    datamodule: VisionDataModule = field(init=False, to_dict=False)

    callbacks: List[Callback] = field(init=False, default_factory=list, to_dict=False)
    loggers: List[LightningLoggerBase] = field(init=False, default_factory=list, to_dict=False)
    options = OmegaConf.to_object(raw_options)
    actual_callbacks: Dict[str, Callback] = {}
    # Create the callbacks
    assert isinstance(options['callbacks'], dict)
    for name, callback in options['callbacks'].items():
        if isinstance(callback, dict):
            callback = hydra.utils.instantiate(callback)
        elif not isinstance(callback, Callback):
            raise ValueError(f"Invalid callback value {callback}")
        actual_callbacks[name] = callback
        callbacks = list(actual_callbacks.values())
    # Create the loggers, if any.
    assert isinstance(options['logger'], dict)
    actual_loggers: Dict[str, LightningLoggerBase] = {}
    for name, lightning_logger in options['logger'].items():
        if isinstance(lightning_logger, dict):
            lightning_logger = hydra.utils.instantiate(lightning_logger)
        elif not isinstance(lightning_logger, LightningLoggerBase):
            raise ValueError(f"Invalid logger value {lightning_logger}")
        actual_loggers[name] = lightning_logger
    logger = list(actual_loggers.values())
    assert isinstance(options['trainer'], dict)
    if options["debug"]:
        logger.info(f"Setting the max_epochs to 1, since the 'debug' flag was passed.")
        self.options['trainer']["max_epochs"] = 1
    trainer = hydra.utils.instantiate(
        options['trainer'], callbacks=callbacks, logger=logger,
    )
    from target_prop.datasets.dataset_config import DatasetConfig
    options = OmegaConf.create(raw_options)
    dataset =DatasetConfig(options.dataset)
    datamodule = dataset.make_datamodule(batch_size=options.model.batch_size)
    # datamodule = VisionDataModule(dataset.make_datamodule(batch_size=options.model.batch_size))
    # datamodule = VisionDataModule #= dataset.make_datamodule(batch_size=options.model.batch_size)

    network = ViT(in_channels = datamodule.dims[0],n_classes=datamodule.num_classes,hparams=options.network)
    # datamodule= VisionDataModule
    options = OmegaConf.to_object(raw_options)
    dict(OmegaConf.to_object(OmegaConf.create(options['model'])))
    hparams = DTP.HParams()

    hparams.feedback_training_iterations = tuneconfig['feedback_training_iterations']
    print("HParams:", hparams.dumps_json(indent="\t"))


    model = DTP(network=network,datamodule=datamodule,hparams= DTP.HParams(),network_hparams=options['network'],config=Config(seed = options['seed'],debug=options['debug']))


    trainer.fit(model,datamodule=datamodule)
def train_mnist_no_tune():

    config = {
        "feedback_training_iterations": [5,10,20,20,40,3],
        "layer_2_size": 256,
        "lr": 1e-3,
        "batch_size": 64
    }
    train_mnist(config)


# train_mnist_no_tune()



def train_mnist_tune(tuneconfig, num_epochs=90,num_gpus=1,
                                                    data_dir='~/scalingDTP/data'):



    logger = get_logger(__name__)
    raw_options = OmegaConf.create({'enable_progress_bar':False,'dataset': {'dataset': 'cifar10', 'data_dir': os.path.expanduser('~/scalingDTP/data/'), 'num_workers': 12, 'shuffle': True, 'normalize': True, 'image_crop_pad': 4, 'val_split': 0.1, 'use_legacy_std': False}, 'model': {'lr_scheduler': {'interval': 'epoch', 'frequency': 1, 'T_max': 85, 'eta_min': 1e-05}, 'batch_size': 128, 'use_scheduler': True, 'feedback_training_iterations': [20, 25, 30, 35, 15], 'max_epochs': 90, 'b_optim': {'type': 'sgd', 'lr': [0.0001, 0.00035, 0.001, 0.002, 0.08], 'weight_decay': None, 'momentum': 0.9}, 'noise': [0.4, 0.4, 0.2, 0.2, 0.08], 'f_optim': {'type': 'sgd', 'lr': [0.04], 'weight_decay': 0.0001, 'momentum': 0.9}, 'beta': 0.7, 'feedback_samples_per_iteration': 1, 'early_stopping_patience': 0, 'init_symetric_weights': False, 'plot_every': 1000}, 'network': {'activation': 'elu', 'batch_size': 128, 'channels': [128, 128, 256, 256, 512], 'bias': True}, 'trainer': {'_target_': 'pytorch_lightning.Trainer', 'gpus': -1, 'strategy': 'dp', 'min_epochs': 1, 'max_epochs': 90, 'resume_from_checkpoint': None}, 'callbacks': {'model_checkpoint': {'_target_': 'pytorch_lightning.callbacks.ModelCheckpoint', 'monitor': 'val/accuracy', 'mode': 'max', 'save_top_k': 1, 'save_last': True, 'verbose': False, 'dirpath': 'checkpoints/', 'filename': 'epoch_{epoch:03d}', 'auto_insert_metric_name': False}, 'early_stopping': {'_target_': 'pytorch_lightning.callbacks.EarlyStopping', 'monitor': 'val/accuracy', 'mode': 'max', 'patience': 100, 'min_delta': 0}, 'model_summary': {'_target_': 'pytorch_lightning.callbacks.RichModelSummary', 'max_depth': 1}, 'rich_progress_bar': {'_target_': 'pytorch_lightning.callbacks.RichProgressBar'}}, 'logger': {'wandb': {'_target_': 'pytorch_lightning.loggers.wandb.WandbLogger', 'project': 'scalingDTP', 'name': '${name}', 'save_dir': '.', 'offline': False, 'id': None, 'log_model': False, 'prefix': '', 'job_type': 'train', 'group': '', 'tags': []}}, 'debug': False, 'verbose': False, 'seed': 4248715256, 'name': ''}
                      )
    trainer =Trainer = field(init=False, to_dict=False)
    model: Model = field(init=False, to_dict=False)
    network: Network = field(init=False, to_dict=False)
    datamodule: VisionDataModule = field(init=False, to_dict=False)

    callbacks: List[Callback] = field(init=False, default_factory=list, to_dict=False)
    loggers: List[LightningLoggerBase] = field(init=False, default_factory=list, to_dict=False)
    options = OmegaConf.to_object(raw_options)
    actual_callbacks: Dict[str, Callback] = {}
    # Create the callbacks
    assert isinstance(options['callbacks'], dict)
    for name, callback in options['callbacks'].items():
        if isinstance(callback, dict):
            callback = hydra.utils.instantiate(callback)
        elif not isinstance(callback, Callback):
            raise ValueError(f"Invalid callback value {callback}")
        actual_callbacks[name] = callback
    actual_callbacks['tune_callback']= TuneReportCallback(
                {
                    "loss": "val/Loss",
                    "mean_accuracy": "val/accuracy",
                    "top_5": "val/top5_accuracy"
                },
                on="validation_end")
    callbacks = list(actual_callbacks.values())
    # Create the loggers, if any.
    assert isinstance(options['logger'], dict)
    actual_loggers: Dict[str, LightningLoggerBase] = {}
    for name, lightning_logger in options['logger'].items():
        if isinstance(lightning_logger, dict):
            lightning_logger = hydra.utils.instantiate(lightning_logger)
        elif not isinstance(lightning_logger, LightningLoggerBase):
            raise ValueError(f"Invalid logger value {lightning_logger}")
        actual_loggers[name] = lightning_logger
    logger = list(actual_loggers.values())
    assert isinstance(options['trainer'], dict)
    if options["debug"]:
        logger.info(f"Setting the max_epochs to 1, since the 'debug' flag was passed.")
        self.options['trainer']["max_epochs"] = 1
    trainer = hydra.utils.instantiate(
        options['trainer'], callbacks=callbacks, logger=logger,
    )
    from target_prop.datasets.dataset_config import DatasetConfig
    options = OmegaConf.create(raw_options)
    dataset =DatasetConfig(options.dataset)
    datamodule = dataset.make_datamodule(batch_size=options.model.batch_size)
    # datamodule = VisionDataModule(dataset.make_datamodule(batch_size=options.model.batch_size))
    # datamodule = VisionDataModule #= dataset.make_datamodule(batch_size=options.model.batch_size)

    network = ViT(in_channels = datamodule.dims[0],n_classes=datamodule.num_classes,hparams=options.network)
    # datamodule= VisionDataModule
    options = OmegaConf.to_object(raw_options)
    dict(OmegaConf.to_object(OmegaConf.create(options['model'])))


    hparams = DTP.HParams()

    feedback_training_iterations=[tuneconfig["l1i"],tuneconfig["l2i"],tuneconfig["l3i"],tuneconfig["l4i"],tuneconfig["l5i"]]
    hparams.feedback_training_iterations = feedback_training_iterations
    # hparams.backward_lr =[tuneconfig["blr1"],tuneconfig["blr2"],tuneconfig["blr3"],tuneconfig["blr4"],tuneconfig["blr5"]]
    # hparams.forward_lr = tuneconfig["forward_lr"]
    print("HParams:", hparams.dumps_json(indent="\t"))


    model = DTP(network=network,datamodule=datamodule,hparams= DTP.HParams(),network_hparams=options['network'],config=Config(seed = options['seed'],debug=options['debug']))


    trainer.fit(model,datamodule=datamodule)


def tune_mnist_asha(num_samples=50, num_epochs=90, gpus_per_trial=1, data_dir="~/scalingDTP/data"):



    config = {


        "l1i" :tune.qrandint(3,20,2),
        "l2i" : tune.qrandint(5,30,5),
        "l3i":  tune.qrandint(10,50,5),
        "l4i" :tune.qrandint(20,60,5),
        "l5i": tune.qrandint(3,40,2),


        # "blr1" :tune.qloguniform(1e-5, 1e-3, 1e-5),
        # "blr2" : tune.qloguniform(1e-4,2e-3,1e-5),
        # "blr3":  tune.qloguniform(5e-4,2e-3,1e-5),
        # "blr4" :tune.qloguniform(1e-3,5e-3,1e-5),
        # "blr5": tune.qloguniform(5e-3,1e-1,1e-5),

        # "forward_lr": tune.qloguniform(0.01, 0.09, 5e-4),
        # "backward_lr": tune.choice([64, 128, 256]),
        # "lr": tune.loguniform(1e-4, 1e-1),
        # "batch_size": tune.choice([32, 64, 128]),
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train_mnist_tune,
                                                    num_epochs=num_epochs,
                                                    num_gpus=gpus_per_trial,
                                                    data_dir=data_dir)
    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mnist_asha")

    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == "__main__":
    # 🌍 Prepare the environment -----------------------------------------------
    # load_dotenv(verbose=True)
    print(f'Cuda available: {torch.cuda.is_available()}')
    # os.environ['TUNE_SYNCER_VERBOSITY'] = '3'  # shows full ssh commands when debugging on the cluster
    wandb.login(key='3819aa873a634d5ce3929c0f0ef2d98e2a83d322')

    # Initialize ray -----------------------------------------------------------
    ray.init(
        address='auto',

        # Enable these if debugging locally
        # local_mode=True,
        #resources={'cpu':4,"gpu":1},
    )
    tune_mnist_asha()
