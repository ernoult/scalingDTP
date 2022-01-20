""" Script that runs Pytorch lightning version of the DTP models.

Use `python main_pl.py --help` to see a list of all available arguments.
"""
import argparse
import dataclasses
import json
import logging
import textwrap
import warnings
from argparse import Namespace
from dataclasses import asdict
from typing import Callable, List, Literal, Type, TypeVar, Union

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from simple_parsing import ArgumentParser
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from torch import nn

import wandb
from target_prop.config import Config
from target_prop.models import DTP, BaselineModel, ParallelDTP, TargetProp, VanillaDTP
from target_prop.utils import make_reproducible
from target_prop.networks import (
    ResNet18,
    ResNet34,
    SimpleVGG,
    LeNet,
)

HParams = TypeVar("HParams", bound=HyperParameters)

Model = Union[BaselineModel, DTP, ParallelDTP, VanillaDTP, TargetProp]


def main(parser: ArgumentParser = None):
    """Main script."""
    # Allow passing a parser in, in case this is used as a subparser action for another program.
    parser = parser or ArgumentParser(description=__doc__)
    assert parser is not None
    action_subparsers = parser.add_subparsers(
        title="action", description="Which action to take.", required=True
    )

    run_parser = action_subparsers.add_parser("run", help="Single run.", description=run.__doc__)
    run_parser.set_defaults(_action=run)
    add_run_args(run_parser)

    sweep_parser = action_subparsers.add_parser(
        "sweep", help="Hyper-Parameter sweep.", description=sweep.__doc__
    )
    sweep_parser.set_defaults(_action=sweep)
    add_sweep_args(sweep_parser)

    args = parser.parse_args()
    # convert the parsed args into the arguments that will be passed to the chosen action function.
    action_kwargs = vars(args)
    action = action_kwargs.pop("_action")

    action(**action_kwargs)


def add_run_args(parser: ArgumentParser):
    """Adds the command-line arguments for launching a run."""
    # NOTE: if args for config are added here, then command becomes
    # python main.py (config args) [dtp|parallel_dtp] (model ags)
    # parser.add_arguments(Config, dest="config")

    subparsers = parser.add_subparsers(
        title="model", description="Type of model to use.", required=True
    )

    for option_str, help_str, model_type in [
        ("dtp", "Use DTP", DTP),
        ("parallel_dtp", "Use the parallel variant of DTP", ParallelDTP),
        ("vanilla_dtp", "Use 'vanilla' DTP", VanillaDTP),
        ("tp", "Use 'vanilla' Target Propagation", TargetProp),
        ("backprop", "Use regular backprop", BaselineModel),
    ]:
        subparser = subparsers.add_parser(option_str, help=help_str, description=model_type.__doc__)
        # subparser.add_arguments(Config, dest="config")
        # subparser.add_arguments(model_type.HParams, dest="hparams")  # type: ignore
        subparser.set_defaults(model_type=model_type)

        net_subparsers = subparser.add_subparsers(
            title="network", description="Which network architecture to use", required=True
        )

        for option_str, net_help_str, net_fn, net_hparams in [
            ("simple_vgg", "VGG-like architecture", SimpleVGG, SimpleVGG.HParams),
            ("resnet18", "ResNet18 architecture", ResNet18, ResNet18.HParams),
            ("resnet34", "ResNet34 architecture", ResNet34, ResNet34.HParams),
            ("lenet", "LeNet-like architecture", LeNet, LeNet.HParams),
        ]:
            net_subparser = net_subparsers.add_parser(
                option_str, help=help_str + " with a " + net_help_str, description=net_fn.__doc__,
            )
            net_subparser.add_arguments(model_type.HParams, dest="hparams")
            net_subparser.add_arguments(net_hparams, dest="network_hparams")
            net_subparser.add_arguments(Config, dest="config")
            net_subparser.set_defaults(network_type=net_fn)

    # Fixes a weird little argparse bug with metavar.
    subparsers.metavar = "{" + ",".join(subparsers._name_parser_map.keys()) + "}"


def run(
    config: Config,
    model_type: Type[Model],
    hparams: HyperParameters,
    network_type: Callable[..., nn.Sequential],
    network_hparams: HyperParameters,
) -> float:
    """Executes a run, where a model of the given type is trained, with the given hyper-parameters."""
    print(f"Type of model used: {model_type}")
    print("Config:")
    print(config.dumps_json(indent="\t"))
    print("HParams:")
    print(hparams.dumps_json(indent="\t"))
    print("Network HParams:")
    print(network_hparams.dumps_json(indent="\t"))
    print(f"Selected seed: {config.seed}")
    seed_everything(seed=config.seed, workers=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    assert isinstance(hparams, model_type.HParams), "HParams type should match model type"

    if config.debug:
        print(f"Setting the max_epochs to 1, since '--debug' was passed.")
        hparams.max_epochs = 1
        root_logger.setLevel(logging.DEBUG)

    # Create the datamodule:
    datamodule = config.make_datamodule(batch_size=hparams.batch_size)

    # Create the network
    network: nn.Sequential = network_type(
        in_channels=datamodule.dims[0], n_classes=datamodule.num_classes, hparams=network_hparams,
    )

    # Create the model
    model: Model = model_type(
        datamodule=datamodule,
        network=network,
        hparams=hparams,
        config=config,
        network_hparams=network_hparams,
    )

    # --- Create the trainer.. ---
    # NOTE: Now each algo can customize how the Trainer gets created.
    trainer = model.create_trainer()

    # --- Run the experiment. ---
    trainer.fit(model, datamodule=datamodule)

    # Run on the test set:
    test_results = trainer.test(model, datamodule=datamodule, verbose=True)

    wandb.finish()
    test_accuracy: float = test_results[0]["test/accuracy"]
    print(f"Test accuracy: {test_accuracy:.1%}")
    return test_accuracy


def add_sweep_args(parser: ArgumentParser):
    subparsers = parser.add_subparsers(
        title="model", description="Type of model to use for the sweep.", required=True, help=None,
    )

    for option_str, help_str, model_type in [
        ("dtp", "Use DTP", DTP),
        ("parallel_dtp", "Use the parallel variant of DTP", ParallelDTP),
        ("vanilla_dtp", "Use 'vanilla' DTP", VanillaDTP),
        ("tp", "Use 'vanilla' Target Propagation", TargetProp),
        ("backprop", "Use regular backprop", BaselineModel),
    ]:
        subparser = subparsers.add_parser(option_str, help=help_str, description=model_type.__doc__)
        # NOTE: Add the config to the subparsers.
        # subparser.add_arguments(Config, dest="config")  # note: moved to the last subparser below.
        # NOTE: Don't add the command-line options for the arguments here, since they will get
        # overwritten with sampled values.
        # subparser.add_arguments(model_type.HParams, dest="hparams")
        subparser.set_defaults(model_type=model_type)

        net_subparsers = subparser.add_subparsers(
            title="network", description="Which network architecture to use", required=True
        )

        for option_str, net_help_str, net_fn, net_hparams in [
            ("simple_vgg", "VGG-like architecture", SimpleVGG, SimpleVGG.HParams),
            ("resnet18", "ResNet18 architecture", ResNet18, ResNet18.HParams),
            ("resnet34", "ResNet34 architecture", ResNet34, ResNet34.HParams),
            ("lenet", "LeNet-like architecture", LeNet, LeNet.HParams),
        ]:
            net_subparser = net_subparsers.add_parser(
                option_str, help=help_str + " with a " + net_help_str, description=net_fn.__doc__,
            )
            net_subparser.add_arguments(Config, dest="config")
            net_subparser.set_defaults(network_hparams_type=net_hparams)
            net_subparser.set_defaults(network_type=net_fn)
            # NOTE: This will be a 'fixed' hyper-parameter. We don't want to sample it.
            net_subparser.add_argument(
                "--max_epochs",
                type=int,
                default=10,
                help="How many epochs to run for each configuration.",
            )

    parser.add_argument("--n-runs", "--n_runs", type=int, default=1, help="How many runs to do.")
    # Fixes a weird little argparse bug with metavar.
    subparsers.metavar = "{" + ",".join(subparsers._name_parser_map.keys()) + "}"


def sweep(
    config: Config,
    model_type: Type[Model],
    network_type: Callable[..., nn.Sequential],
    network_hparams_type: Type[HyperParameters],
    n_runs: int = 1,
    **fixed_hparams,`
):
    """Performs a hyper-parameter sweep.

    The hyper-parameters are sampled randomly from their priors. This then calls `run` with the
    sampled hyper-parameters.
    """
    print("Config:")
    print(config.dumps_json(indent="\t"))

    if config.seed is None:
        config.seed = 123
        warnings.warn(RuntimeWarning(f"Using default random seed of {config.seed}."))

    print(f"Type of model used: {model_type}")
    hparam_type = model_type.HParams
    hparam_space_dict = hparam_type().get_orion_space()
    network_hparam_space_dict = network_hparams_type().get_orion_space()
    print("Algorithm HPO search space:")
    # print(space_dict)
    print(json.dumps(hparam_space_dict, indent="\t"))
    print("Network HPO search space:")
    print(json.dumps(network_hparam_space_dict, indent="\t"))

    if fixed_hparams:
        print(f"Fixed hyper-parameter values: {fixed_hparams}")

    # Sample the hparams of each run in advance, just to make sure that they are properly seeded and
    # not always the same.
    # NOTE: This is fine for now, since we use random search.
    run_hparams = []
    run_network_hparams = []
    assert config.seed is not None
    with make_reproducible(seed=config.seed):
        for run_index in range(n_runs):
            hparams = hparam_type.sample()
            network_hparams = network_hparams_type.sample()
            if fixed_hparams:
                # Check if they are for the network or for the Model (algo) hparams:
                # Fix some of the hyper-paremters
                hparams_dict = hparams.to_dict()
                from dataclasses import replace

                hparams = replace(
                    hparams, **{k: v for k, v in fixed_hparams.items() if hasattr(hparams, k)}
                )
                network_hparams = replace(
                    network_hparams,
                    **{k: v for k, v in fixed_hparams.items() if hasattr(network_hparams, k)},
                )
            run_hparams.append(hparams)
            run_network_hparams.append(network_hparams)

    performances = []
    for run_index, (run_hparams, network_hparams) in enumerate(
        zip(run_hparams, run_network_hparams)
    ):
        print(f"\n\n----------------- Starting run #{run_index+1}/{n_runs} -------------------\n\n")
        performance = run(
            config=config,
            model_type=model_type,
            hparams=run_hparams,
            network_type=network_type,
            network_hparams=network_hparams,
        )
        performances.append(performance)
    return performances


if __name__ == "__main__":
    main()
