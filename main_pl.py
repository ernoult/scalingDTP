""" Script that runs Pytorch lightning version of the DTP models.

Use `python main_pl.py --help` to see a list of all available arguments.
"""
import dataclasses
import logging
from typing import Type, Union
from typing_extensions import *
from pytorch_lightning.core.lightning import LightningModule

from pytorch_lightning.utilities.seed import seed_everything
from target_prop.models import DTP, ParallelDTP, BaselineModel
from target_prop.config import Config
from simple_parsing import ArgumentParser
import json
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
import torch
import textwrap


def main(running_sweep: bool = False):
    """ Main script. If `running_sweep` is True, then the hyper-parameters are sampled
    from their corresponding priors, else they take their default value (or the value
    passed from the command-line, if present).
    """

    parser = ArgumentParser(description=__doc__)

    # Hard-set to use the Sequential model, for now.
    parser.set_defaults(model=DTP)

    # NOTE: if args for config are added here, then command becomes
    # python main.py (config args) [dtp|parallel_dtp] (model ags)
    # parser.add_arguments(Config, dest="config")

    subparsers = parser.add_subparsers(
        title="model", description="Type of model to use.", required=True
    )
    sequential_parser: ArgumentParser = subparsers.add_parser(
        "dtp", help="Use DTP", description=DTP.__doc__
    )
    sequential_parser.add_arguments(Config, dest="config")
    sequential_parser.add_arguments(DTP.HParams, "hparams")
    sequential_parser.set_defaults(model_type=DTP)

    parallel_parser: ArgumentParser = subparsers.add_parser(
        "parallel_dtp",
        help="Use the parallel variant of DTP",
        description=ParallelDTP.__doc__,
    )
    parallel_parser.add_arguments(Config, dest="config")
    parallel_parser.add_arguments(ParallelDTP.HParams, "hparams")
    parallel_parser.set_defaults(model_type=ParallelDTP)

    baseline_parser: ArgumentParser = subparsers.add_parser(
        "backprop", help="Use regular backprop", description=BaselineModel.__doc__
    )
    baseline_parser.add_arguments(Config, dest="config")
    baseline_parser.add_arguments(BaselineModel.HParams, "hparams")
    baseline_parser.set_defaults(model_type=BaselineModel)

    subparsers.metavar = "{" + ",".join(subparsers._name_parser_map.keys()) + "}"

    # Compute Canada params (probably temporary)
    parser.add_argument('--array', type=int, default=0,
                        help='Array number returned from SLURM.')

    parser.add_argument('--running_sweep', type=bool, default=False,
                        help='Whether or not to overwrite hparams with sampled priors.')

    # Parse the arguments
    args = parser.parse_args()

    config: Config = args.config
    hparams: HyperParameters = args.hparams

    if running_sweep:
        hparams = sample_hparams(base_hparams=hparams)

    model_class: Union[Type[DTP], Type[BaselineModel]] = args.model_type
    print(f"Type of model used: {model_class}")

    print("HParams:")
    print(hparams.dumps_yaml(indent=1), "\t")
    print("Config:")
    print(config.dumps_yaml(indent=1), "\t")

    print(f"Selected seed: {config.seed}")
    seed_everything(seed=config.seed, workers=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if config.debug:
        print(f"Setting the max_epochs to 1, since '--debug' was passed.")
        hparams.max_epochs = 1
        root_logger.setLevel(logging.DEBUG)

    # Create the datamodule:
    datamodule = config.make_datamodule(batch_size=hparams.batch_size)

    # Create the model
    model: Union[BaselineModel, DTP, ParallelDTP] = model_class(
        datamodule=datamodule, hparams=hparams, config=config
    )

    # --- Create the trainer.. ---
    # NOTE: Now each algo can customize how the Trainer gets created.
    trainer = model.create_trainer()

    # --- Run the experiment. ---
    trainer.fit(model, datamodule=datamodule)

    # Run on the test set:
    test_results = trainer.test(model, datamodule=datamodule, verbose=True)

    wandb.finish()
    print(test_results)
    test_accuracy: float = test_results[0]["test/accuracy"]
    return test_accuracy


from dataclasses import asdict
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from typing import TypeVar
from typing import List

HParams = TypeVar("HParams", bound=HyperParameters)


def sample_hparams(base_hparams: HParams) -> HParams:
    # Sample some hyper-parameters from the prior, and overwrite the sampled values with those
    # passed through the command-line.
    hparams_type = type(base_hparams)
    sampled_hparams = hparams_type.sample()

    default_hparams = hparams_type()
    default_hparams_dict = asdict(default_hparams)

    # FIXME (later): Because we don't check for equality between nested dicts here, this means that,
    # for example, we won't sample a learning rate if a type of optimizer to use is passed from the
    # command-line.
    nondefault_hparams: List[str] = [
        k for k, v in asdict(base_hparams).items() if v != default_hparams_dict[k]
    ]
    fixed_hparams = {k: getattr(sampled_hparams, k) for k in nondefault_hparams}
    if fixed_hparams:
        print(f"These hparams won't be sampled from their priors: {fixed_hparams}")
        return dataclasses.replace(sampled_hparams, **fixed_hparams)
    return sampled_hparams


if __name__ == "__main__":
    main()
