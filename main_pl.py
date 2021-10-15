""" Script that runs Pytorch lightning version of the prototype DTP model.

Use `python main.py --help` for a list of all available arguments,
    
    or (even better), take a look at the [`Config`](target_prop/config.py) and the
    [`Model.HParams`](target_prop/model.py) classes to see their definition.
"""
import logging
from typing import Type

from pytorch_lightning.utilities.seed import seed_everything
from target_prop.models import BaseModel, SequentialModel, ParallelModel, BaselineModel
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
    parser.set_defaults(model=SequentialModel)
    parser.add_arguments(Config, dest="config")

    subparsers = parser.add_subparsers(
        title="model", description="Type of model to use.", metavar="<model_type>", required=True
    )
    sequential_parser: ArgumentParser = subparsers.add_parser("sequential")
    sequential_parser.add_arguments(SequentialModel.HParams, "hparams")
    sequential_parser.set_defaults(model_type=SequentialModel)

    parallel_parser: ArgumentParser = subparsers.add_parser("parallel")
    parallel_parser.add_arguments(ParallelModel.HParams, "hparams")
    parallel_parser.set_defaults(model_type=ParallelModel)

    baseline_parser: ArgumentParser = subparsers.add_parser("baseline")
    baseline_parser.add_arguments(BaselineModel.HParams, "hparams")
    baseline_parser.set_defaults(model_type=BaselineModel)

    # Parse the arguments
    args = parser.parse_args()

    config: Config = args.config
    hparams: BaseModel.HParams = args.hparams

    if running_sweep:
        # Sample some hyper-parameters from the prior, and overwrite the sampled values with those
        # passed through the command-line.
        sampled_hparams = BaseModel.HParams.sample()
        default_hparams_dict = BaseModel.HParams().to_dict()
        custom_hparams = {
            k: v for k, v in hparams.to_dict().items() if v != default_hparams_dict[k]
        }
        if custom_hparams:
            print(f"Overwriting sampled values for entries {custom_hparams}")
            hparams_dict = sampled_hparams.to_dict()
            hparams_dict.update(custom_hparams)
            hparams = BaseModel.HParams.from_dict(hparams_dict)

    model_class: Type[BaseModel] = args.model_type
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
    model = model_class(datamodule=datamodule, hparams=hparams, config=config)

    # --- Create the trainer.. ---
    # IDEA: Would perhaps be useful to add command-line arguments for DP/DDP/etc. 
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        gpus=torch.cuda.device_count(),
        track_grad_norm=False,
        accelerator=None,
        # NOTE: Not sure why but seems like they are still reloading them after each epoch!
        reload_dataloaders_every_epoch=False,
        # accelerator="ddp",
        # profiler="simple",
        # callbacks=[],
        terminate_on_nan=True,
        logger=WandbLogger() if not config.debug else None,
    )
    
    # --- Run the experiment. ---
    trainer.fit(model, datamodule=datamodule)

    # Run on the test set:
    test_results = trainer.test(model, datamodule=datamodule)

    wandb.finish()
    print(test_results)
    test_accuracy: float = test_results[0]["test/accuracy"]
    return test_accuracy

if __name__ == "__main__":
    main()
