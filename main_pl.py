""" Script that runs Pytorch lightning version of the prototype DTP model.

Use `python main.py --help` for a list of all available arguments,
    
    or (even better), take a look at the [`Config`](target_prop/config.py) and the
    [`Model.HParams`](target_prop/model.py) classes to see their definition.
"""
import logging
from typing import Type

from pytorch_lightning.utilities.seed import seed_everything
from target_prop.models import BaseModel, SequentialModel, ParallelModel
from target_prop.config import Config
from simple_parsing import ArgumentParser
import json
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
import torch


def main(sample_hparams: bool = False):
    """ Main script. If `sample_hparams` is True, then the hyper-parameters are sampled
    from their corresponding priors, else they take their default value (or the value
    passed from the command-line, if present).
    """
    parser = ArgumentParser(description=__doc__)

    # Hard-set to use the Sequential model, for now.
    parser.set_defaults(model=SequentialModel)
    parser.add_arguments(SequentialModel.HParams, "hparams")
    parser.add_arguments(Config, "config")

    ## TODO: Once the parallel model is usable (currently focussing on the Sequential model), then
    ## use this to make it possible to swap between them.
    # subparsers = parser.add_subparsers(
    #     title="model", description="Type of model to use.", metavar="<model_type>", required=True
    # )
    # sequential_parser: ArgumentParser = subparsers.add_parser("sequential")
    # sequential_parser.add_arguments(SequentialModel.HParams, "hparams")
    # sequential_parser.add_arguments(Config, dest="config")
    # sequential_parser.set_defaults(model=SequentialModel)

    # parallel_parser: ArgumentParser = subparsers.add_parser("parallel")
    # parallel_parser.add_arguments(ParallelModel.HParams, "hparams")
    # parallel_parser.add_arguments(Config, dest="config")
    # parallel_parser.set_defaults(model=ParallelModel)

    # NOTE: we unfortunately can't add the PL Trainer arguments directly atm, because they don't
    # play nice with simple-parsing.
    # trainer_parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    config: Config = args.config
    hparams: BaseModel.HParams = args.hparams
    if not sample_hparams:
        # Use the values from the command-line.
        hparams = args.hparams
    else:
        # Sample some hyper-parameters from the prior, and overwrite the sampled values with those
        # passed through the command-line.
        hparams = BaseModel.HParams.sample()
        default_hparams_dict = BaseModel.HParams().to_dict()
        cmd_hparams: BaseModel.HParams = args.hparams
        custom_hparams = {
            k: v for k, v in cmd_hparams.to_dict().items() if v != default_hparams_dict[k]
        }
        if custom_hparams:
            print(f"Overwriting sampled values for entries {custom_hparams}")
            hparams_dict = hparams.to_dict()
            hparams_dict.update(custom_hparams)
            hparams = BaseModel.HParams.from_dict(hparams_dict)
    print(f"Config:", config.dumps_json(indent="\t"))
    # print(f"HParams:", hparams)

    if config.seed is not None:
        seed = config.seed
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g = torch.Generator(device=config.device)
        seed = g.seed()
    print(f"Selected seed: {seed}")
    seed_everything(seed=seed, workers=True)

    logging.getLogger().setLevel(logging.INFO)
    if config.debug:
        print(f"Setting the max_epochs to 1, since '--debug' was passed.")
        hparams.max_epochs = 1
        logging.getLogger("target_prop").setLevel(logging.DEBUG)

    print("HParams:", hparams.dumps_json(indent="\t"))
    # Create the datamodule:
    datamodule = config.make_datamodule(batch_size=hparams.batch_size)

    model_class: Type[BaseModel] = args.model
    model = model_class(datamodule=datamodule, hparams=hparams, config=config)
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        gpus=torch.cuda.device_count(),
        track_grad_norm=False,
        accelerator=None,
        # max_steps=10 if config.debug else None, # NOTE: Can be useful when generating a lot of plots.
        # accelerator="ddp",
        # profiler="simple",
        # callbacks=[],
        # terminate_on_nan=True, # BUG: Would like to use this, but doesn't seem to work.
        logger=WandbLogger() if not config.debug else None,
    )
    trainer.fit(model, datamodule=datamodule)

    test_results = trainer.test(model, datamodule=datamodule)
    wandb.finish()
    print(test_results)
    test_accuracy: float = test_results[0]["test/accuracy"]
    return test_accuracy


if __name__ == "__main__":
    main()
