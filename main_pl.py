""" Script that runs Pytorch lightning version of the prototype DTP model.

Use `python main.py --help` for a list of all available arguments,
    
    or (even better), take a look at the [`Config`](target_prop/config.py) and the
    [`Model.HParams`](target_prop/model.py) classes to see their definition.
"""
from target_prop.model import Model
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
    parser.add_arguments(Config, dest="config")
    parser.add_arguments(Model.HParams, dest="hparams")

    # TODO: we unfortunately can't do this directly atm:
    # trainer_parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    config: Config = args.config
    hparams: Model.HParams
    if sample_hparams:
        hparams = Model.HParams.sample()
        # TODO: overwrite any sampled values with those set from the command-line
        default_hparams_dict = Model.HParams().to_dict()
        cmd_hparams: Model.HParams = args.hparams
        custom_hparams = {
            k: v
            for k, v in cmd_hparams.to_dict().items()
            if v != default_hparams_dict[k]
        }
        if custom_hparams:
            print(f"Overwriting sampled values for entries {custom_hparams}")
            hparams_dict = hparams.to_dict()
            hparams_dict.update(custom_hparams)
            hparams = Model.HParams.from_dict(hparams_dict)
    else:
        hparams = args.hparams

    print(f"Config:", config.dumps_json(indent="\t"))
    # print(f"HParams:", hparams)

    print("HParams:", json.dumps(hparams.to_dict(), indent="\t"))
    # Create the datamodule:
    datamodule = config.make_datamodule(batch_size=hparams.batch_size)
    model = Model(datamodule=datamodule, hparams=hparams, config=config)
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        gpus=torch.cuda.device_count(),
        track_grad_norm=False,
        accelerator="ddp",
        # profiler="simple",
        # callbacks=[],
        logger=WandbLogger() if not config.debug else None,
    )
    b = trainer.fit(model, datamodule=datamodule)
    print(f"fit returned {b}")

    test_results = trainer.test(model, datamodule=datamodule)
    wandb.finish()
    print(test_results)
    test_accuracy: float = test_results[0]["test/Accuracy"]
    return test_accuracy


if __name__ == "__main__":
    main()