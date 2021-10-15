import logging
from typing import ClassVar, Type

import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from target_prop.config import Config
from target_prop.models import DTP


class TestDTP:
    # The type of model to test.
    model_class: ClassVar[Type[DTP]] = DTP

    @pytest.mark.parametrize("dataset", ["cifar10"])
    def test_fast_dev_run(self, dataset: str):
        """ Run a fast dev run using a single batch of data for training/validation/testing. """
        # NOTE: Not testing using other datasets for now, because the defaults on the HParams are
        # all made for Cifar10 (e.g. hard-set to 5 layers). This would make the test uglier, as we'd
        # have to pass different values for each dataset.
        config = Config(dataset=dataset, num_workers=0, debug=True)
        hparams = self.model_class.HParams()
        trainer = Trainer(
            max_epochs=1,
            gpus=torch.cuda.device_count(),
            track_grad_norm=False,
            accelerator=None,
            fast_dev_run=True,
            # max_steps=10 if config.debug else None, # NOTE: Can be useful when generating a lot of plots.
            # accelerator="ddp",  # todo: debug DP/DDP
            # profiler="simple",
            # callbacks=[],
            terminate_on_nan=False,  # bug: doesn't work with the SequentialModel (manual optim).
            logger=None,
        )

        if config.seed is not None:
            seed = config.seed
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            g = torch.Generator(device=device)
            seed = g.seed()
        print(f"Selected seed: {seed}")
        seed_everything(seed=seed, workers=True)

        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("target_prop").setLevel(logging.DEBUG)

        print("HParams:", hparams.dumps_json(indent="\t"))
        # Create the datamodule:
        datamodule = config.make_datamodule(batch_size=hparams.batch_size)

        model = self.model_class(datamodule=datamodule, hparams=hparams, config=config)
        trainer.fit(model, datamodule=datamodule)

        test_results = trainer.test(model, datamodule=datamodule)
        print(test_results)
        test_accuracy: float = test_results[0]["test/accuracy"]
        assert test_accuracy > 0
