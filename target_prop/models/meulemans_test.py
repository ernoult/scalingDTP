"""Tests the for the Meuleman's model (DDTP)."""


import itertools

import pytest
import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import Trainer
from torch import Tensor
from torchvision.datasets.fakedata import FakeData
from torchvision.transforms import ToTensor

from target_prop.config import Config
from target_prop.models.meulemans import Meulemans


@pytest.fixture(scope="session")
def dummy_datamodule():
    # Create a dummy datamodule
    image_size = (3, 32, 32)
    num_classes = 10
    transform = ToTensor()
    datamodule = VisionDataModule.from_datasets(
        train_dataset=FakeData(
            size=100, num_classes=num_classes, image_size=image_size, transform=transform
        ),
        val_dataset=FakeData(
            size=100, num_classes=num_classes, image_size=image_size, transform=transform
        ),
        test_dataset=FakeData(
            size=100, num_classes=num_classes, image_size=image_size, transform=transform
        ),
    )
    datamodule.dims = image_size
    datamodule.num_classes = num_classes
    datamodule.prepare_data = lambda *args, **kwargs: None
    datamodule.setup = lambda *args, **kwargs: None
    return datamodule


from target_prop.datasets.dataset_config import DatasetConfig


@pytest.fixture(scope="module")
def datamodule():
    return DatasetConfig(dataset="cifar10", num_workers=0).make_datamodule(batch_size=32)


@pytest.fixture(scope="session")
def dummy_datamodule():
    # Create a dummy datamodule
    image_size = (3, 32, 32)
    num_classes = 10
    transform = ToTensor()
    datamodule = VisionDataModule.from_datasets(
        train_dataset=FakeData(
            size=100, num_classes=num_classes, image_size=image_size, transform=transform
        ),
        val_dataset=FakeData(
            size=100, num_classes=num_classes, image_size=image_size, transform=transform
        ),
        test_dataset=FakeData(
            size=100, num_classes=num_classes, image_size=image_size, transform=transform
        ),
    )
    datamodule.dims = image_size
    datamodule.num_classes = num_classes
    datamodule.prepare_data = lambda *args, **kwargs: None
    datamodule.setup = lambda *args, **kwargs: None
    return datamodule


@pytest.fixture(scope="session")
def config():
    return Config(debug=True)


@pytest.fixture()
def trainer_kwargs(config: Config):
    return dict(
        enable_checkpointing=False,
        fast_dev_run=True,
        gpus=(1 if config.device == "cuda" else 0),
    )


from pl_bolts.datamodules import CIFAR10DataModule


class TestMeulemans:
    """Tests specific to the Meulemans model."""

    def test_forward_gives_predicted_logits(self, datamodule: VisionDataModule, config: Config):
        """Test that the model gives the prediction logits in `forward`."""
        assert hasattr(datamodule, "num_classes")
        num_classes = datamodule.num_classes
        model = Meulemans(datamodule=datamodule, hparams=Meulemans.HParams(), config=config)
        model.to(config.device)
        x = torch.rand([32, *datamodule.dims]).to(config.device)
        logits = model(x)
        assert isinstance(logits, Tensor)
        assert logits.shape == (32, num_classes)

    def test_fast_dev_run(
        self, datamodule: CIFAR10DataModule, config: Config, trainer_kwargs: dict
    ):
        trainer_kwargs.update(fast_dev_run=True)
        trainer = Trainer(**trainer_kwargs)
        model = Meulemans(datamodule=datamodule, hparams=Meulemans.HParams(), config=config)

        trainer.fit(model, datamodule=datamodule)
        eval_performance = trainer.validate(model, datamodule=datamodule)
        assert eval_performance
        test_performance = trainer.test(model, datamodule=datamodule)

    def test_calculates_loss_from_batch(
        self, datamodule: VisionDataModule, config: Config, trainer_kwargs: dict
    ):
        """Tests that the model makes a prediction."""
        # trainer = Trainer(**trainer_kwargs, fast_dev_run=True)
        model = Meulemans(datamodule=datamodule, hparams=Meulemans.HParams(), config=config)
        # Tricky to test the batch manually, since we need to attach a Trainer, or mock the self.optimizers() method.
        model.optimizers = lambda: model.configure_optimizers()

        # model.trainer = trainer
        model.to(config.device)
        datamodule.prepare_data()
        datamodule.setup()

        for batch_index, batch in enumerate(
            itertools.islice(datamodule.train_dataloader(batch_size=32), 5)
        ):
            x, y = batch
            x = x.to(model.device)
            y = y.to(model.device)
            loss = model.training_step((x, y), batch_idx=batch_index)
            print(loss)
            assert isinstance(loss, Tensor)
            assert loss.requires_grad
            assert loss.shape == ()
            assert loss != 0.0
