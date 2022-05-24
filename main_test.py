# ADAPTED FROM https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
from typing import List

import pytest
from hydra import compose, initialize, initialize_config_module
from omegaconf import OmegaConf

from main import Experiment, Options, main
from target_prop.datasets.dataset_config import DatasetConfig
from target_prop.models.model import Model
from target_prop.networks.simple_vgg import SimpleVGG


def test_defaults() -> None:
    with initialize(config_path="conf"):
        # config is relative to a module
        config = compose(config_name="config")
        options = OmegaConf.to_object(config)
        assert isinstance(options, Options)
        assert options.model == DTP.HParams()
        assert options.network == SimpleVGG.HParams()
        assert options.dataset == DatasetConfig()


from dataclasses import replace

from target_prop.models import DTP, BaselineModel, ParallelDTP, TargetProp, VanillaDTP


def _ids(v):
    if isinstance(v, list):
        return ",".join(map(str, v))
    return None


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (["model=dtp"], DTP.HParams()),
        (["model=parallel_dtp"], ParallelDTP.HParams()),
        (["model=target_prop"], TargetProp.HParams()),
        (["model=vanilla_dtp"], VanillaDTP.HParams()),
        (["model=backprop"], BaselineModel.HParams()),
        (
            ["model=dtp", "model.b_optim.lr=[123,456]"],
            DTP.HParams(b_optim=replace(DTP.HParams().b_optim, lr=[123, 456])),
        ),
    ],
    ids=_ids,
)
def test_setting_model(overrides: List[str], expected: Model.HParams) -> None:
    with initialize(config_path="conf"):
        config = compose(config_name="config", overrides=overrides)
        options = OmegaConf.to_object(config)
        assert isinstance(options, Options)
        assert options.model == expected


from target_prop.networks import LeNet, Network, ResNet18, ResNet34, SimpleVGG


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (["model=dtp", "network=lenet"], LeNet.HParams()),
        (["model=dtp", "network=simple_vgg"], SimpleVGG.HParams()),
        (["model=dtp", "network=resnet18"], ResNet18.HParams()),
        (["model=dtp", "network=resnet34"], ResNet34.HParams()),
        # (["network="], BaselineModel.HParams()),
    ],
    ids=_ids,
)
def test_setting_network(overrides: List[str], expected: Model.HParams) -> None:
    with initialize_config_module(config_module="conf"):
        config = compose(config_name="config", overrides=overrides)
        options = OmegaConf.to_object(config)
        assert isinstance(options, Options)
        assert options.network == expected


dtp_model_names = ["dtp", "target_prop", "vanilla_dtp"]
network_names = ["simple_vgg", "lenet", "resnet18", "resnet34"]


@pytest.mark.parametrize(
    "overrides",
    [
        [f"model={model_name}", f"network={network_name}"]
        for model_name in dtp_model_names
        for network_name in network_names
    ],
    ids=_ids,
)
def test_model_network_overrides_fixes_mismatch_in_number_of_values(overrides: List[str]) -> None:
    with initialize(config_path="conf"):
        config = compose(config_name="config", overrides=overrides)
        options = OmegaConf.to_object(config)
        assert isinstance(options, Options)
        assert isinstance(options.model, DTP.HParams)

        # Actually instantiate everything here.
        experiment = Experiment(options)
        assert isinstance(experiment.model, DTP)
        assert isinstance(experiment.model.hp, DTP.HParams)
        # NOTE: -1 since the first feedback layer (closest to the input x) isn't trained.
        n_layers_to_train = len(experiment.network) - 1
        assert len(options.model.feedback_training_iterations) == n_layers_to_train
        assert len(experiment.model.hp.feedback_training_iterations) == n_layers_to_train


# TODO: Add a simple test that overfits to a single batch, for every combination!


@pytest.mark.parametrize(
    "overrides",
    [
        [f"model={model_name}", f"network={network_name}"]
        for model_name in dtp_model_names
        for network_name in network_names
    ],
    ids=_ids,
)
def test_overfit_single_batch(overrides: list[str]) -> None:
    """Test that training with a single batch for many epochs makes it possible to learn that batch
    well.

    If this doesn't work, there's no point in trying to train for longer.
    """
    overrides = overrides + [
        "seed=123",
        "callbacks=no_checkpoints",
        "++trainer.enable_checkpointing=False",
        "++trainer.gpus=1",
        "++trainer.overfit_batches=1",
        "++trainer.limit_val_batches=0.0",
        "++trainer.limit_test_batches=0.0",
        "++trainer.max_epochs=50",
    ]
    print(f"overrides: {' '.join(overrides)}")
    with initialize(config_path="conf"):
        config = compose(
            config_name="config",
            overrides=overrides,
        )
        options = OmegaConf.to_object(config)
        assert isinstance(options, Options)

        # NOTE: Running the experiment manually like this so that we can get the number of classes
        # from the datamodule.
        # TODO: If we create the experiment from the options in another (nicer) way, this will need
        # to be updated.
        experiment = Experiment(options)
        assert hasattr(experiment.datamodule, "num_classes")
        num_classes: int = experiment.datamodule.num_classes  # type: ignore
        chance_accuracy = 1 / num_classes

        classification_error = experiment.run()
        accuracy = 1 - classification_error

        # NOTE: In this particular case, this error below is the training error, not the validation
        # error.
        # FIXME: This threshold is really low, it should be more like > 90% accuracy, but it's
        # currently taking a long time to get those values.
        assert accuracy > (chance_accuracy + 0.10)
