from __future__ import annotations

import sys
import typing
from dataclasses import replace
from typing import Any, ClassVar

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

import main
from main import Experiment, Options
from target_prop.datasets.dataset_config import cifar10_config
from target_prop.models import DTP, BaselineModel, ParallelDTP, TargetProp, VanillaDTP
from target_prop.models.model import Model
from target_prop.networks import LeNet, ResNet18, ResNet34, SimpleVGG
from target_prop.networks.simple_vgg import SimpleVGG

if typing.TYPE_CHECKING:
    from _pytest.mark import ParameterSet

# ADAPTED FROM https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py


TEST_SEED = 123

from typing import Callable, Optional

import torch
from sklearn.datasets import make_classification
from torch.utils.data import TensorDataset
from torchvision.datasets import VisionDataset


class DummyDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train: bool = True,
        download: bool = False,
    ):
        X, y = make_classification(
            n_features=32 * 32 * 3,
            n_repeated=32 * 32 * 2,
            n_informative=10,
            n_classes=10,
            n_clusters_per_class=1,
            n_samples=1000,
            random_state=TEST_SEED,
        )
        X = X.reshape(-1, 3, 32, 32)
        # X *= 256
        # X = X.astype("uint8")
        self.data = TensorDataset(
            torch.as_tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.long)
        )
        super().__init__(
            root=root, transforms=transforms, transform=transform, target_transform=target_transform
        )

    def __getitem__(self, index: Any):
        x, y = self.data[index]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def __len__(self) -> int:
        return len(self.data)


from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from torchvision.datasets import VisionDataset


class DummyDataModule(VisionDataModule):
    dataset_cls: type[VisionDataset] = DummyDataset
    dims: ClassVar[tuple[int, int, int]] = (3, 32, 32)
    num_classes: ClassVar[int] = 10


@pytest.fixture(autouse=True, scope="session")
def dummy_datamodule():
    from torchvision.transforms import Compose, Normalize

    from main import cs
    from target_prop.utils.hydra_utils import builds

    datamodule = builds(
        DummyDataModule,
        train_transforms=builds(
            Compose, transforms=[builds(Normalize, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        ),
    )
    cs.store(
        group="dataset",
        name="dummy",
        node=datamodule,
    ),


@pytest.fixture
def testing_overrides():
    """Fixture that gives normal command-line overrides to use during unit testing."""
    return [
        f"seed={TEST_SEED}",
        "callbacks=no_checkpoints",
        "trainer=debug",
    ]


@pytest.fixture(autouse=True, scope="session")
def set_testing_hydra_dir():
    """TODO: Set the hydra configuration for unit testing, so temporary directories are used.

    NOTE: Might be a good idea to look in `hydra.test_utils` for something useful, e.g.
    `from hydra.test_utils.test_utils import integration_test`
    """


def test_defaults() -> None:
    with initialize(config_path="conf"):
        # config is relative to a module
        config = compose(config_name="config")
        options = OmegaConf.to_object(config)
        assert isinstance(options, Options)
        assert options.model == DTP.HParams()
        assert options.network == SimpleVGG.HParams()
        # NOTE: The equality check is failing with these objects, probably because it creates a
        # class twice or something. But it's the same in yaml form, so it's all good.
        # assert options.dataset == cifar10_config
        assert OmegaConf.to_yaml(options.dataset) == OmegaConf.to_yaml(cifar10_config)


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
def test_setting_model(
    overrides: list[str], expected: Model.HParams, testing_overrides: list[str]
) -> None:
    with initialize(config_path="conf"):
        config = compose(config_name="config", overrides=testing_overrides + overrides)
        assert config.seed == TEST_SEED  # note: from the testing_overrides above.
        options = OmegaConf.to_object(config)
        assert isinstance(options, Options)
        assert options.model == expected


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
def test_setting_network(
    overrides: list[str], expected: Model.HParams, testing_overrides: list[str]
) -> None:
    # NOTE: Still unclear on the difference between initialize and initialize_config_module
    with initialize(config_path="conf"):
        config = compose(config_name="config", overrides=testing_overrides + overrides)
        options = OmegaConf.to_object(config)
        assert isinstance(options, Options)
        assert options.network == expected


# TODO: Determine this programmatically, probably using the ConfigStore API.
model_names: list[str | ParameterSet] = ["dtp", "backprop", "parallel_dtp"]
model_names.extend(
    pytest.param(
        model_name,
        marks=pytest.mark.skipif(
            "-vvv" not in sys.argv,
            reason="Not testing these models atm",
        ),
    )
    for model_name in ("target_prop", "vanilla_dtp")
)
dtp_model_names = ["dtp", "target_prop", "vanilla_dtp"]
network_names = ["simple_vgg", "lenet", "resnet18", "resnet34"]


@pytest.mark.parametrize("network_name", network_names)
@pytest.mark.parametrize("model_name", dtp_model_names)
def test_model_network_overrides_fixes_mismatch_in_number_of_values(
    model_name: str, network_name: str, testing_overrides: list[str]
) -> None:
    """TODO: Move this to DTP tests."""
    with initialize(config_path="conf"):
        config = compose(
            config_name="config",
            overrides=[f"model={model_name}", f"network={network_name}"] + testing_overrides,
        )
        options = OmegaConf.to_object(config)
        assert isinstance(options, Options)
        assert isinstance(options.model, DTP.HParams)

        # Actually instantiate everything here.
        experiment = Experiment.from_options(options)
        assert isinstance(experiment.model, DTP)
        assert isinstance(experiment.model.hp, DTP.HParams)
        # NOTE: -1 since the first feedback layer (closest to the input x) isn't trained.
        n_layers_to_train = len(experiment.network) - 1
        assert len(options.model.feedback_training_iterations) == n_layers_to_train
        assert len(experiment.model.hp.feedback_training_iterations) == n_layers_to_train


@pytest.mark.parametrize("network_name", network_names)
@pytest.mark.parametrize("model_name", model_names)
def test_experiment_reproducible_given_seed(
    network_name: str, model_name: str, testing_overrides: list[str]
) -> None:
    """Test that training with a single batch gives exactly the same results."""

    all_overrides = (
        [f"model={model_name}", f"network={network_name}"]
        + [
            "++trainer.limit_train_batches=10",
            "++trainer.limit_val_batches=5",
            "++trainer.limit_test_batches=0",
            "++trainer.fast_dev_run=False",
            "++trainer.max_epochs=1",
        ]
        + testing_overrides
    )

    with initialize(config_path="conf"):
        config = compose(
            config_name="config",
            overrides=all_overrides,
        )
        first_value = main.main(config)
        other_value_in_same_context = main.main(config)
        assert first_value == other_value_in_same_context

    with initialize(config_path="conf"):
        config = compose(
            config_name="config",
            overrides=all_overrides,
        )
        another_value = main.main(config)
        assert another_value == pytest.approx(first_value)

    # NOTE: Second part of the test, which isn't as reliable or useful (e.g since some algos could
    # be deterministic)
    # with initialize(config_path="conf"):
    #     config = compose(
    #         config_name="config",
    #         overrides=all_overrides + [f"++seed={TEST_SEED+ 1234}"],
    #     )
    #     assert config.seed == TEST_SEED + 1234
    #     value_with_different_seed = main(config)
    #     assert value_with_different_seed != pytest.approx(first_value)


@pytest.mark.parametrize("network_name", network_names)
@pytest.mark.parametrize("model_name", model_names)
def test_overfit_single_batch(
    model_name: str, network_name: str, testing_overrides: list[str]
) -> None:
    """Test that training with a single batch for multiple iterations makes it possible to learn
    that batch well.

    If this doesn't work, there isn't really a point in trying to train for longer.
    """
    # Number of training iterations (NOTE: each iteration is one call to training_step, which
    # itself may do more than a single update, e.g. in the case of DTP).
    num_training_iterations = 30
    # By how much the model should be better than chance accuracy to pass this test.

    # FIXME: This threshold is really low, we should expect more like > 90% accuracy, but it's
    # currently taking a long time to get those values.
    better_than_chance_threshold_pct = 0.10

    all_overrides = testing_overrides + [
        f"model={model_name}",
        f"network={network_name}",
        # NOTE: Could use something like this to make the tests a bit quicker to run, by requiring
        # fewer iterations to learn than cifar10!
        # f"dataset=dummy",
    ]
    print(f"overrides: {' '.join(all_overrides)}")
    with initialize(config_path="conf"):
        config = compose(
            config_name="config",
            overrides=all_overrides,
        )
        config.trainer.overfit_batches = 1
        config.trainer.limit_val_batches = 0.0
        config.trainer.limit_test_batches = 0.0
        config.trainer.fast_dev_run = False
        config.trainer.max_epochs = num_training_iterations

        options = OmegaConf.to_object(config)
        assert isinstance(options, Options)

        # NOTE: Running the experiment manually like this so that we can get the number of classes
        # from the datamodule.
        # TODO: If we create the experiment from the options in another (nicer) way, this will need
        # to be updated.
        experiment = Experiment.from_options(options)
        assert hasattr(experiment.datamodule, "num_classes")
        num_classes: int = experiment.datamodule.num_classes  # type: ignore
        chance_accuracy = 1 / num_classes

        classification_error = experiment.run()
        accuracy = 1 - classification_error

        # NOTE: In this particular case, this error below is the training error, not the validation
        # error.
        assert accuracy > (chance_accuracy + better_than_chance_threshold_pct)


# TODO: Add some more integration tests:
# - running sweeps from Hydra!
# - using the slurm launcher!
