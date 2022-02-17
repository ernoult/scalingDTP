# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ADAPTED FROM https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
from typing import List
from omegaconf import OmegaConf
import pytest

from main import main, Options
from hydra import compose, initialize, initialize_config_module
from target_prop.datasets.dataset_config import DatasetConfig

from target_prop.models.model import Model
from target_prop.networks.simple_vgg import SimpleVGG


# 1. initialize will add config_path the config search path within the context
# 2. The module with your configs should be importable. it needs to have a __init__.py (can be empty).
# 3. The config path is relative to the file calling initialize (this file)
def test_defaults() -> None:
    with initialize(config_path="conf"):
        # config is relative to a module
        config = compose(config_name="config")
        options = OmegaConf.to_object(config)
        assert isinstance(options, Options)
        assert options.model == DTP.HParams()
        assert options.network == SimpleVGG.HParams()
        assert options.dataset == DatasetConfig()


from target_prop.models import DTP, ParallelDTP, VanillaDTP, TargetProp, BaselineModel


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (["model=dtp"], DTP.HParams()),
        (["model=parallel_dtp"], ParallelDTP.HParams()),
        (["model=target_prop"], TargetProp.HParams()),
        (["model=vanilla_dtp"], VanillaDTP.HParams()),
        (["model=backprop"], BaselineModel.HParams()),
    ],
)
def test_user_logic(overrides: List[str], expected: Model.HParams) -> None:
    with initialize_config_module(config_module="conf"):
        config = compose(config_name="config", overrides=overrides)
        options = OmegaConf.to_object(config)
        assert isinstance(options, Options)
        assert options.model == expected

