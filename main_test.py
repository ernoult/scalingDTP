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
from dataclasses import replace


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
)
def test_setting_model(overrides: List[str], expected: Model.HParams) -> None:
    with initialize_config_module(config_module="conf"):
        config = compose(config_name="config", overrides=overrides)
        options = OmegaConf.to_object(config)
        assert isinstance(options, Options)
        assert options.model == expected

