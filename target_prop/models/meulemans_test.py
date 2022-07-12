"""Tests the for the Meuleman's model (DDTP)."""
from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from numpy.testing import assert_equal
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import Trainer
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from main import Experiment
from meulemans_dtp import main
from meulemans_dtp.lib.utils import FbOptimizerList, OptimizerList, choose_optimizer
from target_prop.config import MiscConfig
from target_prop.datasets.dataset_config import DATA_DIR
from target_prop.models.meulemans import _CIFAR10_ARGS, Meulemans, MeulemansNetwork


@pytest.fixture(scope="session")
def config():
    return MiscConfig(debug=True)


@pytest.fixture()
def trainer_kwargs(config: MiscConfig):
    return dict(
        enable_checkpointing=False,
        fast_dev_run=True,
        gpus=(1 if config.device == "cuda" else 0),
    )


@pytest.fixture(scope="session")
def data_dir():
    return str(DATA_DIR)


@pytest.fixture(scope="module")
def datamodule(data_dir: Path):
    """Returns the cifar10 datamodule, with the same normalization as in the Meulemans codebase."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (3 * 0.2023, 3 * 0.1994, 3 * 0.2010)),
        ]
    )
    return CIFAR10DataModule(
        num_workers=0,
        batch_size=_CIFAR10_ARGS.batch_size,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform,
        val_split=0.1,
    )


@pytest.fixture()
def network(datamodule: CIFAR10DataModule):
    # TODO: Make sure it's equivalent to this one:
    # return build_network(DEFAULT_ARGS)
    return MeulemansNetwork(
        in_channels=datamodule.dims[0],
        n_classes=datamodule.num_classes,
        hparams=None,
    )


def _remove_ndarrays(d: dict) -> dict:
    return {
        k: v.tolist()
        if isinstance(v, np.ndarray)
        else _remove_ndarrays(v)
        if isinstance(v, dict)
        else v
        for k, v in d.items()
    }


class TestEquivalence:
    """Tests that our version of the Meuleman's DTP model is equivalent to the original
    implementation.
    """

    def test_args_are_the_same(self):
        """Check that the 'Args' object loaded and used by the Meulemans codebase is the same as
        ours.
        """
        our_hparams = Meulemans.HParams()
        their_hparams = _CIFAR10_ARGS

        for key, their_value in vars(their_hparams).items():
            our_value = getattr(our_hparams, key)
            if key == "dataset":
                assert isinstance(our_value, main.DatasetOptions)
                assert our_value.dataset == their_value
            elif key == "out_dir":
                assert Path(their_value) == Path(our_value)
            else:
                # assert our_value == their_value, key
                np.testing.assert_equal(our_value, their_value)

    def test_args_from_hydra_are_the_same(self):
        """TODO: Test that the values we get through Hydra (e.g. by running
        `python main.py model=meulemans`) give the same values for the arguments as those in the meulemans codebase.

        For example, the batch size is a property of our datamodule, but is part of the root
        namespace in their codebase. It would be important to check that the values for such
        arguments are the same, at least when choosing to explicitly reproduce their results.
        We could also add one such option, like `python main.py model=meulemans_reproduce_exact`,
        which would override the other values to give us exactly what we want.
        """

        their_config = _CIFAR10_ARGS
        their_config_dict = their_config.to_dict()
        their_config_dict = _remove_ndarrays(their_config_dict)

        with initialize(config_path="../../conf"):
            config = compose(
                config_name="config",
                overrides=["model=meulemans", "network=meulemans"],
            )
            experiment = Experiment.from_options(config)

            our_config_dict = experiment.model.hp.to_dict()

            # Remove one level of nesting from the dictionary.
            our_flat_config_dict = {}
            for k, v in our_config_dict.items():
                if isinstance(v, dict):
                    for k_v, v_v in v.items():
                        if k_v in our_flat_config_dict:
                            # There shouldn't be any key collisions. But if there are, the values
                            # should be the same.
                            assert our_flat_config_dict[k_v] == v_v
                        our_flat_config_dict[k_v] = v_v
                else:
                    our_flat_config_dict[k] = v
            our_flat_config_dict = _remove_ndarrays(our_flat_config_dict)

            # Ignore some unused keys:
            assert our_flat_config_dict.pop("out_dir") == "././logs"
            assert their_config_dict.pop("out_dir") == "logs"
            assert our_flat_config_dict.pop("lr_scheduler") == None

            #  The 'lr' is different at this stage (scale_lr is applied in our config class, so we
            # see it here, but it's done in their optimizer class (I think), so after loading
            # these.
            # TODO: Double-check that we are only applying this scaling once, and that we do it at
            # the same stage as they do.
            our_target_stepsize = our_flat_config_dict["target_stepsize"]
            their_target_stepsize = their_config_dict["target_stepsize"]
            assert our_target_stepsize == their_target_stepsize
            our_lr: list[float] = our_flat_config_dict.pop("lr")
            their_lr: list[float] = their_config_dict.pop("lr")

            assert (np.array(their_lr) / np.array(our_lr)) == pytest.approx(our_target_stepsize)

            # All the other arguments / configuration options should be the same.
            assert our_flat_config_dict == their_config_dict

    @pytest.fixture(scope="session")
    def meulemans_cifar10_dataloaders(self, data_dir: str):
        """Creates the dataloaders for CIFAR10. Extracted from the meulemans codebase."""

        args = _CIFAR10_ARGS
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (3 * 0.2023, 3 * 0.1994, 3 * 0.2010)
                ),
            ]
        )

        trainset_total = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        if args.no_val_set:
            train_loader = DataLoader(
                trainset_total, batch_size=args.batch_size, shuffle=True, num_workers=0
            )
            val_loader = None
        else:
            # NOTE: This isn't exactly the same as in our case.
            g_cuda = torch.Generator(device="cpu")

            trainset, valset = random_split(trainset_total, [45000, 5000], generator=g_cuda)
            train_loader = DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g_cuda
            )
            val_loader = DataLoader(
                valset, batch_size=args.batch_size, shuffle=False, num_workers=0, generator=g_cuda
            )
        testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader, test_loader

    def test_data_is_the_same(
        self,
        meulemans_cifar10_dataloaders: tuple[DataLoader, DataLoader, DataLoader],
        datamodule: CIFAR10DataModule,
    ):
        datamodule.prepare_data()
        datamodule.setup()
        our_train_dataloader: DataLoader = datamodule.train_dataloader()
        our_valid_dataloader = datamodule.val_dataloader()
        our_test_dataloader = datamodule.test_dataloader()
        assert isinstance(our_valid_dataloader, DataLoader)
        assert isinstance(our_test_dataloader, DataLoader)

        (
            their_train_dataloader,
            their_valid_dataloader,
            their_test_dataloader,
        ) = meulemans_cifar10_dataloaders

        from itertools import islice

        from torchvision.transforms import Compose

        our_train_transforms: Compose = our_train_dataloader.dataset.dataset.transform  # type: ignore
        their_train_transforms: Compose = their_train_dataloader.dataset.dataset.transform  # type: ignore

        assert len(our_train_transforms.transforms) == len(their_train_transforms.transforms)
        for our_transform, their_transform in zip(
            our_train_transforms.transforms, their_train_transforms.transforms
        ):
            assert str(our_transform) == str(their_transform)

        for stage, our_dataloader, their_dataloader in zip(
            ["train", "val", "test"],
            [our_train_dataloader, our_valid_dataloader, our_test_dataloader],
            [their_train_dataloader, their_valid_dataloader, their_test_dataloader],
        ):
            assert our_dataloader.batch_size == their_dataloader.batch_size, stage
            assert len(our_dataloader) == len(their_dataloader), stage
            # NOTE: We don't check the individual examples are equal here, because they use a
            # different way to set the RNG than we do.
            for (our_x, our_y), (their_x, their_y) in islice(
                zip(our_train_dataloader, their_train_dataloader), 1
            ):
                assert our_x.shape == their_x.shape
                assert our_y.shape == their_y.shape
                # torch.testing.assert_allclose(our_x, their_x)
                # torch.testing.assert_allclose(our_y, their_y)
                # break

    def test_trainining_step_equivalent_to_train_parallel(
        self,
        network: MeulemansNetwork,
        datamodule: CIFAR10DataModule,
        config: MiscConfig,
    ):
        """TODO: Test that the training produces the exact same weights, given the same
        intialization."""

        train_dataloader = datamodule.train_dataloader()

        args = _CIFAR10_ARGS
        network.to(config.device)
        # TODO: Initialize the model the same way in both cases, and set the global RNG the same
        # way.
        x, y = next(iter(train_dataloader))
        x = x.to(config.device)
        y = y.to(config.device)

        forward_optimizers, feedback_optimizers = choose_optimizer(args, network)
        from meulemans_dtp.lib.train import (
            train_feedback_parameters,
            train_forward_parameters,
        )

        model_state = copy.deepcopy(network.state_dict())

        def get_optim_states(optimizerlist: OptimizerList | FbOptimizerList):
            return [copy.deepcopy(optim.state_dict()) for optim in optimizerlist._optimizer_list]

        from torch.optim.optimizer import Optimizer

        def set_optim_states(
            optimizerlist: list[Optimizer] | OptimizerList | FbOptimizerList, states: list[dict]
        ):
            optimizers = (
                optimizerlist if isinstance(optimizerlist, list) else optimizerlist._optimizer_list
            )
            for optimizer, state_dict in zip(optimizers, states):
                optimizer.load_state_dict(state_dict)
            return optimizerlist

        f_optim_states = get_optim_states(forward_optimizers)
        b_optim_states = get_optim_states(feedback_optimizers)

        with torch.random.fork_rng():
            torch.manual_seed(123)

            # TODO: This currently isn't working because of a flatten somewhere.
            # There are also probably some missing values that need to be saved in this "train_var"
            # train_var = argparse.Namespace()
            # train_var.forward_optimizer = forward_optimizers
            # train_var.feedback_optimizer = feedback_optimizers
            # train_parallel(
            #     args=args,
            #     train_var=train_var,
            #     device="cuda",
            #     train_loader=train_dataloader,
            #     net=network,
            #     writer=None,
            # )

            # NOTE: instead of just calling `train_parallel`, we extract the main portions here.

            predictions = network(x)
            train_forward_parameters(
                args,
                network,
                predictions,
                targets=y,
                loss_function=torch.nn.CrossEntropyLoss(),
                forward_optimizer=forward_optimizers,
            )
            if not args.freeze_fb_weights:
                train_feedback_parameters(args, network, feedback_optimizers)
            if not args.freeze_forward_weights:
                forward_optimizers.step()
        their_network_state = copy.deepcopy(network.state_dict())
        their_forward_optim_states = get_optim_states(forward_optimizers)
        their_feedback_optim_states = get_optim_states(feedback_optimizers)

        network.load_state_dict(model_state)
        model = Meulemans(
            datamodule=datamodule, network=network, hparams=Meulemans.HParams(), config=config
        )
        assert model.network is network
        model.optimizers = lambda: model.configure_optimizers()

        set_optim_states(model.forward_optimizers, f_optim_states)
        set_optim_states(model.feedback_optimizers, b_optim_states)

        training_outputs = model.training_step((x, y), 0)

        our_network_state = network.state_dict()
        our_forward_optim_states = [optim.state_dict() for optim in model.forward_optimizers]
        our_feedback_optim_states = [optim.state_dict() for optim in model.feedback_optimizers]
        # BUG: Not quite equal!
        assert_equal(our_network_state.values(), their_network_state.values())

        for our_optim_state, their_optim_state in zip(
            our_forward_optim_states, their_forward_optim_states
        ):
            assert_equal(our_optim_state, their_optim_state)
        for our_optim_state, their_optim_state in zip(
            our_feedback_optim_states, their_feedback_optim_states
        ):
            assert_equal(our_optim_state, their_optim_state)


class TestMeulemans:
    """Tests specific to the Meulemans model."""

    def test_forward_gives_predicted_logits(
        self, datamodule: CIFAR10DataModule, network: MeulemansNetwork, config: MiscConfig
    ):
        """Test that the model gives the prediction logits in `forward`."""
        assert hasattr(datamodule, "num_classes")
        num_classes = datamodule.num_classes
        model = Meulemans(
            datamodule=datamodule, network=network, hparams=Meulemans.HParams(), config=config
        )
        model.to(config.device)
        x = torch.rand([32, *datamodule.dims]).to(config.device)
        logits = model(x)
        assert isinstance(logits, Tensor)
        assert logits.shape == (32, num_classes)

    def test_fast_dev_run(
        self,
        datamodule: CIFAR10DataModule,
        network: MeulemansNetwork,
        config: MiscConfig,
        trainer_kwargs: dict,
    ):
        trainer_kwargs.update(fast_dev_run=True)
        trainer = Trainer(**trainer_kwargs)
        model = Meulemans(
            datamodule=datamodule, network=network, hparams=Meulemans.HParams(), config=config
        )

        trainer.fit(model, datamodule=datamodule)
        eval_performance = trainer.validate(model, datamodule=datamodule)
        assert eval_performance
        test_performance = trainer.test(model, datamodule=datamodule)

    def test_calculates_loss_from_batch(
        self,
        datamodule: CIFAR10DataModule,
        network: MeulemansNetwork,
        config: MiscConfig,
        trainer_kwargs: dict,
    ):
        """Tests that the model makes a prediction."""
        # trainer = Trainer(**trainer_kwargs, fast_dev_run=True)
        model = Meulemans(
            datamodule=datamodule, network=network, hparams=Meulemans.HParams(), config=config
        )
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
            step_output = model.training_step((x, y), batch_idx=batch_index)
            assert "loss" not in step_output
            # loss = step_output["loss"]
            # assert isinstance(loss, Tensor)
            # # Since we're not using automatic optimization, the loss shouldn't have a gradient.
            # assert not loss.requires_grad
            # assert loss.shape == ()
            # assert loss != 0.0
