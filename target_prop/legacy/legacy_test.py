import logging
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import ClassVar, Iterable, List, Optional, Tuple, Type

import pytest
from simple_parsing.helpers.serialization.serializable import Serializable
import torch
from itertools import islice
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from simple_parsing.helpers import choice, list_field
from target_prop._weight_operations import init_symetric_weights
from target_prop.backward_layers import mark_as_invertible
from target_prop.config import Config
from target_prop.layers import Reshape, forward_all, invert
from target_prop.legacy import (
    VGG,
    createDataset,
    createOptimizers,
    train_backward,
    train_batch,
    train_forward,
)
from target_prop.metrics import compute_dist_angle
from target_prop.models import DTP
from target_prop.models.dtp import FeedbackOptimizerConfig, ForwardOptimizerConfig
from target_prop.networks.simple_vgg import SimpleVGG
from target_prop.utils.utils import is_trainable, named_trainable_parameters
from torch import Tensor, nn
from torch.nn import functional as F
from target_prop.datasets.dataset_config import DatasetConfig, get_datamodule


@dataclass
class LegacyHparams(Serializable):
    """Hyper-Parameters used in @ernoult's VGG model."""

    # Channels per conv layer.
    C: List[int] = list_field(128, 128, 256, 256, 512)

    # Number of training steps for the feedback weights per batch. A list of
    # integers, where each value represents the number of iterations for that layer.
    iter: List[int] = list_field(20, 30, 35, 55, 20)

    # The scale of the gaussian random variable in the feedback loss calculation.
    noise: List[float] = list_field(0.4, 0.4, 0.2, 0.2, 0.08)

    # Learning rate for feed forward weights
    lr_f: float = 0.08

    # Learning rates for feedback weights
    lr_b: List[float] = list_field(1e-4, 3.5e-4, 8e-3, 8e-3, 0.18)

    # Type of activation to use.
    activation: str = "elu"

    # Nudging parameter: Used when calculating the first target.
    beta: float = 0.7

    # Weight decay for optimizer
    wdecay: float = 1e-4

    # Batch size
    batch_size = 128


@pytest.fixture
def legacy_hparams():
    return LegacyHparams()


@pytest.fixture
def legacy_model(legacy_hparams: LegacyHparams):
    legacy_model = VGG(legacy_hparams)
    return legacy_model


@pytest.fixture
def pl_hparams(legacy_hparams: LegacyHparams):
    # NOTE: The default value for the forward optimizer learning rate was lowered. Here we make it
    # match the legacy parameters.
    return DTP.HParams(
        f_optim=ForwardOptimizerConfig(
            type="sgd", lr=legacy_hparams.lr_f, weight_decay=legacy_hparams.wdecay
        ),
        b_optim=FeedbackOptimizerConfig(type="sgd", lr=legacy_hparams.lr_b),
        noise=legacy_hparams.noise,
        beta=legacy_hparams.beta,
        feedback_training_iterations=legacy_hparams.iter,
    )


@pytest.fixture
def pl_model(pl_hparams: DTP.HParams):
    config = Config(debug=True, seed=123, device="cpu")
    dataset_config = DatasetConfig(dataset="cifar10", num_workers=0)
    datamodule = dataset_config.make_datamodule(batch_size=pl_hparams.batch_size)
    network = SimpleVGG(in_channels=datamodule.dims[0], n_classes=datamodule.num_classes)
    pl_model = DTP(datamodule=datamodule, hparams=pl_hparams, config=config, network=network)
    return pl_model


class TestLegacyCompatibility:
    def test_input_batches_are_same(self, pl_hparams: DTP.HParams, legacy_hparams: LegacyHparams):
        check_mark = "\u2705"
        cross_mark = "\u274C"
        num_iterations = 5
        legacy_hparams.batch_size = 200
        pl_hparams.batch_size = 200

        # Get legacy dataloaders
        legacy_train_loader, legacy_test_loader = createDataset(legacy_hparams)

        # Get PL dataloaders
        # dataset_config = DatasetConfig(dataset="cifar10_noval", num_workers=1)
        datamodule = get_datamodule(
            dataset="cifar10_noval",
            num_workers=1,
            batch_size=pl_hparams.batch_size,
            use_legacy_std=True,
        )
        datamodule.setup()
        pl_train_loader, pl_test_loader = (
            datamodule.train_dataloader(),
            datamodule.test_dataloader(),
        )
        assert isinstance(pl_test_loader, DataLoader)

        # Ensure that number of batches in each split are equal
        assert len(legacy_train_loader) == len(pl_train_loader)
        assert len(legacy_test_loader) == len(pl_test_loader)

        # Compare data and labels
        # NOTE: We only compare `test` baches here because they are not shuffled in dataloader
        data_errors = []
        label_errors = []

        def assert_transforms_are_the_same(pl_loader: DataLoader, legacy_loader: DataLoader):
            legacy_transforms = legacy_loader.dataset.transform.transforms
            pl_transforms = pl_loader.dataset.transform.transforms
            assert len(legacy_transforms) == len(pl_transforms)
            assert str(legacy_transforms) == str(pl_transforms)

        assert_transforms_are_the_same(pl_train_loader, legacy_train_loader)
        assert_transforms_are_the_same(pl_test_loader, legacy_test_loader)

        for i, (legacy_batch, pl_batch) in islice(
            enumerate(zip(legacy_test_loader, pl_test_loader)), num_iterations
        ):
            assert legacy_batch[0].shape == pl_batch[0].shape
            data_error = torch.abs((legacy_batch[0] - pl_batch[0]).sum()).item()
            label_error = torch.abs((legacy_batch[1] - pl_batch[1]).sum()).item()
            print(
                f"[Batch {i}] L1 error between legacy and PL batch (data, label): {data_error}, {label_error} ",
                end="",
                flush=True,
            )
            print(check_mark) if data_error + label_error < 1e-5 else print(cross_mark)
            data_errors.append(data_error)
            label_errors.append(label_error)

        assert (sum(data_errors)) == 0
        assert (sum(label_errors)) == 0

    def test_forward_passes_are_same(self, pl_model: DTP, legacy_model: VGG):
        seed_everything(seed=123, workers=True)

        # Initialize both the models with same weights
        self._initialize_pl_from_legacy(legacy_model, pl_model)

        # Do forward pass through both the models and check if outputs exactly match
        batch_size = 16
        example_inputs = torch.rand([batch_size, 3, 32, 32])
        legacy_outputs = legacy_model(example_inputs)
        pl_outputs = pl_model.forward_net(example_inputs)
        assert torch.equal(legacy_outputs, pl_outputs)

    def test_forward_updates_are_same(
        self,
        pl_model: DTP,
        legacy_model: VGG,
        pl_hparams: DTP.HParams,
        legacy_hparams: LegacyHparams,
    ):
        seed_everything(seed=123, workers=True)

        # Initialize both the models with same weights
        forward_mapping, backward_mapping = self._initialize_pl_from_legacy(legacy_model, pl_model)

        # Generate random inputs and labels
        batch_size = 16
        num_classes = 10
        check_mark = "\u2705"
        cross_mark = "\u274C"
        example_inputs = torch.rand([batch_size, 3, 32, 32])
        example_labels = torch.randint(0, num_classes, [batch_size])

        # Compute forward layer losses in both the models
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        optimizers = createOptimizers(legacy_model, legacy_hparams, forward=True)
        optimizer_f = optimizers[0]
        _, legacy_layer_losses = train_forward(
            legacy_model, example_inputs, example_labels, criterion, optimizer_f, legacy_hparams
        )
        forward_optimizer = pl_hparams.f_optim.make_optimizer(pl_model.forward_net)
        pl_model._forward_optimizer = forward_optimizer
        pl_output = pl_model.forward_loss(example_inputs, example_labels, phase="train")
        pl_forward_loss, pl_layer_losses = pl_output["loss"], pl_output["layer_losses"]
        pl_forward_loss.backward()
        forward_optimizer.step()

        # Ensure that layer losses are equal
        for i, (legacy_layer_loss, pl_layer_loss) in enumerate(
            zip(legacy_layer_losses, pl_layer_losses)
        ):
            legacy_layer_loss = legacy_layer_loss.detach()
            pl_layer_loss = pl_layer_loss.detach()
            diff = torch.abs(legacy_layer_loss - pl_layer_loss)
            print(
                f"[Layer {i}] Legacy loss: {legacy_layer_loss}, PL loss: {pl_layer_loss}, Diff: {diff} ",
                end="",
                flush=True,
            )
            # If losses match closely, display check mark otherwise cross mark for easy inspection
            print(check_mark) if torch.allclose(legacy_layer_loss, pl_layer_loss) else print(
                cross_mark
            )

        # Ensure that weights after forward update are equal
        # NOTE: We don't check if gradients are equal since legacy model clears gradients
        # after each layer backward pass. However, if layer losses are equal, gradients
        # should be equal given that weights are initialized same in both the models
        errors = self._compare_weights(legacy_model, pl_model, forward_mapping)
        assert sum(errors) < 1e-4

    # @pytest.mark.xfail(
    #     reason="TODO: Need to adapt this test a bit since there's now one optim per feedback layer."
    # )
    def test_feedback_updates_are_same(
        self,
        pl_model: DTP,
        legacy_model: VGG,
        pl_hparams: DTP.HParams,
        legacy_hparams: LegacyHparams,
    ):
        seed_everything(seed=123, workers=True)

        # Initialize both the models with same weights
        forward_mapping, backward_mapping = self._initialize_pl_from_legacy(legacy_model, pl_model)

        # Generate random inputs and labels
        batch_size = 16
        num_classes = 10
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        example_inputs = torch.rand([batch_size, 3, 32, 32]).to(device)
        example_labels = torch.randint(0, num_classes, [batch_size]).to(device)
        legacy_model = legacy_model.to(device)
        pl_model.to(device)

        # Do feedback updates in legacy model
        # Save random state for sampling same noise vectors in both the models
        if torch.cuda.is_available():
            rng_state = torch.cuda.get_rng_state(device)
        else:
            rng_state = torch.get_rng_state()
        createOptimizers(legacy_model, legacy_hparams, forward=True)
        train_backward(legacy_model, example_inputs)

        # Do feedback updates in PL model
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state, device)
        else:
            torch.set_rng_state(rng_state)

        feedback_optimizers = []
        for i, (feedback_layer, lr) in enumerate(zip(pl_model.backward_net, pl_model.feedback_lrs)):
            layer_optimizer = None
            if i == (len(pl_model.backward_net) - 1) or not is_trainable(feedback_layer):
                assert lr == 0.0, (i, lr, pl_model.feedback_lrs, type(feedback_layer))
            else:
                assert lr != 0.0
                layer_optimizer = pl_model.hp.b_optim.make_optimizer(feedback_layer, lrs=[lr])
            feedback_optimizers.append(layer_optimizer)
        pl_model._feedback_optimizers = feedback_optimizers
        pl_model.feedback_loss(example_inputs, example_labels, phase="train")

        # Compare weights
        errors = self._compare_weights(legacy_model, pl_model, backward_mapping)
        assert sum(errors) < 1e-4

    def _initialize_pl_from_legacy(self, legacy_model, pl_model):
        # Get state dict mapping PL -> legacy
        forward_mapping, backward_mapping = self._get_pl_legacy_dict_mapping(
            legacy_model=legacy_model, pl_model=pl_model
        )

        # Copy legacy model weights to PL model
        pl_dict = pl_model.state_dict()
        legacy_dict = legacy_model.state_dict()
        for pl_key, legacy_key in forward_mapping.items():
            pl_dict[pl_key] = legacy_dict[legacy_key].clone()

        for pl_key, legacy_key in backward_mapping.items():
            pl_dict[pl_key] = legacy_dict[legacy_key].clone()
        pl_model.load_state_dict(pl_dict)
        return forward_mapping, backward_mapping

    def _compare_weights(self, legacy_model, pl_model, mapping):
        check_mark = "\u2705"
        cross_mark = "\u274C"
        pl_dict = pl_model.state_dict()
        legacy_dict = legacy_model.state_dict()
        errors = []
        for pl_key, legacy_key in mapping.items():
            pl_param = pl_dict[pl_key]
            legacy_param = legacy_dict[legacy_key]
            assert pl_param.shape == legacy_param.shape
            error = torch.abs((pl_param - legacy_param).sum()).item()
            errors.append(error)
            print(
                f"[Param: {pl_key}] L1 error between legacy and PL param after update: {error} ",
                end="",
                flush=True,
            )
            # If params match closely, display check mark otherwise cross mark for easy inspection
            print(check_mark) if error < 1e-5 else print(cross_mark)
        return errors

    def _get_pl_legacy_dict_mapping(self, legacy_model, pl_model):
        forward_mapping = {}
        backward_mapping = {}

        legacy_dict = legacy_model.state_dict()
        pl_dict = pl_model.state_dict()
        for i, layer in enumerate(legacy_model.layers):
            if isinstance(layer.f, nn.Conv2d):
                forward_mapping[f"forward_net.conv_{i}.conv.weight"] = f"layers.{i}.f.weight"
                forward_mapping[f"forward_net.conv_{i}.conv.bias"] = f"layers.{i}.f.bias"
                assert (
                    pl_dict[f"forward_net.conv_{i}.conv.weight"].shape
                    == legacy_dict[f"layers.{i}.f.weight"].shape
                )
                assert (
                    pl_dict[f"forward_net.conv_{i}.conv.bias"].shape
                    == legacy_dict[f"layers.{i}.f.bias"].shape
                )

                # Legacy network doesn't have backward weights in first layer
                if i > 0:
                    backward_mapping[f"backward_net.conv_{i}.conv.weight"] = f"layers.{i}.b.weight"
                    backward_mapping[f"backward_net.conv_{i}.conv.bias"] = f"layers.{i}.b.bias"
                    assert (
                        pl_dict[f"backward_net.conv_{i}.conv.weight"].shape
                        == legacy_dict[f"layers.{i}.b.weight"].shape
                    )
                    assert (
                        pl_dict[f"backward_net.conv_{i}.conv.bias"].shape
                        == legacy_dict[f"layers.{i}.b.bias"].shape
                    )

            elif isinstance(layer.f, nn.Linear):
                forward_mapping["forward_net.fc.linear.weight"] = f"layers.{i}.f.weight"
                forward_mapping["forward_net.fc.linear.bias"] = f"layers.{i}.f.bias"
                assert (
                    pl_dict[f"forward_net.fc.linear.weight"].shape
                    == legacy_dict[f"layers.{i}.f.weight"].shape
                )
                assert (
                    pl_dict[f"forward_net.fc.linear.bias"].shape
                    == legacy_dict[f"layers.{i}.f.bias"].shape
                )

                backward_mapping["backward_net.fc.linear.weight"] = f"layers.{i}.b.weight"
                backward_mapping["backward_net.fc.linear.bias"] = f"layers.{i}.b.bias"
                assert (
                    pl_dict[f"backward_net.fc.linear.weight"].shape
                    == legacy_dict[f"layers.{i}.b.weight"].shape
                )
                assert (
                    pl_dict[f"backward_net.fc.linear.bias"].shape
                    == legacy_dict[f"layers.{i}.b.bias"].shape
                )

        return forward_mapping, backward_mapping

    def _get_model_grads(self, model):
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().clone())
        return grads
