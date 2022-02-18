from dataclasses import dataclass
import dataclasses
import logging
from collections import OrderedDict
from typing import ClassVar, List, Optional, Type

import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from target_prop._weight_operations import init_symetric_weights
from target_prop.backward_layers import mark_as_invertible
from target_prop.config import Config
from target_prop.datasets.dataset_config import DatasetConfig
from target_prop.layers import Reshape, forward_all, invert
from target_prop.metrics import compute_dist_angle
from target_prop.models import DTP
from target_prop.networks import Network
from target_prop.networks.lenet import LeNet
from target_prop.networks.resnet import ResNet18, ResNet34
from target_prop.networks.simple_vgg import SimpleVGG
from target_prop.utils.utils import named_trainable_parameters
from torch import Tensor, nn
from torch.nn import functional as F

networks = [SimpleVGG, ResNet18, ResNet34, LeNet]


class TestDTP:
    # The type of model to test.
    model_class: ClassVar[Type[DTP]] = DTP

    @pytest.fixture(params=networks, name="network_type")
    @classmethod
    def network_type(cls, request):
        """ Fixture that yields each type of network. """
        net_type = request.param
        yield net_type

    @pytest.fixture(name="debug_hparams")
    @classmethod
    def debug_hparams(cls, network_type: Type[Network]):
        """Hyper-parameters to use for debugging, depending on the network type."""
        default_hps = cls.model_class.HParams()
        # Reduce the number of iterations, so that tests run much quicker.
        shorter_feedback_training_iterations = [2 for _ in default_hps.feedback_training_iterations]
        return dataclasses.replace(
            default_hps,
            feedback_training_iterations=shorter_feedback_training_iterations,
            max_epochs=1,
        )

    @pytest.mark.parametrize("dataset", ["cifar10"])
    def test_fast_dev_run(
        self, dataset: str, network_type: Type[Network], debug_hparams: Network.HParams
    ):
        """ Run a fast dev run using a single batch of data for training/validation/testing. """
        # NOTE: Not testing using other datasets for now, because the defaults on the HParams are
        # all made for Cifar10 (e.g. hard-set to 5 layers). This would make the test uglier, as we'd
        # have to pass different values for each dataset.
        dataset_config = DatasetConfig(dataset=dataset, num_workers=0)
        hparams = debug_hparams
        trainer = Trainer(
            max_epochs=1,
            gpus=torch.cuda.device_count(),
            accelerator=None,
            fast_dev_run=True,
            # accelerator="ddp",  # todo: debug DP/DDP
            logger=None,
            checkpoint_callback=False,
        )
        config = Config(debug=True)
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
        datamodule = dataset_config.make_datamodule(batch_size=hparams.batch_size)

        # Create the network
        network: nn.Sequential = network_type(
            in_channels=datamodule.dims[0], n_classes=datamodule.num_classes, hparams=None,
        )
        assert isinstance(debug_hparams, self.model_class.HParams)
        model = self.model_class(
            datamodule=datamodule,
            hparams=hparams,  # type: ignore
            config=config,
            network=network,
            network_hparams=network.hparams,
        )
        trainer.fit(model, datamodule=datamodule)

        test_results = trainer.test(model, datamodule=datamodule)
        print(test_results)
        test_accuracy: float = test_results[0]["test/accuracy"]
        assert test_accuracy > 0

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("dataset", ["cifar10"])
    @pytest.mark.parametrize("seed", [123, 456])
    def test_run_is_reproducible_given_seed(
        self, dataset: str, network_type: Type[Network], debug_hparams: Network.HParams, seed: int
    ):
        """ Checks that `run` produces the same result when passed the same args and seed. """
        perf_1 = self._get_debug_performance_given_seed(
            dataset=dataset,
            network_type=network_type,
            network_hparams=network_type.HParams(),
            hparams=debug_hparams,
            seed=seed,
        )
        perf_2 = self._get_debug_performance_given_seed(
            dataset=dataset,
            network_type=network_type,
            network_hparams=network_type.HParams(),
            hparams=debug_hparams,
            seed=seed,
        )
        assert perf_1 == perf_2

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("dataset", ["cifar10"])
    @pytest.mark.parametrize("seed", [123, 456])
    @pytest.mark.parametrize("network_type", networks)
    def test_seed_has_impact(
        self, dataset: str, network_type: Type[Network], debug_hparams: Network.HParams, seed: int
    ):
        """ Tests that when `run` is passed different seeds, the results are different.
        """
        perf_1 = self._get_debug_performance_given_seed(
            dataset=dataset,
            network_type=network_type,
            network_hparams=network_type.HParams(),
            hparams=debug_hparams,
            seed=seed,
        )
        perf_2 = self._get_debug_performance_given_seed(
            dataset=dataset,
            network_type=network_type,
            network_hparams=network_type.HParams(),
            hparams=debug_hparams,
            seed=seed + 456,
        )
        assert perf_1 != perf_2

    def _get_debug_performance_given_seed(
        self,
        dataset: str,
        network_type: Type[Network],
        network_hparams: Network.HParams,
        hparams: DTP.HParams,
        seed: int,
        batches: int = 1,
    ) -> float:
        print(f"Selected seed: {seed}")
        # Note: using 0 workers to speed up testing.
        dataset_config = DatasetConfig(dataset=dataset, num_workers=0)

        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("target_prop").setLevel(logging.DEBUG)

        from main import run, Options

        options = Options(
            dataset=dataset_config,
            model=hparams,
            network=network_hparams,
            trainer=Trainer(
                max_epochs=1,
                gpus=torch.cuda.device_count(),
                fast_dev_run=True,
                logger=None,
                limit_train_batches=batches,
                limit_val_batches=batches,
                limit_test_batches=batches,
            ),
            debug=True,
            seed=seed,
        )
        test_accuracy = run(options=options)

        # print(f"Test accuracy: {test_accuracy:.1%}")
        return test_accuracy


def get_forward_weight_losses(
    forward_net: nn.Sequential, feedback_net: nn.Sequential, x: Tensor, y: Tensor, beta: float,
) -> List[Tensor]:
    # NOTE: Sanity check: Use standard backpropagation for training rather than TP.
    ## --------
    # return super().forward_loss(x=x, y=y)
    ## --------

    ys: List[Tensor] = forward_all(
        forward_net, x, allow_grads_between_layers=False,
    )
    logits = ys[-1]
    labels = y

    # Calculate the first target using the gradients of the loss w.r.t. the logits.
    # NOTE: Need to manually enable grad here so that we can also compute the first
    # target during validation / testing.
    with torch.set_grad_enabled(True):
        ce_loss = F.cross_entropy(logits, labels, reduction="sum")
        grads = torch.autograd.grad(
            ce_loss,
            logits,
            only_inputs=True,  # Do not backpropagate further than the input tensor!
            create_graph=False,
        )
        assert len(grads) == 1

    y_n_grad = grads[0]

    delta = -beta * y_n_grad

    print(f"Delta norm: {delta.norm().item()}")

    # NOTE: Initialize the list of targets with Nones, and we'll replace all the
    # entries with tensors corresponding to the targets of each layer.
    targets: List[Optional[Tensor]] = [None for _ in ys]

    # Compute the first target (for the last layer of the forward network):
    t = logits.detach() + delta
    targets[-1] = t

    N = len(forward_net)

    # Reverse the ordering of the layers, just to make the indexing in the code below match those of
    # the math equations.
    reordered_feedback_net = feedback_net[::-1]  # type: ignore

    # Calculate the targets for each layer, moving backward through the forward net:
    # N-1, N-2, ..., 2, 1
    # NOTE: Starting from N-1 since we already have the target for the last layer).
    with torch.no_grad():
        for i in reversed(range(1, N)):

            G = reordered_feedback_net[i]
            # G = feedback_net[-1 - i]

            assert targets[i - 1] is None  # Make sure we're not overwriting anything.
            # NOTE: Shifted the indices by 1 compared to @ernoult's eq.
            # t^{n-1} = s^{n-1} + G(t^{n}; B) - G(s^{n} ; B).
            targets[i - 1] = ys[i - 1] + G(targets[i]) - G(ys[i])

            # NOTE: Alternatively, just target propagation:
            # targets[i - 1] = G(targets[i])

    # NOTE: targets[0] is the targets for the output of the first layer, not for x.
    # Make sure that all targets have been computed and that they are fixed (don't require grad)
    assert all(target is not None and not target.requires_grad for target in targets)

    # Calculate the losses for each layer:
    forward_loss_per_layer = [
        0.5 * ((ys[i] - targets[i]) ** 2).view(ys[i].size(0), -1).sum(1).sum()
        # NOTE: Apparently NOT equivalent to the following.
        # 0.5 * F.mse_loss(ys[i], targets[i], reduction="sum")  # type: ignore
        for i in range(0, N)
    ]
    assert len(ys) == len(targets) == len(forward_loss_per_layer) == len(forward_net)

    for i, layer_loss in enumerate(forward_loss_per_layer):
        print(f"F_loss[{i}]", layer_loss)

    # NOTE: Returning the loss for each layer here rather than a single loss tensor, so we can debug
    # things a bit easier.
    # loss_tensor = torch.stack(forward_loss_per_layer, -1)
    # return loss_tensor.sum()
    return forward_loss_per_layer


# def beta():
#     return 0.01


@pytest.fixture()
def channels():
    return [1, 16, 32]


@pytest.fixture
def forward_net(channels: List[int]) -> nn.Sequential:
    n_classes = 10
    torch.random.manual_seed(123)

    example_input_array = torch.rand([32, channels[0], 32, 32])
    example_labels = torch.randint(0, n_classes, [32])

    forward_net = nn.Sequential(
        OrderedDict(
            [
                *(
                    (
                        f"conv_{i}",
                        # NOTE: Using a simpler network architecture for the sake of testing.
                        nn.Conv2d(
                            channels[i],
                            channels[i + 1],
                            kernel_size=3,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                        # Sequential(
                        #     OrderedDict(
                        #         conv=nn.Conv2d(
                        #             channels[i],
                        #             channels[i + 1],
                        #             kernel_size=3,
                        #             stride=1,
                        #             padding=0,
                        #         ),
                        #         rho=nn.ELU(),
                        #         pool=MaxPool2d(kernel_size=2, stride=2, return_indices=False),
                        #         # pool=nn.AvgPool2d(kernel_size=2),
                        #     )
                        # ),
                    )
                    for i in range(0, len(channels) - 1)
                ),
                ("reshape", Reshape(target_shape=(-1,))),
                # NOTE: Using LazyLinear so we don't have to know the hidden size in advance
                ("linear", nn.LazyLinear(out_features=n_classes, bias=False)),
            ]
        )
    )
    forward_net = mark_as_invertible(forward_net)
    # Pass an example input through the forward net so that all the layers which
    # need to know their inputs/output shapes get a chance to know them.
    # This is necessary for creating the backward network, as some layers
    # (e.g. Reshape) need to know what the input shape is.
    _ = forward_net(example_input_array)
    assert forward_net.input_shape
    assert forward_net.output_shape
    return forward_net


@pytest.fixture
def feedback_net(forward_net: nn.Sequential) -> nn.Sequential:
    print(f"Forward net: ")
    print(forward_net)
    torch.random.manual_seed(123)
    example_input_array = torch.rand([32, 1, 32, 32])
    example_out: Tensor = forward_net(example_input_array)
    assert example_out.requires_grad
    # Get the "pseudo-inverse" of the forward network:
    # TODO: Initializing the weights of the backward net with the transpose of the weights of
    # the forward net, and will check if the gradients are similar.
    feedback_net = invert(forward_net)
    init_symetric_weights(forward_net, feedback_net)
    print(f"Feedback net: ")
    print(feedback_net)
    return feedback_net


def test_weights_are_initialized_symmetrically(forward_net: nn.Sequential):
    feedback_net = invert(forward_net)
    init_symetric_weights(forward_net, feedback_net)
    print(f"Feedback net: ")
    print(feedback_net)

    trainable_forward_weights = dict(named_trainable_parameters(forward_net))
    trainable_feedback_weights = dict(named_trainable_parameters(feedback_net))

    # Make sure that the weights are initialized symmetrically.
    for name, forward_param in trainable_forward_weights.items():
        assert name in trainable_forward_weights
        feedback_param = trainable_feedback_weights[name]
        if forward_param.dim() == 4:
            assert forward_param.shape == feedback_param.shape
            assert (forward_param == feedback_param).all()
        if forward_param.dim() == 2:
            assert forward_param.shape == feedback_param.t().shape
            assert (forward_param == feedback_param.t()).all()


@pytest.mark.parametrize("beta", [1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.7])
def test_losses_of_each_layer_are_independent(
    forward_net: nn.Sequential, feedback_net: nn.Sequential, beta: float
):
    # Asserts that losses only affect their own layer and not the layers before.
    n_classes = 10

    torch.random.manual_seed(123)
    example_input_array = torch.rand([32, 1, 32, 32])
    example_labels = torch.randint(0, n_classes, [32])

    dtp_losses: List[Tensor] = get_forward_weight_losses(
        forward_net=forward_net,
        feedback_net=feedback_net,
        x=example_input_array,
        y=example_labels,
        beta=beta,
    )

    dtp_grads_v0 = {}
    for i, ((layer_name, layer), layer_loss) in reversed(
        list(enumerate(zip(forward_net.named_children(), dtp_losses)))
    ):
        # Make sure that this layer has never received a gradient before.
        assert all(p.grad is None or (p.grad == 0).all() for p in layer.parameters())

        print(f"Backpropagating the loss for layer {i}.")
        if any(p.requires_grad for p in layer.parameters()):
            # there should be a 'live', non-zero loss for the trainable layers.
            assert layer_loss.requires_grad
            assert (layer_loss != 0.0).any()
        else:
            # Some layers (e.g. Reshape) have a 'target' but their loss doesn't require any gradients.
            assert not layer_loss.requires_grad

        #  NOTE: This also checks that the losses for each layer are independant, since we
        # backpropagate the losses starting from the end of the forward net and moving backward
        # without calling `zero_grad()`.
        if layer_loss.requires_grad:
            layer_loss.backward()
            dtp_grads_v0.update(
                {
                    f"{layer_name}.{name}": p.grad.clone().detach()
                    for name, p in named_trainable_parameters(layer)
                }
            )

    forward_net.zero_grad()
    dtp_losses = get_forward_weight_losses(
        forward_net=forward_net,
        feedback_net=feedback_net,
        x=example_input_array,
        y=example_labels,
        beta=beta,
    )
    total_loss = sum(dtp_losses, start=torch.zeros(1))
    total_loss.backward()
    dtp_grads_v1 = {
        name: p.grad.clone().detach() for name, p in named_trainable_parameters(forward_net)
    }

    for parameter_name, grad_v0 in dtp_grads_v0.items():
        grad_v1 = dtp_grads_v1[parameter_name]
        assert (grad_v0 == grad_v1).all()
    # If we get here, then gradients are identical wether we call .backward() on the total loss, or
    # if we backpropagate the losses for each layer moving backward (as you'd expect).


@pytest.mark.parametrize("beta", [0.005])
def test_grads_are_similar(forward_net: nn.Sequential, beta: float):
    n_classes = 10

    torch.random.manual_seed(123)
    example_input_array = torch.rand([32, 1, 32, 32])
    example_labels = torch.randint(0, n_classes, [32])
    # Pass an example input through the forward net so that all the layers which
    # need to know their inputs/output shapes get a chance to know them.
    # This is necessary for creating the backward network, as some layers
    # (e.g. Reshape) need to know what the input shape is.
    example_out: Tensor = forward_net(example_input_array)
    assert example_out.requires_grad

    print(f"Forward net: ")
    print(forward_net)

    # Get the "pseudo-inverse" of the forward network:
    # Initializing the weights of the feedback net with the transpose of the weights of
    # the forward net.
    feedback_net = invert(forward_net)
    init_symetric_weights(forward_net, feedback_net)

    print(f"Feedback net: ")
    print(feedback_net)

    # Get the normal backprop loss and gradients:
    forward_net.zero_grad()
    logits = forward_all(forward_net, example_input_array, allow_grads_between_layers=True)[-1]
    true_backprop_loss = F.cross_entropy(logits, example_labels)
    true_backprop_loss.backward()
    true_backprop_grads = {
        name: p.grad.clone().detach() for name, p in named_trainable_parameters(forward_net)
    }

    # Calculate the gradients obtained with DTP (the function above).
    forward_net.zero_grad()  # Reset the grads first.
    # Get the loss for each layer:
    dtp_losses: List[Tensor] = get_forward_weight_losses(
        forward_net=forward_net,
        feedback_net=feedback_net,
        x=example_input_array,
        y=example_labels,
        beta=beta,
    )
    # NOTE: The losses for each layer are independant. Backpropagating the sum of the losses is the
    # same as backpropagating each loss individually.
    total_loss = sum(dtp_losses, start=torch.zeros(1))
    total_loss.backward()
    dtp_grads = {
        name: p.grad.clone().detach() for name, p in named_trainable_parameters(forward_net)
    }
    dtp_grads = {key: (1 / beta) * grad for key, grad in dtp_grads.items()}

    print(f"Beta: {beta}")
    for name, backprop_grad in true_backprop_grads.items():
        if name.endswith("bias"):
            continue
        print(f"Gradients for parameter: {name} ")
        dtp_grad = dtp_grads[name]

        l1_norm = backprop_grad.norm(p=1).item()
        l2_norm = backprop_grad.norm(p=2).item()
        print(f"\tGradient norm (backprop):\t {l1_norm=}, {l2_norm=}")

        l1_norm = dtp_grad.norm(p=1).item()
        l2_norm = dtp_grad.norm(p=2).item()
        print(f"\tGradient norm (diff-tp) :\t {l1_norm=}, {l2_norm=}")

        if backprop_grad.dim() > 1:
            distance, angle = compute_dist_angle(backprop_grad, dtp_grad)
            print(f"\tDistance between grads: {distance:.3f}, angle between grads: {angle:.3f}")
        # l1_distance_between_grads = (backprop_grad - dtp_grad).abs().sum().item()
        # l2_distance_between_grads = F.mse_loss(backprop_grad, dtp_grad).item()
        # print(f"\tDistance between grads: {l1_distance_between_grads=}, {l2_distance_between_grads=}")

    # assert False, "TODO: Check if the distances/angles of the gradients printed above make sense."
