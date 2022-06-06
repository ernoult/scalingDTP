from typing import ClassVar, Type

import pytest

from target_prop.networks.network import Network
from target_prop.networks.resnet import ResNet, ResNet18
from target_prop.networks.simple_vgg import SimpleVGG

from .dtp import DTP
from .dtp_test import TestDTP as DTPTests
from .parallel_dtp import ParallelDTP


@pytest.fixture(autouse=True)
def xfail_tests_affected_by_bug28(request: pytest.FixtureRequest):
    """Xfails the tests where the distance and angle metrics aren't handled properly."""
    if request.function.__name__ in ["test_fast_dev_run", "test_run_is_reproducible_given_seed"]:
        network_type = request.getfixturevalue("network_type")
        if network_type not in (SimpleVGG, ResNet18):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=ValueError,
                    reason="Bug #28: https://github.com/ernoult/scalingDTP/issues/28",
                    # strict=True,
                )
            )


# @pytest.mark.xfail(reason="Bug #28", raises=ValueError)
class TestParallelDTP(DTPTests):
    model_class: ClassVar[Type[ParallelDTP]] = ParallelDTP

    @pytest.fixture(name="debug_hparams")
    @classmethod
    def debug_hparams(
        cls, network_type: Type[Network], request: pytest.FixtureRequest
    ) -> DTP.HParams:
        """Hyper-parameters to use for debugging, depending on the network type."""
        # For ParallelDTP, it might use feedback_samples_per_iteration > 1, which multiplies the
        # effective batch size (and memory consumption), hence we reduce the value here.

        if issubclass(network_type, ResNet):
            # Use a smaller number of samples, since we can easily run out of memory.
            return cls.model_class.HParams(feedback_samples_per_iteration=1, max_epochs=1)
        return cls.model_class.HParams(feedback_samples_per_iteration=2, max_epochs=1)
