from typing import ClassVar, Type

from target_prop.networks.network import Network
from target_prop.networks.resnet import ResNet

from .parallel_dtp import ParallelDTP
from .dtp_test import TestDTP as DTPTests
from .dtp import DTP
import pytest


@pytest.mark.xfail(reason="Bug #28", raises=ValueError)
class TestParallelModel(DTPTests):
    model_class: ClassVar[Type[ParallelDTP]] = ParallelDTP

    @pytest.fixture(name="debug_hparams")
    @classmethod
    def debug_hparams(cls, network_type: Type[Network]):
        """Hyper-parameters to use for debugging, depending on the network type."""
        # For ParallelDTP, it might use feedback_samples_per_iteration > 1, which multiplies the
        # effective batch size (and memory consumption), hence we reduce the value here.
        if issubclass(network_type, ResNet):
            # Use a smaller number of samples, since we can easily run out of memory.
            return cls.model_class.HParams(feedback_samples_per_iteration=1, max_epochs=1)
        return cls.model_class.HParams(feedback_samples_per_iteration=2, max_epochs=1)
