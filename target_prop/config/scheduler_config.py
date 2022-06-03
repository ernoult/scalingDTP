from abc import abstractmethod
from dataclasses import dataclass

from simple_parsing.helpers.serialization.serializable import Serializable
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, _LRScheduler
from torch.optim.optimizer import Optimizer


@dataclass
class LRSchedulerConfig(Serializable):

    # Keys used when creating the scheduler in PyTorch-Lightning's `configure_optimizers` method.
    interval: str = "epoch"
    frequency: int = 1

    @abstractmethod
    def make_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        pass


@dataclass
class StepLRConfig(LRSchedulerConfig):
    step_size: int = 30
    gamma: float = 0.1

    def make_scheduler(self, optimizer: Optimizer) -> StepLR:
        return StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)


@dataclass
class CosineAnnealingLRConfig(LRSchedulerConfig):
    T_max: int = 85
    eta_min: float = 1e-5

    def make_scheduler(self, optimizer: Optimizer) -> CosineAnnealingLR:
        return CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.eta_min)
