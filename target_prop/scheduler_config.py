from abc import abstractmethod
from typing import Any
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, _LRScheduler
from dataclasses import dataclass
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from simple_parsing.helpers.hparams import uniform, log_uniform
from torch.optim.optimizer import Optimizer


@dataclass
class LRSchedulerConfig(HyperParameters):
    interval: str = "epoch"
    frequency: int = 1

    @abstractmethod
    def make_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        pass


@dataclass
class StepLRConfig(LRSchedulerConfig):
    step_size: int = uniform(10, 50, default=30)
    gamma: float = log_uniform(0.01, 0.5, default=0.1)

    def make_scheduler(self, optimizer: Optimizer) -> StepLR:
        return StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)


@dataclass
class CosineAnnealingLRConfig(LRSchedulerConfig):
    T_max: int = uniform(80, 90, default=85, discrete=True)
    eta_min: float = log_uniform(1e-6, 1e-3, default=1e-5)

    def make_scheduler(self, optimizer: Optimizer) -> CosineAnnealingLR:
        return CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.eta_min)

