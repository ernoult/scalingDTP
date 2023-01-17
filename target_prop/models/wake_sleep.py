"""
An adapter for the WakeSleep learning algorithm in the proteus codebase.

To debug:
```
python main.py model=wake_sleep network=wake_sleep_mnist dataset=mnist trainer=overfit_one_batch callbacks=no_checkpoints
```

"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.optim
from proteus.learning_algorithm import WakeSleep
from proteus.network import WSLayeredModel
from torch import Tensor
from torch.optim import Optimizer

from target_prop.config import MiscConfig
from target_prop.networks import Network

from .model import Model, PhaseStr, StepOutputDict, VisionDataModule


class WakeSleepAdaptedNetwork(WSLayeredModel, Network):
    @dataclass
    class HParams(Network.HParams):
        l1_N: int = 784
        l2_N: int = 200
        l3_N: int = 10
        sigma_inf: float = 0.01
        sigma_gen: float = 0.01
        batch_size: int = 64

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: WakeSleepAdaptedNetwork.HParams | None = None,
    ):
        hparams = hparams or self.HParams()
        WSLayeredModel.__init__(self, params=dataclasses.asdict(hparams))
        Network.__init__(self, in_channels=in_channels, n_classes=n_classes, hparams=hparams)
        self.hparams: WakeSleepAdaptedNetwork.HParams = hparams

    def __iter__(self):
        yield from self.graph

    def __len__(self):
        return len(self.graph)

    def forward(self, x: Tensor | Sequence[Tensor]) -> Tensor:
        if isinstance(x, Tensor):
            x = (x,)
        output = super().forward(x)
        if output is None:
            output = self.l3.output
        assert output is not None
        return output


class WakeSleepModel(Model[WakeSleepAdaptedNetwork]):
    @dataclass
    class HParams(Model.HParams):
        batch_size: int = 64
        """ batch size """

        n_epochs: int = 3
        batch_size_train: int = 64
        batch_size_test: int = 1000
        learning_rate: float = 0.001 * (0.01**2)
        momentum: float = 0.5

    def __init__(
        self,
        datamodule: VisionDataModule,
        network: WakeSleepAdaptedNetwork,
        hparams: WakeSleepModel.HParams | None = None,
        config: MiscConfig | None = None,
    ):
        hparams = hparams or self.HParams()
        self.example_input_array = (
            torch.randn(hparams.batch_size, 1, 28, 28),
            torch.randint(10, (hparams.batch_size,)),
        )
        super().__init__(datamodule, network, hparams, config)
        self.automatic_optimization = False
        self.hp: WakeSleepModel.HParams
        self.learn_alg: WakeSleep | None = None

    def on_fit_start(self) -> None:
        optimizers = self.optimizers()
        assert isinstance(optimizers, list)
        inference_optim = optimizers[0]
        generation_optim = optimizers[1]
        assert isinstance(inference_optim, Optimizer)
        assert isinstance(generation_optim, Optimizer)
        self.learn_alg = WakeSleep(
            self.network,
            (inference_optim, generation_optim),
            params={"n_epochs": 3, "gen_batch_num": 1},
        )

    def configure_optimizers(self) -> tuple[Optimizer, Optimizer]:
        optim_inf = torch.optim.SGD(
            self.network.inf_group.parameters(), lr=self.hp.learning_rate, momentum=self.hp.momentum
        )
        optim_gen = torch.optim.SGD(
            self.network.gen_group.parameters(),
            lr=self.hp.learning_rate * 10,
            momentum=self.hp.momentum,
        )
        return optim_inf, optim_gen

    def shared_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, phase: PhaseStr
    ) -> StepOutputDict:
        data, target = batch
        assert self.learn_alg
        self.network.forward((data, target))

        if phase == "train":
            self.learn_alg.update_params()

        logits = self.network.l3.output
        loss = self.network.loss
        assert isinstance(loss, Tensor)
        assert loss.requires_grad
        assert logits is not None
        assert loss is not None

        # output = torch.argmax(torch.round(self.network.l3.output), axis=1)
        # cat_loss = torch.mean((output == target).float())
        # loss = self.network.loss.detach()
        # output = self.network.l3.output
        # gen_output = self.network.l3.gen_output

        return StepOutputDict(logits=logits, y=target, loss=loss.detach(), log={})
