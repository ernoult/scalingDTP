from .model import BaseModel
from torch import Tensor
from typing import List, Tuple, Union
from torch import nn
import torch
from target_prop.utils import is_trainable


class ParallelModel(BaseModel):
    """ "Parallel" version of the sequential model, uses more noise samples but a single
    iteration for the training of the feedback weights, which makes it possible to use
    the automatic optimization and distributed training features of PyTorch-Lightning.
    """

    def __init__(self, datamodule, hparams, config):
        super().__init__(datamodule, hparams, config)
        # Here we can do automatic optimization, since we don't need to do multiple
        # sequential optimization steps per batch ourselves.
        self.automatic_optimization = True
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def training_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int = None
    ) -> Union[Tensor, float]:
        loss = self.shared_step(
            batch=batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx, phase="train"
        )
        if self.automatic_optimization:
            # Should have a loss with gradients if we're using automatic optimization
            # from PL.
            assert loss.requires_grad, (loss, optimizer_idx)
            return loss
        elif isinstance(loss, Tensor):
            # Need to NOT return a Tensor when not using automatic optimization.
            # BUG: Pytorch Lightning complains that we're returning a Tensor, even if
            # it's a float!
            return float(loss.item())
        return loss

    def validation_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Union[Tensor, float]:
        return self.shared_step(
            batch=batch, batch_idx=batch_idx, optimizer_idx=None, phase="val"
        )

    def test_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Union[Tensor, float]:
        return self.shared_step(batch, batch_idx=batch_idx, phase="test")

    def training_step_end(self, step_results: Union[Tensor, List[Tensor]]) -> Tensor:
        """ Called with the results of each worker / replica's output.

        See the `training_step_end` of pytorch-lightning for more info.
        """
        # TODO: For now we're kinda losing the logs and stuff that happens within the
        # workers in DP (they won't show up in the progress bar for instance).
        # merged_step_results = {
        #     k: sum(v_i.to(self.device) for v_i in v)
        #     for k, v in step_results
        # }
        merged_step_result = (
            step_results
            if isinstance(step_results, (Tensor, float))
            else sum(step_results)
        )

        # TODO: If NOT in automatic differentiation, but still in a scenario where we
        # can do a single update, do it here.
        loss = merged_step_result
        self.log(f"{self.phase}/total loss", loss, on_step=True, prog_bar=True)

        if (
            not self.automatic_optimization
            and isinstance(loss, Tensor)
            and loss.requires_grad
        ):
            forward_optimizer = self.forward_optimizer
            backward_optimizer = self.feedback_optimizer
            forward_optimizer.zero_grad()
            backward_optimizer.zero_grad()

            self.manual_backward(loss)

            forward_optimizer.step()
            backward_optimizer.step()
            return float(loss)

        elif not self.automatic_optimization:
            return float(merged_step_result)

        assert self.automatic_optimization
        return merged_step_result

    def forward_loss(self, x: Tensor, y: Tensor) -> Tensor:
        # NOTE: Could use the same exact forward loss as the sequential model, at the
        # moment.
        pass

    def feedback_loss(self, x, y):
        raise NotImplementedError("Fix this, focusing on the SequentialModel for now.")

        # Get the outputs for all layers.
        # NOTE: no need for gradients w.r.t. forward parameters.
        with torch.no_grad():
            ys = self.forward_net.forward_all(x)

        # Input of each intermediate layer
        xs = ys[:-1]
        # Inputs to each backward layer:
        # NOTE: Using ys[1:] since we don't need the `r` of the first layer (`G(x_1)`)
        yr = ys[1:]
        # Reverse the backward net, just to make the code a bit easier to read:
        reversed_backward_net = self.backward_net[::-1]
        # NOTE: This saves one forward-pass, but makes the code a tiny bit uglier:

        rs = reversed_backward_net[1:].forward_each(ys[1:])

        noise_scale_vector = self.get_noise_scale_per_layer(reversed_backward_net[1:])
        x_noise_distributions: List[Normal] = [
            Normal(loc=torch.zeros_like(x_i), scale=noise_i)
            for x_i, noise_i in zip(xs, noise_scale_vector)
        ]

        # NOTE: Notation for the tensor shapes below:
        # I: number of training iterations
        # S: number of noise samples
        # N: number of layers
        # B: batch dimension
        # X_i: shape of the inputs to layer `i`

        # List of losses, one per iteration.
        # If this is called during training, and there is more than one iteration, then
        # the losses in this list will be detached.
        dr_losses_list: List[Tensor] = []  # [I]

        n_layers = len(self.backward_net)
        if not isinstance(self.hp.feedback_training_iterations, int):
            assert len(set(self.hp.feedback_training_iterations)) == 1
            self.hp.feedback_training_iterations = self.hp.feedback_training_iterations[
                0
            ]

        iterations: int = self.hp.feedback_training_iterations
        # Only do one iteration when evaluating or testing.
        if self.phase != "train":
            iterations = 1

        noise_scale_per_layer = self._get_noise_scale_per_layer()

        for iteration in range(iterations):
            # List of losses, one per noise sample.
            dr_losses_per_sample: List[Tensor] = []  # [S]

            feedback_losses = [
                get_feedback_loss(
                    reversed_backward_net[i],
                    self.forward_net[i],
                    noise_scale=noise_scale_per_layer[i],
                    noise_samples=self.hp.feedback_,
                )
                if is_trainable(reversed_backward_net[i])
                else 0
                for i in range(n_layers)
            ]
            loss = torch.stack(feedback_losses)
            iteration_dr_loss = iteration_dr_loss.sum(1).mean()  # [1]

            # TODO: Could perhaps do some sort of 'early stopping' here if the dr
            # loss is sufficiently small?
            if self.phase != "train":
                assert not iteration_dr_loss.requires_grad
            if not self.automatic_optimization and iteration_dr_loss.requires_grad:
                optimizer = self.optimizers()[self._feedback_optim_index]
                optimizer.zero_grad()
                self.manual_backward(iteration_dr_loss)
                optimizer.step()
                iteration_dr_loss = iteration_dr_loss.detach()
            self.log(
                f"{self.phase}/dr_loss_{iteration}",
                iteration_dr_loss,
                prog_bar=False,
                logger=True,
            )

            dr_losses_list.append(iteration_dr_loss)

        dr_losses = torch.stack(dr_losses_list)
        # TODO: Take the average, or the sum here?
        dr_loss: Tensor = dr_losses.sum()
        return dr_loss

    def _get_iterations_per_layer(self) -> List[int]:
        """ Returns the number of iterations to perform for each of the feedback layers.

        NOTE: Returns it in the same order as the backward_net, i.e. [GN ... G0]
        """
        return [1 if is_trainable(layer) else 0 for layer in self.backward_net]
