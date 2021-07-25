import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypeVar

import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from simple_parsing.helpers import list_field
from target_prop.config import Config
from target_prop.feedback_loss import get_feedback_loss
from target_prop.metrics import compute_dist_angle
from target_prop.models.model import Model
from target_prop.utils import get_list_of_values, is_trainable
from torch import Tensor, nn
from torch.nn import functional as F

logger = logging.getLogger(__file__)
T = TypeVar("T")


class SequentialModel(Model):
    """ Model that trains the forward weights and feedback weights sequentially.

    NOTE: This is basically the same as @ernoult's implementation.
    """

    @dataclass
    class HParams(Model.HParams):
        """ Hyper-Parameters of the model.

        TODO: Set these values as default (cifar10)
        ```console
        python main.py --batch-size 128 \
        --C 128 128 256 256 512 \
        --iter 20 30 35 55 20 \
        --epochs 90 \
        --lr_b 1e-4 3.5e-4 8e-3 8e-3 0.18 \
        --noise 0.4 0.4 0.2 0.2 0.08 \
        --lr_f 0.08 \
        --beta 0.7 \
        --path CIFAR-10 \
        --scheduler \
        --wdecay 1e-4 \
        ```
        """

        # Number of training steps for the feedback weights per batch. Can be a list of
        # integers, where each value represents the number of iterations for that layer.
        feedback_training_iterations: List[int] = list_field(20, 30, 35, 55, 20)

    def __init__(
        self,
        datamodule: VisionDataModule,
        hparams: "SequentialModel.HParams",
        config: Config,
    ):
        super().__init__(datamodule, hparams, config)
        # Can't do automatic optimization here, since we do multiple sequential updates
        # per batch.
        self.automatic_optimization = False
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        print("Hyper-Parameters:")
        print(self.hp.dumps_json(indent="\t"))
        # TODO: Use something like this:
        # self.supervised_metrics: List[Metrics]

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int,
    ) -> float:
        return self.shared_step(batch, batch_idx=batch_idx, phase="train")

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int,
    ) -> float:
        return self.shared_step(batch, batch_idx=batch_idx, phase="val")

    def test_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int,
    ) -> float:
        return self.shared_step(batch, batch_idx=batch_idx, phase="test")

    def shared_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        phase: str,
    ):
        """ Main step, used by the `[training/valid/test]_step` methods.
        """
        
        x, y = batch
        # Setting this value just so we don't have to pass `phase=...` to `forward_loss`
        # and `feedback_loss` below.
        self.phase = phase

        dtype: Optional[torch.dtype] = self.dtype if isinstance(
            self.dtype, torch.dtype
        ) else None
        
        # ----------- Optimize the feedback weights -------------
        feedback_loss = self.feedback_loss(x, y)
        self.log(f"{phase}/B_loss", feedback_loss, prog_bar=True)
        
        # This is never a 'live' loss, since we do the optimization steps sequentially
        # inside `feedback_loss`.
        assert not feedback_loss.requires_grad

        # ----------- Optimize the forward weights -------------
        forward_loss = self.forward_loss(x, y)
        self.log(f"{phase}/F_loss", forward_loss, prog_bar=True)

        # During training, the forward loss will be a 'live' loss tensor, since we
        # gather the losses for each layer. Here we perform only one step.
        if forward_loss.requires_grad:
            f_optimizer = self.forward_optimizer
            self.manual_backward(forward_loss)
            f_optimizer.step()
            f_optimizer.zero_grad()
            forward_loss = forward_loss.detach()
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()  # type: ignore

        return float(forward_loss + feedback_loss)

    def feedback_loss(self, x: Tensor, y: Tensor) -> Tensor:
        feedback_optimizer = self.feedback_optimizer

        n_layers = len(self.backward_net)

        # Expand these values to get one value for each feedback layer to train
        noise_scale_per_layer = self._get_noise_scale_per_feedback_layer(forward_ordering=True)
        iterations_per_layer = self._get_iterations_per_feedback_layer(forward_ordering=True)
        # Reverse the backward net, just for ease of readability:
        reversed_backward_net = self.backward_net[::-1]

        # NOTE: We never train the last layer of the feedback net (G_0).
        assert iterations_per_layer[0] == 0
        assert noise_scale_per_layer[0] == 0

        # NOTE: We can compute all the ys for all the layers up-front, because we don't
        # update the forward weights.
        # 1- Compute the forward activations (no grad).
        with torch.no_grad():
            ys: List[Tensor] = self.forward_net.forward_all(
                x, allow_grads_between_layers=False
            )

        # 2- Layer-wise autoencoder training begins:
        # NOTE: Skipping the first layer
        layer_losses: List[Optional[Tensor]] = []
        for layer_index in range(1, n_layers):
            # Forward layer
            F_i = self.forward_net[layer_index]
            # Feedback layer
            G_i = reversed_backward_net[layer_index]
            iterations = iterations_per_layer[layer_index]
            noise_scale = noise_scale_per_layer[layer_index]

            if iterations == 0:
                # Skip this layer, since it's not trainable.
                continue

            if self.phase != "train":
                # Only perform one iteration per layer when not training.
                iterations = 1

            assert noise_scale > 0
            losses_per_iteration: List[Tensor] = []
            angles = []
            distances = []
            # 3- Train the current autoencoder:
            for iteration in range(iterations):
                # Get the loss (see `feedback_loss.py`)
                layer_input = ys[layer_index-1]
                loss = get_feedback_loss(
                    G_i,
                    F_i,
                    input=layer_input,
                    noise_scale=noise_scale,
                    noise_samples=self.hp.feedback_samples_per_iteration,
                )

                # Compute the angle and distance for debugging the training of the
                # feedback weights:
                with torch.no_grad():
                    angle, distance = compute_dist_angle(F_i, G_i)
                logger.debug(
                    f"Layer {layer_index}, Iteration {iteration}, angle={angle}, distance={distance}"
                )
                angles.append(angle)
                distances.append(distance)

                if isinstance(loss, Tensor) and loss.requires_grad:
                    feedback_optimizer.zero_grad()
                    self.manual_backward(loss)
                    feedback_optimizer.step()
                    loss = loss.detach()
                    losses_per_iteration.append(loss)

                else:
                    if isinstance(loss, float):
                        loss = torch.as_tensor(loss, device=y.device)
                    losses_per_iteration.append(loss)

            iteration_losses = torch.stack(losses_per_iteration)
            total_loss_for_layer = iteration_losses.sum()
            layer_losses.append(total_loss_for_layer)
            self.log(f"{self.phase}/B_loss[{layer_index}]", total_loss_for_layer)
            # IDEA: Could log this if we add some kind of early stopping for the feedback
            # training
            # self.log(f"{self.phase}/B_iterations[{layer_index}]", iterations)
            if angles:
                self.log(f"{self.phase}/B_angles[{layer_index}]", angles)
            if distances:
                self.log(f"{self.phase}/B_distances[{layer_index}]", distances)

        return torch.stack(layer_losses).sum()

    def forward_loss(self, x: Tensor, y: Tensor) -> Tensor:
        ys: List[Tensor] = self.forward_net.forward_all(
            x, allow_grads_between_layers=False
        )
        y_n = ys[-1]

        labels = y
        logits = ys[-1]

        # Calculate the first target using the gradients of the loss w.r.t. the logits.
        # NOTE: Need to manually enable grad here so that we can also compute the first
        # target during validation / testing.
        with torch.set_grad_enabled(True):
            accuracy = self.accuracy(torch.softmax(logits, -1), labels)
            self.log(f"{self.phase}/accuracy", accuracy, prog_bar=True)

            # NOTE: detaching the logits before computing the first target, because we don't
            # want the cross-entropy loss to backpropagate itself into the last layer?
            temp_logits = logits.detach()
            temp_logits.requires_grad_(True)

            ce_loss = self.criterion(temp_logits, labels)
            batch_size = x.size(0)
            # NOTE: Need to pass a grad_outputs argument because `ce_loss` isn't a
            # scalar here.
            out_grads = ce_loss.new_ones([batch_size], requires_grad=False)
            grads = torch.autograd.grad(
                ce_loss,
                temp_logits,
                grad_outputs=out_grads,
                only_inputs=True,
                create_graph=False,
            )

        y_n_grad = grads[0]
        delta = -self.hp.beta * y_n_grad

        # NOTE: Initialize the list of targets with Nones, and we'll replace all the
        # entries with tensors corresponding to the targets of each layer.
        targets: List[Optional[Tensor]] = [None for _ in ys]

        # Compute the first target:
        t = y_n + delta
        targets[-1] = t

        n_layers = len(self.forward_net)

        # Calculate the targets for each layer, moving backward through the forward net:
        # N-1, N-2, ..., 2, 1
        # NOTE: Starting from N-1 since we already have the target for the last layer).
        for i in reversed(range(1, n_layers)):
            G = self.backward_net[-1 - i]
            with torch.no_grad():
                assert (
                    targets[i - 1] is None
                )  # Make sure we're not overwriting anything.
                # NOTE: Shifted the indices by 1 compared to @ernoult's eq.
                # t^{n-1} = s^{n-1} + G(t^{n}; B) - G(s^{n} ; B).
                targets[i - 1] = ys[i - 1] + G(targets[i]) - G(ys[i])

        # NOTE: targets[0] is the targets for the output of the first layer, not of x.
        # Calculate the losses for each layer:
        assert all(targets[i] is not None for i in range(0, n_layers))
        forward_loss_per_layer = torch.stack(
            [F.mse_loss(ys[i], targets[i]) for i in range(0, n_layers)]
        )

        forward_loss = forward_loss_per_layer.sum()
        return forward_loss

    def __unused_original(self):
        """ Keeping this here just for reference """

        # 1- Compute the output layer (y) and the reconstruction of the penultimate layer (r = G(y))
        """
        '''
        NOTE 1: the flag ind_layer specifies where the forward pass stops (default: None)
        NOTE 2: if ind_layer=n, layer n-1 is detached from the computational graph
        '''
        y, r = net(data, ind_layer = len(net.layers))

        #2- Compute the loss
        L = criterion(y.float(), target).squeeze()

        #3- Compute the first target 
        init_grads = torch.tensor([1 for i in range(y.size(0))], dtype=torch.float, device=y.device, requires_grad=True) 
        grads = torch.autograd.grad(L, y, grad_outputs=init_grads, create_graph = True)
        delta = -args.beta*grads[0]
        t = y + delta

        #4- Layer-wise feedforward training begins
        for id_layer in range(len(net.layers)):
            #5- Train current forward weights so that current layer matches its target   
            loss_f = net.layers[-1 - id_layer].weight_f_train(y, t, optimizer_f)

            #6- Compute the previous target (except if we already have reached the first hidden layer)    
            if (id_layer < len(net.layers) - 1):
                #7- Compute delta^n = G(s^{n+1} + t^{n+1}) - G(s^{n+1})
                delta = net.layers[-1 - id_layer].propagateError(r, t)
                
                #8- Compute the feedforward prediction s^n and r=G(s^n)
                y, r = net(data, ind_layer = len(net.layers) - 1 - id_layer)
        
                #9- Compute the target t^n= s^n + delta^n
                t = (y + delta).detach()
            
            if id_layer == 0:
                loss = loss_f
        """
