from torch import Tensor
import torch
from typing import List, Optional
from torch import nn

from target_prop.feedback_loss import feedback_loss
from target_prop.models.model import Model
from dataclasses import dataclass
from simple_parsing.helpers import list_field
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from target_prop.config import Config
from target_prop.utils import get_list_of_values, is_trainable
from torch.nn import functional as F


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
        # Adjust the number of iterations so it "lines up" with the backward_net.
        self.feedback_iterations_per_layer: List[int] = self._get_iterations_per_layer()

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.target_loss_fn = nn.MSELoss()

        # TODO: Use something like this:
        # self.supervised_metrics: List[Metrics]

    def feedback_loss(self, x: Tensor, y: Tensor) -> Tensor:
        feedback_optimizer = self.feedback_optimizer

        # 1- Compute the first hidden layer (no grad).
        with torch.no_grad():
            y = self.forward_net[0](x)

        n_layers = len(self.backward_net)

        # Expand the 'noise' hparam to a list of values for each *trained* layer.
        noise_scale_per_layer = self._get_noise_scale_per_layer()
        iterations_per_layer = self._get_iterations_per_layer()

        # Reverse the backward net, just for ease of readability:
        reversed_backward_net = self.backward_net[::-1]

        layer_losses: List[Tensor] = []

        ys: List[Tensor] = self.forward_net.forward_all(
            x, allow_grads_between_layers=False
        )

        # 2- Layer-wise autoencoder training begins:
        for layer_index in range(1, n_layers):
            # Forward layer
            F_i = self.forward_net[layer_index]
            # Feedback layer
            G_i = reversed_backward_net[layer_index]
            iterations = iterations_per_layer[layer_index]
            noise_scale = noise_scale_per_layer[layer_index]

            s = ys[layer_index]

            t_n = y
            # t^n = s^n + G(t^{n+1}; B) - G(s^{n+1} ; B)

            if iterations == 0:
                # Skip this layer, since it's not trainable, but still compute the `y`
                # for the next layer.
                with torch.no_grad():
                    y = F_i(y)
                continue

            losses_per_iteration: List[Tensor] = []
            for iteration in range(iterations):
                # 3- Train the current autoencoder:
                loss = feedback_loss(
                    G_i,
                    F_i,
                    input=y,
                    noise_scale=noise_scale,
                    noise_samples=self.hp.feedback_samples_per_iteration,
                )
                if isinstance(loss, Tensor) and loss.requires_grad:
                    feedback_optimizer.zero_grad()
                    self.manual_backward(loss)
                    feedback_optimizer.step()
                    losses_per_iteration.append(loss.detach())
                else:
                    if isinstance(loss, float):
                        loss = torch.as_tensor(loss, device=y.device)
                    losses_per_iteration.append(loss)

            iteration_losses = torch.stack(losses_per_iteration)
            total_loss_for_layer = iteration_losses.sum()
            layer_losses.append(total_loss_for_layer)
            self.log(
                f"{self.phase}/Feedback Loss [{layer_index}]", total_loss_for_layer
            )
            self.log(
                f"{self.phase}/Feedback Loss iterations [{layer_index}]", iterations
            )
            # logger.debug(
            #     f"layer {layer_index}: losses per iteration: {losses_per_iteration}"
            # )
            # 4- Compute the next hidden layer
            with torch.no_grad():
                y = F_i(y)

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
            self.log(f"{self.phase}/accuracy", accuracy, prog_bar=True, logger=True)

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
                ce_loss, temp_logits, grad_outputs=out_grads, only_inputs=True, create_graph=False
            )

        y_n_grad = grads[0]
        delta = - self.hp.beta * y_n_grad

        # NOTE: Initialize the list of targets with zeros, and we'll replace all the
        # entries with tensors corresponding to the targets of each layer.

        targets: List[Optional[Tensor]] = [None for _ in ys]
        t = y_n + delta
        targets[-1] = t

        n_layers = len(self.forward_net)
        
        # Calculate the targets for each layer.
        for i in reversed(range(2, n_layers)):
            # Move backward through the forward net: N, N-1, ..., 2, 1
            G = self.backward_net[-1 - i]

            # NOTE: Shifted the indices by 1 compared to @ernoult's eq.
            # t^{n-1} = s^{n-1} + G(t^{n}; B) - G(s^{n} ; B).
            with torch.no_grad():
                assert targets[i-1] is None
                targets[i-1] = ys[i-1] + G(targets[i]) - G(ys[i])

        assert targets[0] is None

        # Calculate the losses for each layer:
        forward_loss_per_layer = torch.stack([
            self.target_loss_fn(ys[i], targets[i]) for i in range(1, n_layers) 
        ])
        
        forward_loss = forward_loss_per_layer.sum()
        
        if forward_loss.requires_grad and not self.automatic_optimization:
            optimizer = self.forward_optimizer
            optimizer.zero_grad()
            # forward_loss.backward()
            self.manual_backward(forward_loss)
            optimizer.step()
            forward_loss = float(forward_loss.detach().item())

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

    def _get_iterations_per_layer(self) -> List[int]:
        """ Returns the number of iterations to perform for each of the feedback layers.

        NOTE: Returns it in the same order as the backward_net, i.e. [GN ... G0]
        """
        # TODO: Make the number of iterations align directly with the trainable layers.
        # This is required because there may be some layers (e.g. Reshape), which are
        # present in the architecture of the backward network, but aren't trainable.
        n_trainable_layers = sum(map(is_trainable, self.backward_net))

        if is_trainable(self.backward_net[-1]):
            # Don't count the last layer of the backward net (i.e. G_0), since we don't
            # train it.
            n_trainable_layers -= 1

        trainable_layer_iterations = get_list_of_values(
            self.hp.feedback_training_iterations, out_length=n_trainable_layers
        ).copy()

        offset = 0
        iterations_per_layer: List[int] = []
        for layer in self.backward_net:
            # Iterating from GN ... G0,
            if is_trainable(layer) and trainable_layer_iterations:
                iterations_per_layer.append(trainable_layer_iterations.pop(0))
            else:
                iterations_per_layer.append(0)

        assert iterations_per_layer[-1] == 0
        return iterations_per_layer
