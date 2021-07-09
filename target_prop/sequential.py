import torch
from torch import Tensor, nn
from typing import Union, List, Iterable, Sequence, Tuple


class Sequential(nn.Sequential, Sequence[nn.Module]):
    def forward_each(
        self, xs: List[Tensor]
    ) -> List[Tensor]:
        """Gets the outputs of every layer, given inputs for each layer `xs`.

        Parameters
        ----------
        x : List[Tensor]
            A list of tensors, one per layer, which will be used as the inputs for each
            forward layer.

        Returns
        -------
        List[Tensor]
            The outputs of each forward layer.
        """
        xs = list(xs) if not isinstance(xs, (list, tuple)) else xs
        assert len(xs) == len(self), (len(xs), len(self))
        return [
            layer(x_i)
            for layer, x_i in zip(self, xs)
        ]

    def forward_all(
        self,
        x: Tensor,
        allow_grads_between_layers: bool = False,
    ) -> List[Tensor]:
        """Gets the outputs of all forward layers for the given input. 
        
        Parameters
        ----------
        x : Tensor
            Input tensor.

        allow_grads_between_layers : bool, optional
            Wether to allow gradients to flow from one layer to the next.
            When `False` (default), outputs of each layer are detached before being
            fed to the next layer.

        Returns
        -------
        List[Tensor]
            The outputs of each forward layer.
        """
        return forward_accumulate(
            self,
            x,
            allow_grads_between_layers=allow_grads_between_layers,
        )

# @torch.jit.script
def forward_accumulate(
    model: nn.Sequential,
    x: Tensor,
    allow_grads_between_layers: bool = False,
) -> List[Tensor]:
    """ IDEA: Gather all the forward activations into a list. """
    activations: List[Tensor] = []
    for layer in model:
        x = layer(x if allow_grads_between_layers else x.detach())
        activations.append(x)
    return activations

