from torch import Tensor, nn
from typing import Union, List, Iterable, Sequence, Tuple


class TargetPropSequential(nn.Module):
    def __init__(
        self,
        forward_layers: Union[nn.Sequential, Sequence[nn.Module]],
        backward_layers: Union[nn.Sequential, Sequence[nn.Module]],
    ):
        super().__init__()
        self.forward_net = nn.Sequential(*forward_layers)
        self.backward_net = nn.Sequential(*backward_layers)

    def forward_parameters(self) -> Iterable[nn.Parameter]:
        """ Returns an iterator over the forward parameters. """
        yield from self.forward_net.parameters()

    def backward_parameters(self) -> Iterable[nn.Parameter]:
        """ Returns an iterator over the backward parameters. """
        yield from self.backward_net.parameters()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.forward_net(x)

    def forward_each(self, xs: List[Tensor]) -> List[Tensor]:
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
        assert len(xs) == len(self.forward_net)
        return [layer(x_i) for layer, x_i in zip(self.forward_net, xs)]

    def backward_each(
        self, ys: List[Tensor], forward_ordering: bool = True
    ) -> List[Tensor]:
        """Gets the outputs of every feedback/backward layer, given inputs `ys`.

        Parameters
        ----------
        ys : List[Tensor]
            A list of tensors, one per layer, which will be used as the inputs for each
            backward/feedback layer.

        forward_ordering: bool, optional
            Wether `ys` is ordered from front to back (default) or from back to front.
            The outputs are also returned in the same order as the inputs.

        Returns
        -------
        List[Tensor]
            The outputs of each backward layer, ordered from front to back when
            `forward_ordering` is True (default), or from back to front when
            `forward_ordering` is False.
        """
        assert len(ys) == len(self.backward_net)
        inputs = reversed(ys) if forward_ordering else ys
        outputs = [layer(y_i) for layer, y_i in zip(self.backward_net, inputs)]
        if forward_ordering:
            outputs = list(reversed(outputs))
        return outputs
        # Expects `y` to be ordered **from the output to the input** (i.e. same order
        # as the backward net).

    def forward_all(
        self, x: Tensor, allow_grads_between_layers: bool = False
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
        if allow_grads_between_layers:
            return forward_accumulate(self.forward_net, x)
        else:
            return layerwise_independant_forward_accumulate(self.forward_net, x)

    def backward_all(
        self, y: Tensor, allow_grads_between_layers: bool = False
    ) -> List[Tensor]:
        """Gets the outputs of all forward layers for the given inputs. 
        
        Parameters
        ----------
        y : Tensor
            
        allow_grads_between_layers : bool, optional
            Wether to allow gradients to flow from one layer to the next.
            Only used when `x` is a single `Tensor`. When `False` (default), outputs of
            each layer are detached before being passed to the next layer.

        Returns
        -------
        List[Tensor]
            The outputs of each backward layer **ordered from the output to the input**.
        """
        if isinstance(y, list):
            assert len(y) == len(self.backward_net)
            return [layer(y_i) for layer, y_i in zip(self.backward_net, y)]
        if allow_grads_between_layers:
            return forward_accumulate(self.backward_net, y)
        else:
            return layerwise_independant_forward_accumulate(self.backward_net, y)


# @torch.jit.script
def forward_accumulate(model: nn.Sequential, x: Tensor) -> List[Tensor]:
    """ IDEA: Gather all the forward activations into a list. """
    activations: List[Tensor] = []
    for layer in model:
        x = layer(x)
        activations.append(x)
    return activations


def layerwise_independant_forward_accumulate(
    model: nn.Sequential, x: Tensor
) -> List[Tensor]:
    """ IDEA: Gather all the forward activations into a list, and have layer's output be
    disconnected from that of the previous layer.
    """
    activations: List[Tensor] = []
    for layer in model:
        x = layer(x.detach())
        activations.append(x)
    return activations
