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

    def forward_each(
        self, xs: List[Tensor], start_layer_index: int = 0, end_layer_offset: int = 0,
    ) -> List[Tensor]:
        """Gets the outputs of every layer, given inputs for each layer `xs`.

        Parameters
        ----------
        x : List[Tensor]
            A list of tensors, one per layer, which will be used as the inputs for each
            forward layer.

        start_layer_index: int, optional
            Index of the first backward layer to use.

        end_layer_offset: int, optional
            offset from the end, indicates the last layer to use.
            By default uses all layers until the end.

        Returns
        -------
        List[Tensor]
            The outputs of each forward layer.
        """
        assert (
            len(xs) == len(self.forward_net) - start_layer_index - end_layer_offset
        ), (
            len(xs),
            len(self.forward_net),
            start_layer_index,
        )
        end_layer_index = len(self.forward_net) - end_layer_offset

        return [
            layer(x_i)
            for layer, x_i in zip(
                self.forward_net[start_layer_index:end_layer_index], xs
            )
        ]

    def backward_each(
        self,
        ys: List[Tensor],
        forward_ordering: bool = True,
        start_layer_index: int = 0,
        end_layer_offset: int = 0,
    ) -> List[Tensor]:
        """Gets the outputs of the feedback/backward layers given inputs `ys`.

        Parameters
        ----------
        ys : List[Tensor]
            A list of tensors, one per layer, which will be used as the inputs for each
            backward/feedback layer.

        forward_ordering: bool, optional
            Wether `ys` is ordered from front to back (default) or from back to front.
            The outputs are also returned in the same order as the inputs.

        start_layer_index: int, optional
            Index of the first backward layer to use.

        end_layer_offset: int, optional
            Offset from the end of the last layer to use.

        Returns
        -------
        List[Tensor]
            The outputs of each backward layer, ordered from front to back when
            `forward_ordering` is True (default), or from back to front when
            `forward_ordering` is False.
        """
        assert len(ys) == len(self.backward_net) - start_layer_index - end_layer_offset, (len(ys), len(self.backward_net), start_layer_index, end_layer_offset)
        # Reverse the inputs if required.
        inputs = reversed(ys) if forward_ordering else ys

        n_layers = len(self.backward_net)
        if forward_ordering:
            _start_layer_index = end_layer_offset
            _end_layer_index = n_layers - start_layer_index
        else:
            _start_layer_index = start_layer_index
            _end_layer_index = n_layers - end_layer_offset

        backward_layers = self.backward_net[_start_layer_index:_end_layer_index]
        outputs = [layer(y_i) for layer, y_i in zip(backward_layers, inputs)]
        return list(reversed(outputs)) if forward_ordering else outputs

    def forward_all(
        self,
        x: Tensor,
        allow_grads_between_layers: bool = False,
        start_index: int = 0,
        end_offset: int = 0,
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
            self.forward_net,
            x,
            allow_grads_between_layers=allow_grads_between_layers,
            start_index=start_index,
            end_offset=end_offset,
        )

    def backward_all(
        self,
        y: Tensor,
        allow_grads_between_layers: bool = False,
        start_index: int = 0,
        end_offset: int = 0,
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
        # NOTE: The `start-index` and `end_offset` arguments might be a bit confusing
        # here, because they apply to the backward network (reverse ordering compared to
        # the same in `forward_all`.
        return forward_accumulate(
            self.backward_net,
            y,
            allow_grads_between_layers=allow_grads_between_layers,
            start_index=start_index,
            end_offset=end_offset,
        )


# @torch.jit.script
def forward_accumulate(
    model: nn.Sequential,
    x: Tensor,
    allow_grads_between_layers: bool = False,
    start_index: int = 0,
    end_offset: int = 0,
) -> List[Tensor]:
    """ IDEA: Gather all the forward activations into a list. """
    activations: List[Tensor] = []
    n_layers = len(model)
    end_index = n_layers - end_offset
    for layer in model[start_index:end_index]:
        x = layer(x if allow_grads_between_layers else x.detach())
        activations.append(x)
    return activations

