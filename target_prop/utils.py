from torch.distributions import Normal as Normal_
from torch import Tensor
from typing import Union, Any


class Normal(Normal_):
    """ Little 'patch' for the `Normal` class from `torch.distributions` that makes it
    possible to add an offset to a distribution.
    """

    def __add__(self, other: Union[int, float, Tensor]) -> "Normal":
        return type(self)(
            loc=self.loc + other,
            scale=self.scale,
        )

    def __radd__(self, other: Union[int, float, Tensor]) -> Union["Normal", Any]:
        return self.__add__(other)
