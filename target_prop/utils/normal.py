from torch.distributions import Normal as Normal_
from torch import Tensor
from typing import Union

class Normal(Normal_):
    def __add__(self, other: Union[int, float, Tensor]) -> "Normal":
        return type(self)(
            loc=self.loc + other,
            scale=self.scale,
        )
    def __radd__(self, other: Union[int, float, Tensor]) -> "Normal":
        return self.__add__(other)
