from __future__ import annotations

import contextlib
import warnings
from logging import getLogger as get_logger
from typing import Any, Iterable, Tuple, TypeVar, Union

from simple_parsing.helpers import field
from torch import Tensor, nn
from torch.distributions import Normal as Normal_
from torch.nn.parameter import Parameter

logger = get_logger(__name__)

T = TypeVar("T")
V = TypeVar("V", bound=Union[int, float])


def flag(v: Any, *args, **kwargs):
    return field(default=v, *args, nargs=1, **kwargs)


class Normal(Normal_):
    """Little 'patch' for the `Normal` class from `torch.distributions` that makes it
    possible to add an offset to a distribution.
    """

    def __add__(self, other: int | float | Tensor) -> Normal:
        return type(self)(
            loc=self.loc + other,
            scale=self.scale,
        )

    def __radd__(self, other: int | float | Tensor) -> Normal | Any:
        return self.__add__(other)


def get_list_of_values(values: V | list[V], out_length: int, name: str = "") -> list[V]:
    """Gets a list of values of length `out_length` from `values`.

    If `values` is a single value, it gets repeated `out_length` times to form the
    output list.
    If `values` is a list:
        - if it has the right length, it is returned unchanged;
        - if it is too short, the last value is repeated to get the right length;
        - if it is too long, a warning is raised and the extra values are dropped.

    Parameters
    ----------
    values : Union[V, List[V]]
        value or list of values.
    out_length : int
        desired output length.
    name : str, optional
        Name to use in the warning, empty by default.

    Returns
    -------
    List[V]
        List of values of length `out_length`
    """
    out: list[V]
    if isinstance(values, list):
        n_passed_values = len(values)
        if n_passed_values == out_length:
            out = values
        elif n_passed_values < out_length:
            # Repeat the last value.
            out = values + [values[-1]] * (out_length - n_passed_values)
        else:
            assert n_passed_values > out_length
            extra_values = values[out_length:]
            warnings.warn(
                UserWarning(
                    f"{n_passed_values} {name} values passed, but expected "
                    f"{out_length}! Dropping extra values: {extra_values}"
                )
            )
            out = values[:out_length]
    else:
        out = [values] * out_length
    return out


def is_trainable(layer: nn.Module) -> bool:
    return any(p.requires_grad for p in layer.parameters())


def named_trainable_parameters(module: nn.Module) -> Iterable[Tuple[str, Parameter]]:
    for name, param in module.named_parameters():
        if param.requires_grad:
            yield name, param


def repeat_batch(v: Tensor, n: int) -> Tensor:
    """Repeats the elements of tensor `v` `n` times along the batch dimension:

    Example:

    input:  [[1, 2, 3], [4, 5, 6]] of shape=(2, 3), n = 2
    output: [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]] of shape=(4, 3)

    >>> import torch
    >>> input = torch.as_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> repeat_batch(input, 2).tolist()
    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]]
    """
    b = v.shape[0]
    batched_v = v.unsqueeze(1).expand([b, n, *v.shape[1:]])  # [B, N, ...]
    flattened_batched_v = batched_v.reshape([b * n, *v.shape[1:]])  # [N*B, ...]
    return flattened_batched_v


def split_batch(batched_v: Tensor, n: int) -> Tensor:
    """Reshapes the output of `repeat_batch` from shape [B*N, ...] back to a shape of [B, N, ...]

    Example:

    input: [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [4.0, 5.0, 6.0], [4.1, 5.1, 6.1]], shape=(4, 3)
    output: [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]], shape=(2, 2, 3)

    >>> import numpy as np
    >>> input = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [4.0, 5.0, 6.0], [4.1, 5.1, 6.1]])
    >>> split_batch(input, 2).tolist()
    [[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]], [[4.0, 5.0, 6.0], [4.1, 5.1, 6.1]]]
    """
    assert batched_v.shape[0] % n == 0
    # [N*B, ...] -> [N, B, ...]
    return batched_v.reshape([-1, n, *batched_v.shape[1:]])


import random

import numpy as np
import torch


@contextlib.contextmanager
def make_reproducible(seed: int):
    """Makes the random operations within a block of code reproducible for a given seed."""
    # First: Get the starting random state, and restore it after.
    start_random_state = random.getstate()
    start_np_rng_state = np.random.get_state()
    with torch.random.fork_rng():
        # Set the random state, using the given seed.
        random.seed(seed)
        np_seed = random.randint(0, 2**32 - 1)
        np.random.seed(np_seed)

        torch_seed = random.randint(0, 2**32 - 1)
        torch.random.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

        yield

    # Restore the random state to the original state.
    np.random.set_state(start_np_rng_state)
    random.setstate(start_random_state)


from typing import Any, Callable, Type, TypeVar

from simple_parsing.helpers.serialization import encode
from simple_parsing.helpers.serialization.decoding import _register

_DecodingFn = Callable[[Any], T]


def register_decode(some_type: Type[T]) -> Callable[[_DecodingFn[T]], _DecodingFn[T]]:
    """Register a decoding function for the type `some_type`."""

    def wrapper(f: _DecodingFn[T]) -> _DecodingFn[T]:
        _register(some_type, f)
        return f

    return wrapper


@encode.register(torch.device)
def _encode_device(v: torch.device) -> str:
    return v.type


@register_decode(torch.device)
def _decode_device(v: str) -> torch.device:
    return torch.device(v)


from typing import Sequence

import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    TAKEN FROM https://github.com/ashleve/lightning-hydra-template/blob/6a92395ed6afd573fa44dd3a054a603acbdcac06/src/utils/__init__.py#L56

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    for field in print_order:
        queue.append(field) if field in config else logger.info(
            f"Field '{field}' not found in config"
        )

    for field in config:
        if field not in queue:
            queue.append(field)

    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


# @rank_zero_only
# def log_hyperparameters(
#     config: DictConfig,
#     model: pl.LightningModule,
#     datamodule: pl.LightningDataModule,
#     trainer: pl.Trainer,
#     callbacks: list[pl.Callback],
#     logger: list[LightningLoggerBase],
# ) -> None:
#     """Controls which config parts are saved by Lightning loggers.

#     Additionaly saves:
#     - number of model parameters
#     """

#     if not trainer.logger:
#         return

#     hparams = {}

#     # choose which parts of hydra config will be saved to loggers
#     hparams["model"] = config["model"]

#     # save number of model parameters
#     hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
#     hparams["model/params/trainable"] = sum(
#         p.numel() for p in model.parameters() if p.requires_grad
#     )
#     hparams["model/params/non_trainable"] = sum(
#         p.numel() for p in model.parameters() if not p.requires_grad
#     )

#     hparams["datamodule"] = config["datamodule"]
#     hparams["trainer"] = config["trainer"]

#     if "seed" in config:
#         hparams["seed"] = config["seed"]
#     if "callbacks" in config:
#         hparams["callbacks"] = config["callbacks"]

#     # send hparams to all loggers
#     trainer.logger.log_hyperparams(hparams)
