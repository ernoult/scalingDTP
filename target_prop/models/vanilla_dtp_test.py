from typing import ClassVar, Tuple, Type

import pytest
import torch
from target_prop.backward_layers import invert
from torch import Tensor, nn

from .dtp_test import TestDTP as DTPTests
from .vanilla_dtp import (
    DTP,
    VanillaDTP,
    vanilla_DTP_feedback_loss,
    vanilla_DTP_feedback_loss_parallel,
)


class TestVanillaDTP(DTPTests):
    # The type of model to test. (In this case, Vanilla DTP)
    model_class: ClassVar[Type[DTP]] = VanillaDTP


@pytest.mark.xfail(reason="TODO: Still a small difference.")
@pytest.mark.parametrize(
    "forward_layer_and_input", [(nn.Conv2d(3, 16, kernel_size=3), torch.ones([1, 3, 32, 32]))]
)
@pytest.mark.parametrize("seed", [123])
@pytest.mark.parametrize("noise_scale", [0.1, 0.01])
@pytest.mark.parametrize("num_samples", [1, 2])
def test_feedback_loss_functions(
    forward_layer_and_input: Tuple[nn.Module, Tensor],
    seed: int,
    noise_scale: float,
    num_samples: int,
):
    # TODO: There is still a small difference between the values obtained with the sequential vs
    # parallel versions of the feedback loss calculations. Using the sequential one for now, just to
    # be "safe".
    # TODO: The weights of the forward layer aren't fully seeded yet, since they are inside the
    # parametrize list above.
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    forward_layer, input = forward_layer_and_input
    feedback_layer = invert(forward_layer)
    output = forward_layer(input)

    kwargs = dict(
        forward_layer=forward_layer,
        feedback_layer=feedback_layer,
        input=input,
        output=output,
        noise_scale=noise_scale,
        noise_samples=num_samples,
    )
    rng_state = torch.random.get_rng_state()
    f_state_dict = {k: v.detach().clone() for k, v in forward_layer.state_dict().items()}
    b_state_dict = {k: v.detach().clone() for k, v in feedback_layer.state_dict().items()}
    seq_result = vanilla_DTP_feedback_loss(**kwargs)  # type: ignore

    torch.random.set_rng_state(rng_state)
    # Reload the state dicts, just in case doing a forward pass changes anything about the model
    # state somehow (e.g. maybe BatchNorm?)
    forward_layer.load_state_dict(f_state_dict)  # type: ignore
    feedback_layer.load_state_dict(b_state_dict)
    parallel_result = vanilla_DTP_feedback_loss_parallel(**kwargs)  # type: ignore
    assert seq_result.item() == parallel_result.item()
