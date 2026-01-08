# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Union

import numpy as np
import torch
from tensordict import TensorDict

from mbrl.types import TransitionBatch


def transition_batch_to_tensordict(
    batch: TransitionBatch, device: Optional[Union[str, torch.device]] = None
) -> TensorDict:
    """Converts a TransitionBatch to a TensorDict.

    Args:
        batch (TransitionBatch): The batch to convert.
        device (str or torch.device, optional): The device to move the tensors to.

    Returns:
        (TensorDict): The converted TensorDict.
    """
    data = {
        "observation": torch.as_tensor(batch.obs, device=device),
        "action": torch.as_tensor(batch.act, device=device),
        "next": {
            "observation": torch.as_tensor(batch.next_obs, device=device),
            "reward": torch.as_tensor(batch.rewards, device=device),
            "terminated": torch.as_tensor(batch.terminateds, device=device),
            "truncated": torch.as_tensor(batch.truncateds, device=device),
        },
    }
    # Ensure rewards/terminated/truncated have an extra dimension if they are 1D
    for key in ["reward", "terminated", "truncated"]:
        if data["next"][key].ndim == 1:
            data["next"][key] = data["next"][key].unsqueeze(-1)

    batch_size = data["observation"].shape[0]
    return TensorDict(data, batch_size=[batch_size], device=device)


def tensordict_to_transition_batch(td: TensorDict) -> TransitionBatch:
    """Converts a TensorDict to a TransitionBatch.

    Args:
        td (TensorDict): The TensorDict to convert.

    Returns:
        (TransitionBatch): The converted TransitionBatch.
    """
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    return TransitionBatch(
        obs=_to_numpy(td["observation"]),
        act=_to_numpy(td["action"]),
        next_obs=_to_numpy(td["next", "observation"]),
        rewards=_to_numpy(td["next", "reward"].squeeze(-1))
        if td["next", "reward"].shape[-1] == 1
        else _to_numpy(td["next", "reward"]),
        terminateds=_to_numpy(td["next", "terminated"].squeeze(-1))
        if td["next", "terminated"].shape[-1] == 1
        else _to_numpy(td["next", "terminated"]),
        truncateds=_to_numpy(td["next", "truncated"].squeeze(-1))
        if td["next", "truncated"].shape[-1] == 1
        else _to_numpy(td["next", "truncated"]),
    )
