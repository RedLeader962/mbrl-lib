# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

TensorType = Union[torch.Tensor, np.ndarray]
RewardFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TermFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ObsProcessFnType = Callable[[TensorType], TensorType]
TrajectoryEvalFnType = Callable[[TensorType, torch.Tensor], torch.Tensor]

Transition = Tuple[
    TensorType, TensorType, TensorType, TensorType, TensorType, TensorType
]


def _type_aware_indexing(data: TensorType) -> TensorType | float | int | bool:
    if isinstance(data, torch.Tensor) and data.ndim == 0:
        return data.item()
    else:
        return data


@dataclass
class TransitionBatch:
    """Represents a batch of transitions"""

    obs: Optional[TensorType]
    act: Optional[TensorType]
    next_obs: Optional[TensorType]
    rewards: Optional[TensorType]
    terminateds: Optional[TensorType]
    truncateds: Optional[TensorType]

    def __len__(self):
        return self.obs.shape[0]

    def astuple(self) -> Transition:
        return (
            self.obs,
            self.act,
            self.next_obs,
            self.rewards,
            self.terminateds,
            self.truncateds,
        )

    def __getitem__(self, item):
        return TransitionBatch(
                _type_aware_indexing(self.obs[item]),
                _type_aware_indexing(self.act[item]),
                _type_aware_indexing(self.next_obs[item]),
                _type_aware_indexing(self.rewards[item]),
                _type_aware_indexing(self.terminateds[item]),
                _type_aware_indexing(self.truncateds[item]),
        )

    @staticmethod
    def _get_new_shape(old_shape: Tuple[int, ...], batch_size: int):
        new_shape = list((1,) + old_shape)
        new_shape[0] = batch_size
        new_shape[1] = old_shape[0] // batch_size
        return tuple(new_shape)

    def add_new_batch_dim(self, batch_size: int):
        if not len(self) % batch_size == 0:
            raise ValueError(
                "Current batch of transitions size is not a "
                "multiple of the new batch size. "
            )
        return TransitionBatch(
            self.obs.reshape(self._get_new_shape(self.obs.shape, batch_size)),
            self.act.reshape(self._get_new_shape(self.act.shape, batch_size)),
            self.next_obs.reshape(self._get_new_shape(self.obs.shape, batch_size)),
            self.rewards.reshape(self._get_new_shape(self.rewards.shape, batch_size)),
            self.terminateds.reshape(
                self._get_new_shape(self.terminateds.shape, batch_size)
            ),
            self.truncateds.reshape(
                self._get_new_shape(self.truncateds.shape, batch_size)
            ),
        )


ModelInput = Union[torch.Tensor, TransitionBatch]
