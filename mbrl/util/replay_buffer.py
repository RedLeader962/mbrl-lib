# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import warnings
from typing import Any, Dict, List, Optional, Sequence, Sized, Tuple, Type, Union

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import (
    ListStorage,
    RandomSampler,
    ReplayBuffer as TorchRLReplayBuffer,
    SamplerWithoutReplacement,
    SliceSampler,
    TensorDictReplayBuffer,
    TensorStorage,
)

import mbrl.types as mbrl_types
from mbrl.types import TransitionBatch
from mbrl.util.torchrl_util import (
    tensordict_to_transition_batch,
    transition_batch_to_tensordict,
)


def _to_torch_dtype(dtype_spec: Union[str, torch.dtype, np.dtype]) -> torch.dtype:
    """Convert a dtype specification (numpy or torch) to a torch.dtype.

    Handles np.float32, np.dtype('float32'), torch.float32, and string
    representations like 'float32'.
    """
    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec
    try:
        # Works for np.float32, np.dtype('float32'), etc.
        return getattr(torch, str(np.dtype(dtype_spec)))
    except (TypeError, AttributeError):
        raise TypeError(
            f"Cannot convert {dtype_spec!r} (type={type(dtype_spec).__name__}) "
            f"to a torch.dtype."
        )


def _consolidate_batches(batches: Sequence[TransitionBatch]) -> TransitionBatch:
    len_batches = len(batches)
    b0 = batches[0]
    # Support both torch and numpy fields
    if isinstance(b0.obs, torch.Tensor):
        obs = torch.empty((len_batches,) + b0.obs.shape, dtype=b0.obs.dtype)
        act = torch.empty((len_batches,) + b0.act.shape, dtype=b0.act.dtype)
        next_obs = torch.empty((len_batches,) + b0.obs.shape, dtype=b0.obs.dtype)
        rewards = torch.empty((len_batches,) + b0.rewards.shape, dtype=torch.float32)
        terminateds = torch.empty((len_batches,) + b0.terminateds.shape, dtype=torch.bool)
        truncateds = torch.empty((len_batches,) + b0.truncateds.shape, dtype=torch.bool)
    else:
        obs = np.empty((len_batches,) + b0.obs.shape, dtype=b0.obs.dtype)
        act = np.empty((len_batches,) + b0.act.shape, dtype=b0.act.dtype)
        next_obs = np.empty((len_batches,) + b0.obs.shape, dtype=b0.obs.dtype)
        rewards = np.empty((len_batches,) + b0.rewards.shape, dtype=np.float32)
        terminateds = np.empty((len_batches,) + b0.terminateds.shape, dtype=bool)
        truncateds = np.empty((len_batches,) + b0.truncateds.shape, dtype=bool)
    for i, b in enumerate(batches):
        obs[i] = b.obs
        act[i] = b.act
        next_obs[i] = b.next_obs
        rewards[i] = b.rewards
        terminateds[i] = b.terminateds
        truncateds[i] = b.truncateds
    return TransitionBatch(obs, act, next_obs, rewards, terminateds, truncateds)


class TransitionIterator:
    """An iterator for batches of transitions.

    The iterator can be used doing:

    .. code-block:: python

       for batch in batch_iterator:
           do_something_with_batch()

    Rather than be constructed directly, the preferred way to use objects of this class
    is for the user to obtain them from :class:`ReplayBuffer`.

    Args:
        transitions (:class:`TransitionBatch`): the transition data used to built
            the iterator.
        batch_size (int): the batch size to use when iterating over the stored data.
        shuffle_each_epoch (bool): if ``True`` the iteration order is shuffled everytime a
            loop over the data is completed. Defaults to ``False``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.
    """

    def __init__(
        self,
        transitions: TransitionBatch,
        batch_size: int,
        shuffle_each_epoch: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self.transitions = transitions
        self.num_stored = len(transitions)
        self._order: np.ndarray = np.arange(self.num_stored)
        self.batch_size = batch_size
        self._current_batch = 0
        self._shuffle_each_epoch = shuffle_each_epoch
        self._rng = rng if rng is not None else np.random.default_rng()

    def _get_indices_next_batch(self) -> Sized:
        start_idx = self._current_batch * self.batch_size
        if start_idx >= self.num_stored:
            raise StopIteration
        end_idx = min((self._current_batch + 1) * self.batch_size, self.num_stored)
        order_indices = range(start_idx, end_idx)
        indices = self._order[order_indices]
        self._current_batch += 1
        return indices

    def __iter__(self):
        self._current_batch = 0
        if self._shuffle_each_epoch:
            self._order = self._rng.permutation(self.num_stored)
        return self

    def __next__(self):
        return self[self._get_indices_next_batch()]

    def ensemble_size(self):
        return 0

    def __len__(self):
        return (self.num_stored - 1) // self.batch_size + 1

    def __getitem__(self, item):
        return self.transitions[item]


class BootstrapIterator(TransitionIterator):
    """A transition iterator that can be used to train ensemble of bootstrapped models.

    When iterating, this iterator samples from a different set of indices for each model in the
    ensemble, essentially assigning a different dataset to each model. Each batch is of
    shape (ensemble_size x batch_size x obs_size) -- likewise for
    actions, rewards, terminateds, truncateds.

    Args:
        transitions (:class:`TransitionBatch`): the transition data used to built
            the iterator.
        batch_size (int): the batch size to use when iterating over the stored data.
        ensemble_size (int): the number of models in the ensemble.
        shuffle_each_epoch (bool): if ``True`` the iteration order is shuffled everytime a
            loop over the data is completed. Defaults to ``False``.
        permute_indices (boot): if ``True`` the bootstrap datasets are just
            permutations of the original data. If ``False`` they are sampled with
            replacement. Defaults to ``True``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.

    Note:
        If you want to make other custom types of iterators compatible with ensembles
        of bootstrapped models, the easiest way is to subclass :class:`BootstrapIterator`
        and overwrite ``__getitem()__`` method. The sampling methods of this class
        will then batch the result of of ``self[item]`` along a model dimension, where each
        batch is sampled independently.
    """

    def __init__(
        self,
        transitions: TransitionBatch,
        batch_size: int,
        ensemble_size: int,
        shuffle_each_epoch: bool = False,
        permute_indices: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(
            transitions, batch_size, shuffle_each_epoch=shuffle_each_epoch, rng=rng
        )
        self._ensemble_size = ensemble_size
        self._permute_indices = permute_indices
        self._bootstrap_iter = ensemble_size > 1
        self.member_indices = self._sample_member_indices()

    def _sample_member_indices(self) -> np.ndarray:
        member_indices = np.empty((self.ensemble_size, self.num_stored), dtype=int)
        if self._permute_indices:
            for i in range(self.ensemble_size):
                member_indices[i] = self._rng.permutation(self.num_stored)
        else:
            member_indices = self._rng.choice(
                self.num_stored,
                size=(self.ensemble_size, self.num_stored),
                replace=True,
            )
        return member_indices

    def __iter__(self):
        super().__iter__()
        return self

    def __next__(self):
        if not self._bootstrap_iter:
            return super().__next__()
        indices = self._get_indices_next_batch()
        batches = []
        for member_idx in self.member_indices:
            content_indices = member_idx[indices]
            batches.append(self[content_indices])
        return _consolidate_batches(batches)

    def toggle_bootstrap(self):
        """Toggles whether the iterator returns a batch per model or a single batch."""
        if self.ensemble_size > 1:
            self._bootstrap_iter = not self._bootstrap_iter

    @property
    def ensemble_size(self):
        return self._ensemble_size


def _sequence_getitem_impl(
    transitions: TransitionBatch,
    batch_size: int,
    sequence_length: int,
    valid_starts: np.ndarray,
    item: Any,
):
    start_indices = valid_starts[item].repeat(sequence_length)
    increment_array = np.tile(np.arange(sequence_length), len(item))
    full_trajectory_indices = start_indices + increment_array
    return transitions[full_trajectory_indices].add_new_batch_dim(
        min(batch_size, len(item))
    )


class SequenceTransitionIterator(BootstrapIterator):
    """
    A transition iterator that provides sequences of transitions.

    Returns batches of short sequences of transitions in the buffer, corresponding
    to fixed-length segments of the trajectories indicated by the given trajectory indices.
    The start states of all trajectories are sampled uniformly at random from the set of
    states from which a sequence of the desired length can be started.

    When iterating over this object, batches might contain overlapping trajectories. By default,
    a full loop over this iterator will return as many samples as valid start states
    there are (but start states could be repeated, they are sampled with replacement). Since
    this is unlikely necessary, you can use input argument ``batches_per_epoch`` to
    only return a smaller number of batches.

    Note that this is a bootstrap iterator, so it can return an extra model dimension,
    where each batch is sampled independently. By default, each observation batch is of
    shape (ensemble_size x batch_size x sequence_length x obs_size)  -- likewise for
    actions, rewards, terminateds, truncateds. If not in bootstrap mode,
    then the ensemble_size dimension is removed.


    Args:
        transitions (:class:`TransitionBatch`): the transition data used to built
            the iterator.
        trajectory_indices (list(tuple(int, int)): a list of [start, end) indices for
            trajectories.
        batch_size (int): the batch size to use when iterating over the stored data.
        sequence_length (int): the length of the sequences returned.
        ensemble_size (int): the number of models in the ensemble.
        shuffle_each_epoch (bool): if ``True`` the iteration order is shuffled everytime a
            loop over the data is completed. Defaults to ``False``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If ``None`` (default value), a new default generator will be used.
        max_batches_per_loop (int, optional): if given, specifies how many batches
            to return (at most) over a full loop of the iterator.
    """

    def __init__(
        self,
        transitions: TransitionBatch,
        trajectory_indices: Sequence[Tuple[int, int]],
        batch_size: int,
        sequence_length: int,
        ensemble_size: int,
        shuffle_each_epoch: bool = False,
        rng: Optional[np.random.Generator] = None,
        max_batches_per_loop: Optional[int] = None,
    ):
        self._sequence_length = sequence_length
        self._valid_starts = self._get_indices_valid_starts(
            trajectory_indices, sequence_length
        )
        self._max_batches_per_loop = max_batches_per_loop
        if len(self._valid_starts) < 0.5 * len(trajectory_indices):
            warnings.warn(
                "More than 50% of the trajectories were discarded for being shorter "
                "than the specified length."
            )
        # no need to pass transitions to super(), since it's only used by __getitem__,
        # which this class replaces. Passing the set of possible starts allow us to
        # use all the indexing machinery of the superclasses.
        super().__init__(
            self._valid_starts,  # type: ignore
            batch_size,
            ensemble_size,
            shuffle_each_epoch=shuffle_each_epoch,
            permute_indices=False,
            rng=rng,
        )
        self.transitions = transitions

    @staticmethod
    def _get_indices_valid_starts(
        trajectory_indices: Sequence[Tuple[int, int]],
        sequence_length: int,
    ) -> np.ndarray:
        # This is memory and time inefficient but it's only done once when creating the
        # iterator. It's a good price to pay for now, since it simplifies things
        # enormously and it's less error prone
        valid_starts = []
        for start, end in trajectory_indices:
            if end - start < sequence_length:
                continue
            valid_starts.extend(list(range(start, end - sequence_length + 1)))
        return np.array(valid_starts)

    def __iter__(self):
        super().__iter__()
        return self

    def __next__(self):
        if (
            self._max_batches_per_loop is not None
            and self._current_batch >= self._max_batches_per_loop
        ):
            raise StopIteration
        return super().__next__()

    def __len__(self):
        if self._max_batches_per_loop is not None:
            return min(super().__len__(), self._max_batches_per_loop)
        else:
            return super().__len__()

    def __getitem__(self, item):
        return _sequence_getitem_impl(
            self.transitions,
            self.batch_size,
            self._sequence_length,
            self._valid_starts,
            item,
        )


class SequenceTransitionSampler(TransitionIterator):
    """A transition iterator that provides sequences of transitions sampled at random.

    Returns batches of short sequences of transitions in the buffer, corresponding
    to fixed-length segments of the trajectories indicated by the given trajectory indices.
    The start states of all trajectories are sampled uniformly at random from the set of
    states from which a sequence of the desired length can be started.
    When iterating over this object, batches might contain overlapping trajectories.

    Args:
        transitions (:class:`TransitionBatch`): the transition data used to built
            the iterator.
        trajectory_indices (list(tuple(int, int)): a list of [start, end) indices for
            trajectories.
        batch_size (int): the batch size to use when iterating over the stored data.
        sequence_length (int): the length of the sequences returned.
        batches_per_loop (int): if given, specifies how many batches
            to return (at most) over a full loop of the iterator.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If ``None`` (default value), a new default generator will be used.
    """

    def __init__(
        self,
        transitions: TransitionBatch,
        trajectory_indices: Sequence[Tuple[int, int]],
        batch_size: int,
        sequence_length: int,
        batches_per_loop: int,
        rng: Optional[np.random.Generator] = None,
    ):
        self._sequence_length = sequence_length
        self._valid_starts = self._get_indices_valid_starts(
            trajectory_indices, sequence_length
        )
        self._batches_per_loop = batches_per_loop
        if len(self._valid_starts) < 0.5 * len(trajectory_indices):
            warnings.warn(
                "More than 50% of the trajectories were discarded for being shorter "
                "than the specified length."
            )
        # no need to pass transitions to super(), since it's only used by __getitem__,
        # which this class replaces. Passing the set of possible starts allow us to
        # use all the indexing machinery of the superclasses.
        super().__init__(
            self._valid_starts,  # type: ignore
            batch_size,
            shuffle_each_epoch=True,  # this is ignored
            rng=rng,
        )
        self.transitions = transitions

    @staticmethod
    def _get_indices_valid_starts(
        trajectory_indices: Sequence[Tuple[int, int]],
        sequence_length: int,
    ) -> np.ndarray:
        # This is memory and time inefficient but it's only done once when creating the
        # iterator. It's a good price to pay for now, since it simplifies things
        # enormously and it's less error prone
        valid_starts = []
        for start, end in trajectory_indices:
            if end - start < sequence_length:
                continue
            valid_starts.extend(list(range(start, end - sequence_length + 1)))
        return np.array(valid_starts)

    def __iter__(self):
        self._current_batch = 0
        return self

    def __next__(self):
        if self._current_batch >= self._batches_per_loop:
            raise StopIteration
        self._current_batch += 1
        indices = self._rng.choice(self.num_stored, size=self.batch_size, replace=True)
        return self[indices]

    def __len__(self):
        return self._batches_per_loop

    def __getitem__(self, item):
        return _sequence_getitem_impl(
            self.transitions,
            self.batch_size,
            self._sequence_length,
            self._valid_starts,
            item,
        )


class ReplayBuffer:
    """A replay buffer for transitions.

    This class is now a wrapper around torchrl.data.ReplayBuffer.
    It maintains the same API as the original ReplayBuffer.

    Args:
        capacity (int): the maximum number of transitions to store.
        obs_shape (sequence of ints): the shape of observations.
        action_shape (sequence of ints): the shape of actions.
        obs_type (type): the numpy dtype for observations.
        action_type (type): the numpy dtype for actions.
        reward_type (type): the numpy dtype for rewards.
        rng (np.random.Generator, optional): a random number generator.
        max_trajectory_length (int, optional): if given, the buffer will
            store trajectory information.
        output_torch (bool, optional): if ``True`` (default), output methods
            (``get_all``, ``sample``, ``_batch_from_indices``) return
            ``TransitionBatch`` with ``torch.Tensor`` fields.  If ``False``,
            fields are ``np.ndarray`` (legacy behaviour) and a
            ``FutureWarning`` is emitted.
    """

    _BUFFER_FNAME = "replay_buffer.pt"
    _LEGACY_BUFFER_FNAME = "replay_buffer.npz"

    def __init__(
        self,
        capacity: int,
        obs_shape: Sequence[int],
        action_shape: Sequence[int],
        obs_type: Union[torch.dtype, np.dtype, str] = torch.float32,
        action_type: Union[torch.dtype, np.dtype, str] = torch.float32,
        reward_type: Union[torch.dtype, np.dtype, str] = torch.float32,
        rng: Optional[np.random.Generator] = None,
        max_trajectory_length: Optional[int] = None,
        output_torch: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        # NOTE (C1 + F-C1b): ``device`` keeps the ReplayBuffer storage on the
        # provided torch device. When ``None`` (default), storage is allocated
        # on CPU — bit-exact with the pre-patch code path. When set to a CUDA
        # device, every underlying ``torch.zeros(...)`` (and the resulting
        # ``TensorDict`` / ``TensorStorage``) is created directly on-device,
        # so batch gathers performed by the iterator stay on the GPU and the
        # synchronous H→D copy that used to happen on every training batch
        # disappears. Introduced by actions ``C1`` of the RLRC Training Speed
        # & Efficiency ``.junie`` plan and ``F-C1b`` of the stage-1 follow-up
        # ``.junie`` plan (merged into a single submodule patch — see reports
        # ``report_tensordict_torchrl_replaybuffer_perf_20260421.md`` and the
        # S2-3 execution report).
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.obs_type = obs_type
        self.action_type = action_type
        self.reward_type = reward_type
        self._rng = rng if rng else np.random.default_rng()
        self.max_trajectory_length = max_trajectory_length
        self.device: Optional[torch.device] = (
            torch.device(device) if device is not None else None
        )

        # Output format control
        self._output_torch = output_torch if output_torch is not None else True
        if not self._output_torch:
            warnings.warn(
                "ReplayBuffer is configured to return numpy arrays. This mode "
                "is deprecated and will be removed in a future version. "
                "Set output_torch=True or omit the parameter to use torch "
                "tensors for better performance.",
                FutureWarning,
                stacklevel=2,
            )

        self.cur_idx = 0
        self.num_stored = 0

        self.trajectory_indices: Optional[List[Tuple[int, int]]] = None
        if max_trajectory_length:
            self.trajectory_indices = []

        # Internal TorchRL ReplayBuffer
        # We use a TensorStorage to allow for advanced indexing and in-place updates
        # which matches the legacy behavior better.
        # NOTE (C1 + F-C1b): when ``self.device`` is not ``None`` we allocate
        # every tensor directly on that device by passing ``device=`` to the
        # ``torch.zeros(...)`` calls and the enclosing ``TensorDict``. When
        # ``self.device`` is ``None`` the ``device=None`` kwarg matches the
        # pre-patch implicit CPU allocation — bit-exact.
        _dev = self.device
        _total = capacity + (max_trajectory_length or 0)
        self._storage = TensorStorage(
            storage=TensorDict(
                {
                    "observation": torch.zeros((_total, *obs_shape), dtype=_to_torch_dtype(obs_type), device=_dev),
                    "action": torch.zeros((_total, *action_shape), dtype=_to_torch_dtype(action_type), device=_dev),
                    "next": {
                        "observation": torch.zeros((_total, *obs_shape), dtype=_to_torch_dtype(obs_type), device=_dev),
                        "reward": torch.zeros((_total, 1), dtype=_to_torch_dtype(reward_type), device=_dev),
                        "terminated": torch.zeros((_total, 1), dtype=torch.bool, device=_dev),
                        "truncated": torch.zeros((_total, 1), dtype=torch.bool, device=_dev),
                    },
                },
                batch_size=[_total],
                device=_dev,
            )
        )
        # NOTE (F-C0-sampler): use ``SamplerWithoutReplacement`` to match
        # pre-refactor mbrl-lib upstream semantics
        # (``np.random.choice(..., replace=False)``). In ``torchrl >= 0.11``,
        # ``RandomSampler`` is hardcoded *with* replacement and takes no
        # ``replacement`` kwarg; the correct drop-in for unique-draw sampling
        # is ``SamplerWithoutReplacement`` (see
        # ``torchrl.data.replay_buffers.samplers``). Introduced by action
        # ``F-C0-sampler`` (stage 1) of the RLRC Training Speed & Efficiency
        # stage-1 follow-up ``.junie`` plan
        # (``performance_training_speed_efficiency_stage1_followup_plan_20260421.md``);
        # see also ``report_randomsampler_replacement_landmine_20260421.md``.
        self._torchrl_rb = TensorDictReplayBuffer(
            storage=self._storage,
            sampler=SamplerWithoutReplacement(drop_last=False, shuffle=True),
        )

        self._start_last_trajectory = 0

    @property
    def obs(self):
        """Returns stored observations as numpy (legacy inspection API)."""
        td = self._storage[:self.num_stored]
        return td["observation"].detach().cpu().numpy()

    @property
    def next_obs(self):
        """Returns stored next observations as numpy (legacy inspection API)."""
        td = self._storage[:self.num_stored]
        return td["next", "observation"].detach().cpu().numpy()

    @property
    def action(self):
        """Returns stored actions as numpy (legacy inspection API)."""
        td = self._storage[:self.num_stored]
        return td["action"].detach().cpu().numpy()

    @property
    def reward(self):
        """Returns stored rewards as numpy (legacy inspection API)."""
        td = self._storage[:self.num_stored]
        return td["next", "reward"].squeeze(-1).detach().cpu().numpy()

    @reward.setter
    def reward(self, value):
        """Sets rewards in the underlying storage (legacy mutation API)."""
        value_t = torch.as_tensor(np.asarray(value), dtype=_to_torch_dtype(self.reward_type))
        if value_t.ndim == 1:
            value_t = value_t.unsqueeze(-1)
        indices = np.arange(len(value_t))
        td = self._torchrl_rb.storage[indices]
        td["next", "reward"] = value_t
        self._torchrl_rb.storage[indices] = td

    @property
    def terminated(self):
        """Returns stored terminated flags as numpy (legacy inspection API)."""
        td = self._storage[:self.num_stored]
        return td["next", "terminated"].squeeze(-1).detach().cpu().numpy()

    @property
    def truncated(self):
        """Returns stored truncated flags as numpy (legacy inspection API)."""
        td = self._storage[:self.num_stored]
        return td["next", "truncated"].squeeze(-1).detach().cpu().numpy()

    @property
    def stores_trajectories(self) -> bool:
        return self.trajectory_indices is not None

    @staticmethod
    def _check_overlap(segment1: Tuple[int, int], segment2: Tuple[int, int]) -> bool:
        s1, e1 = segment1
        s2, e2 = segment2
        return (s1 <= s2 < e1) or (s1 < e2 <= e1)

    def remove_overlapping_trajectories(self, new_trajectory: Tuple[int, int]):
        cnt = 0
        for traj in self.trajectory_indices:
            if self._check_overlap(new_trajectory, traj):
                cnt += 1
            else:
                break
        for _ in range(cnt):
            self.trajectory_indices.pop(0)

    def _trajectory_bookkeeping(self, terminated: bool):
        self.cur_idx += 1
        if self.num_stored < self.capacity:
            self.num_stored += 1
        if self.cur_idx >= self.capacity:
            self.num_stored = max(self.num_stored, self.cur_idx)
        if terminated:
            self.close_trajectory()
        else:
            partial_trajectory = (self._start_last_trajectory, self.cur_idx + 1)
            self.remove_overlapping_trajectories(partial_trajectory)
        if self.cur_idx >= (self.capacity + (self.max_trajectory_length or 0)):
            warnings.warn(
                "The replay buffer was filled before current trajectory finished. "
                "The history of the current partial trajectory will be discarded. "
                "Make sure you set `max_trajectory_length` to the appropriate value "
                "for your problem."
            )
            self._start_last_trajectory = 0
            self.cur_idx = 0
            self.num_stored = self.capacity

    def close_trajectory(self):
        new_trajectory = (self._start_last_trajectory, self.cur_idx)
        self.remove_overlapping_trajectories(new_trajectory)
        self.trajectory_indices.append(new_trajectory)

        if self.cur_idx - self._start_last_trajectory > self.capacity:
            warnings.warn(
                "A trajectory was saved with length longer than expected. "
                "Unexpected behavior might occur."
            )

        if self.cur_idx >= self.capacity:
            self.cur_idx = 0
        self._start_last_trajectory = self.cur_idx

    def add(
        self,
        obs: mbrl_types.TensorType,
        action: mbrl_types.TensorType,
        next_obs: mbrl_types.TensorType,
        reward: float,
        terminated: bool,
        truncated: bool,
    ):
        """Adds a transition to the replay buffer.

        Accepts both numpy arrays and torch tensors (including CUDA
        tensors — mirrors the `add_batch(...)` tensor-type guard so the
        C1/F-C1b device-resident storage path does not trip on
        `np.asarray(<cuda tensor>)`).
        """
        obs = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(np.asarray(obs))
        action = action if isinstance(action, torch.Tensor) else torch.as_tensor(np.asarray(action))
        next_obs = next_obs if isinstance(next_obs, torch.Tensor) else torch.as_tensor(np.asarray(next_obs))
        reward_t = torch.tensor([reward], dtype=_to_torch_dtype(self.reward_type))
        terminated_t = torch.tensor([terminated], dtype=torch.bool)
        truncated_t = torch.tensor([truncated], dtype=torch.bool)

        td = TensorDict(
            {
                "observation": obs.unsqueeze(0),
                "action": action.unsqueeze(0),
                "next": {
                    "observation": next_obs.unsqueeze(0),
                    "reward": reward_t.unsqueeze(-1),
                    "terminated": terminated_t.unsqueeze(-1),
                    "truncated": truncated_t.unsqueeze(-1),
                },
            },
            batch_size=[1],
        )
        # C1/F-C1b: align source TensorDict with storage device so the
        # per-sample `__setitem__` below does not silently fall back to
        # a CPU copy path.
        if self.device is not None:
            td = td.to(self.device)
        self._storage[int(self.cur_idx)] = td[0]
        if self.stores_trajectories:
            self._trajectory_bookkeeping(bool(terminated or truncated))
        else:
            self.cur_idx = (self.cur_idx + 1) % self.capacity
            self.num_stored = min(self.num_stored + 1, self.capacity)

    def add_batch(
        self,
        obs: mbrl_types.TensorType,
        action: mbrl_types.TensorType,
        next_obs: mbrl_types.TensorType,
        reward: mbrl_types.TensorType,
        terminated: mbrl_types.TensorType,
        truncated: mbrl_types.TensorType,
    ):
        """Adds a batch of transitions to the replay buffer.

        Accepts both numpy arrays and torch tensors.
        """
        obs = torch.as_tensor(np.asarray(obs)) if not isinstance(obs, torch.Tensor) else obs
        action = torch.as_tensor(np.asarray(action)) if not isinstance(action, torch.Tensor) else action
        next_obs = torch.as_tensor(np.asarray(next_obs)) if not isinstance(next_obs, torch.Tensor) else next_obs
        reward = torch.as_tensor(np.asarray(reward)) if not isinstance(reward, torch.Tensor) else reward
        terminated = torch.as_tensor(np.asarray(terminated)) if not isinstance(terminated, torch.Tensor) else terminated
        truncated = torch.as_tensor(np.asarray(truncated)) if not isinstance(truncated, torch.Tensor) else truncated

        # Ensure reward/terminated/truncated have correct shape for storage
        if reward.ndim == 1:
            reward_store = reward.unsqueeze(-1)
        else:
            reward_store = reward
        if terminated.ndim == 1:
            terminated_store = terminated.unsqueeze(-1)
        else:
            terminated_store = terminated
        if truncated.ndim == 1:
            truncated_store = truncated.unsqueeze(-1)
        else:
            truncated_store = truncated

        batch_size = obs.shape[0]

        if not self.stores_trajectories:
            # --- Fast path: vectorized batch write ---
            # Cast to storage dtypes to avoid dtype mismatch on index put
            obs_dtype = _to_torch_dtype(self.obs_type)
            act_dtype = _to_torch_dtype(self.action_type)
            rew_dtype = _to_torch_dtype(self.reward_type)
            obs = obs.to(obs_dtype)
            action = action.to(act_dtype)
            next_obs = next_obs.to(obs_dtype)
            reward_store = reward_store.to(rew_dtype)

            if batch_size >= self.capacity:
                # Batch larger than or equal to capacity: only keep the last
                # `capacity` items (earlier ones would be overwritten anyway).
                # Compute where cur_idx would land after writing all items
                final_cur_idx = (self.cur_idx + batch_size) % self.capacity
                start = batch_size - self.capacity
                obs = obs[start:]
                action = action[start:]
                next_obs = next_obs[start:]
                reward_store = reward_store[start:]
                terminated_store = terminated_store[start:]
                truncated_store = truncated_store[start:]
                batch_size = self.capacity
                self.cur_idx = final_cur_idx  # write from where the tail begins

            end_idx = self.cur_idx + batch_size

            if end_idx <= self.capacity:
                # Contiguous write — no wrap-around
                indices = torch.arange(self.cur_idx, end_idx)
                td = TensorDict(
                    {
                        "observation": obs,
                        "action": action,
                        "next": {
                            "observation": next_obs,
                            "reward": reward_store,
                            "terminated": terminated_store,
                            "truncated": truncated_store,
                        },
                    },
                    batch_size=[batch_size],
                )
                self._storage[indices] = td
                self.cur_idx = end_idx % self.capacity
                self.num_stored = min(self.num_stored + batch_size, self.capacity)
            else:
                # Wrap-around: split into two contiguous writes
                first_chunk = self.capacity - self.cur_idx
                second_chunk = batch_size - first_chunk

                # First chunk: cur_idx → capacity
                indices_1 = torch.arange(self.cur_idx, self.capacity)
                td_1 = TensorDict(
                    {
                        "observation": obs[:first_chunk],
                        "action": action[:first_chunk],
                        "next": {
                            "observation": next_obs[:first_chunk],
                            "reward": reward_store[:first_chunk],
                            "terminated": terminated_store[:first_chunk],
                            "truncated": truncated_store[:first_chunk],
                        },
                    },
                    batch_size=[first_chunk],
                )
                self._storage[indices_1] = td_1

                # Second chunk: 0 → remainder
                indices_2 = torch.arange(0, second_chunk)
                td_2 = TensorDict(
                    {
                        "observation": obs[first_chunk:],
                        "action": action[first_chunk:],
                        "next": {
                            "observation": next_obs[first_chunk:],
                            "reward": reward_store[first_chunk:],
                            "terminated": terminated_store[first_chunk:],
                            "truncated": truncated_store[first_chunk:],
                        },
                    },
                    batch_size=[second_chunk],
                )
                self._storage[indices_2] = td_2

                self.cur_idx = second_chunk
                self.num_stored = min(self.num_stored + batch_size, self.capacity)
        else:
            # --- Slow path: per-item for trajectory bookkeeping ---
            for i in range(batch_size):
                self._storage[int(self.cur_idx)] = TensorDict(
                    {
                        "observation": obs[i],
                        "action": action[i],
                        "next": {
                            "observation": next_obs[i],
                            "reward": reward_store[i],
                            "terminated": terminated_store[i],
                            "truncated": truncated_store[i],
                        },
                    },
                    batch_size=[],
                )
                self._trajectory_bookkeeping(bool(terminated[i].item() if isinstance(terminated[i], torch.Tensor) else terminated[i]) or bool(truncated[i].item() if isinstance(truncated[i], torch.Tensor) else truncated[i]))

    def sample(self, batch_size: int) -> TransitionBatch:
        """Samples a batch of transitions from the replay buffer."""
        td = self._torchrl_rb.sample(batch_size)
        return tensordict_to_transition_batch(td, as_torch=self._output_torch)

    def sample_trajectory(self) -> Optional[TransitionBatch]:
        """Samples a full trajectory from the replay buffer."""
        if not self.stores_trajectories or len(self.trajectory_indices) == 0:
            return None
        idx = self._rng.choice(len(self.trajectory_indices))
        start, end = self.trajectory_indices[idx]
        if start < end:
            indices = np.arange(start, end)
        else:
            # if start > end, the trajectory is wrapped around the end of the buffer
            indices = np.concatenate([np.arange(start, self.capacity), np.arange(end)])
        return self._batch_from_indices(indices)

    def _batch_from_indices(self, indices: Sized) -> TransitionBatch:
        """Returns a batch of transitions from the given indices."""
        # TensorStorage supports advanced indexing
        td = self._torchrl_rb.storage[indices]
        return tensordict_to_transition_batch(td, as_torch=self._output_torch)

    def __len__(self):
        return self.num_stored

    def save(self, save_dir: Union[pathlib.Path, str]):
        """Saves the replay buffer to a torch file."""
        save_dir = pathlib.Path(save_dir)
        all_td = self._torchrl_rb.storage[:int(self.num_stored)]
        torch.save(
            {
                "tensordict": all_td.cpu(),
                "num_stored": int(self.num_stored),
                "cur_idx": int(self.cur_idx),
                "trajectory_indices": self.trajectory_indices,
            },
            save_dir / self._BUFFER_FNAME,
        )

    def load(self, load_dir: Union[pathlib.Path, str]):
        """Loads the replay buffer from a torch file, with legacy npz fallback."""
        load_dir = pathlib.Path(load_dir)
        pt_path = load_dir / self._BUFFER_FNAME
        npz_path = load_dir / self._LEGACY_BUFFER_FNAME

        if pt_path.exists():
            data = torch.load(pt_path, weights_only=False)
            td = data["tensordict"]
            self.num_stored = data["num_stored"]
            self.cur_idx = data["cur_idx"]
            if data.get("trajectory_indices") is not None:
                self.trajectory_indices = data["trajectory_indices"]
            # Write directly into storage
            self._storage[:len(td)] = td
        elif npz_path.exists():
            warnings.warn(
                f"Loading replay buffer from legacy numpy format "
                f"'{self._LEGACY_BUFFER_FNAME}'. Please re-save to migrate "
                f"to the new '{self._BUFFER_FNAME}' format.",
                FutureWarning,
            )
            data = np.load(npz_path, allow_pickle=True)
            if "num_stored" in data:
                self.num_stored = int(data["num_stored"])
            else:
                self.num_stored = len(data["obs"])
            if "cur_idx" in data:
                self.cur_idx = int(data["cur_idx"])
            else:
                self.cur_idx = self.num_stored % self.capacity
            batch = TransitionBatch(
                obs=data["obs"],
                act=data["action"],
                next_obs=data["next_obs"],
                rewards=data["reward"],
                terminateds=data["terminated"],
                truncateds=data["truncated"],
            )
            td = transition_batch_to_tensordict(batch)
            self._torchrl_rb.extend(td)
            if "trajectory_indices" in data and len(data["trajectory_indices"]):
                self.trajectory_indices = data["trajectory_indices"].tolist()
        else:
            raise FileNotFoundError(
                f"No replay buffer found at '{pt_path}' or '{npz_path}'."
            )

    def get_all(self, shuffle: bool = False) -> TransitionBatch:
        """Returns all transitions stored in the replay buffer."""
        if self.num_stored == 0:
            if self._output_torch:
                obs_dtype = _to_torch_dtype(self.obs_type)
                act_dtype = _to_torch_dtype(self.action_type)
                rew_dtype = _to_torch_dtype(self.reward_type)
                return TransitionBatch(
                    obs=torch.empty((0, *self.obs_shape), dtype=obs_dtype),
                    act=torch.empty((0, *self.action_shape), dtype=act_dtype),
                    next_obs=torch.empty((0, *self.obs_shape), dtype=obs_dtype),
                    rewards=torch.empty(0, dtype=rew_dtype),
                    terminateds=torch.empty(0, dtype=torch.bool),
                    truncateds=torch.empty(0, dtype=torch.bool),
                )
            else:
                return TransitionBatch(
                    obs=np.empty((0, *self.obs_shape), dtype=self.obs_type),
                    act=np.empty((0, *self.action_shape), dtype=self.action_type),
                    next_obs=np.empty((0, *self.obs_shape), dtype=self.obs_type),
                    rewards=np.empty(0, dtype=self.reward_type),
                    terminateds=np.empty(0, dtype=bool),
                    truncateds=np.empty(0, dtype=bool),
                )

        # Use slicing on storage for efficiency
        td = self._torchrl_rb.storage[:int(self.num_stored)]

        if shuffle:
            indices = self._rng.permutation(len(td))
            td = td[indices]
        return tensordict_to_transition_batch(td, as_torch=self._output_torch)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng
