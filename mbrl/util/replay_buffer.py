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
    SliceSampler,
    TensorDictReplayBuffer,
    TensorStorage,
)

from mbrl.types import TransitionBatch
from mbrl.util.torchrl_util import (
    tensordict_to_transition_batch,
    transition_batch_to_tensordict,
)


def _consolidate_batches(batches: Sequence[TransitionBatch]) -> TransitionBatch:
    len_batches = len(batches)
    b0 = batches[0]
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
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Sequence[int],
        action_shape: Sequence[int],
        obs_type: Type = np.float32,
        action_type: Type = np.float32,
        reward_type: Type = np.float32,
        rng: Optional[np.random.Generator] = None,
        max_trajectory_length: Optional[int] = None,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.obs_type = obs_type
        self.action_type = action_type
        self.reward_type = reward_type
        self._rng = rng if rng else np.random.default_rng()
        self.max_trajectory_length = max_trajectory_length

        self.cur_idx = 0
        self.num_stored = 0

        self.trajectory_indices: Optional[List[Tuple[int, int]]] = None
        if max_trajectory_length:
            self.trajectory_indices = []

        # Internal TorchRL ReplayBuffer
        # We use a TensorStorage to allow for advanced indexing and in-place updates
        # which matches the legacy behavior better.
        self._storage = TensorStorage(
            storage=TensorDict(
                {
                    "observation": torch.zeros((capacity + (max_trajectory_length or 0), *obs_shape), dtype=getattr(torch, str(np.dtype(obs_type)))),
                    "action": torch.zeros((capacity + (max_trajectory_length or 0), *action_shape), dtype=getattr(torch, str(np.dtype(action_type)))),
                    "next": {
                        "observation": torch.zeros((capacity + (max_trajectory_length or 0), *obs_shape), dtype=getattr(torch, str(np.dtype(obs_type)))),
                        "reward": torch.zeros((capacity + (max_trajectory_length or 0), 1), dtype=getattr(torch, str(np.dtype(reward_type)))),
                        "terminated": torch.zeros((capacity + (max_trajectory_length or 0), 1), dtype=torch.bool),
                        "truncated": torch.zeros((capacity + (max_trajectory_length or 0), 1), dtype=torch.bool),
                    },
                },
                batch_size=[capacity + (max_trajectory_length or 0)],
            )
        )
        self._torchrl_rb = TensorDictReplayBuffer(
            storage=self._storage,
            sampler=RandomSampler(),
        )

        self._start_last_trajectory = 0

    @property
    def obs(self):
        return self._batch_from_indices(np.arange(len(self._storage))).obs

    @property
    def next_obs(self):
        return self._batch_from_indices(np.arange(len(self._storage))).next_obs

    @property
    def action(self):
        return self._batch_from_indices(np.arange(len(self._storage))).act

    @property
    def reward(self):
        return self._batch_from_indices(np.arange(len(self._storage))).rewards

    @property
    def terminated(self):
        return self._batch_from_indices(np.arange(len(self._storage))).terminateds

    @property
    def truncated(self):
        return self._batch_from_indices(np.arange(len(self._storage))).truncateds

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
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ):
        """Adds a transition to the replay buffer."""
        obs = np.array(obs)
        action = np.array(action)
        next_obs = np.array(next_obs)
        batch = TransitionBatch(
            obs=obs[None, ...],
            act=action[None, ...],
            next_obs=next_obs[None, ...],
            rewards=np.array([reward], dtype=self.reward_type),
            terminateds=np.array([terminated], dtype=bool),
            truncateds=np.array([truncated], dtype=bool),
        )
        self.add_batch(
            batch.obs,
            batch.act,
            batch.next_obs,
            batch.rewards,
            batch.terminateds,
            batch.truncateds,
        )

    def add_batch(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ):
        """Adds a batch of transitions to the replay buffer."""
        batch = TransitionBatch(
            obs=obs,
            act=action,
            next_obs=next_obs,
            rewards=reward,
            terminateds=terminated,
            truncateds=truncated,
        )
        td = transition_batch_to_tensordict(batch)
        
        batch_size = obs.shape[0]
        # Manually manage indices to match legacy behavior
        for i in range(batch_size):
            self._storage[int(self.cur_idx)] = td[i]
            if self.stores_trajectories:
                self._trajectory_bookkeeping(bool(terminated[i] or truncated[i]))
            else:
                self.cur_idx = (self.cur_idx + 1) % self.capacity
                self.num_stored = min(self.num_stored + 1, self.capacity)

        if self.stores_trajectories:
             # trajectory_bookkeeping handles cur_idx and num_stored
             pass

    def sample(self, batch_size: int) -> TransitionBatch:
        """Samples a batch of transitions from the replay buffer."""
        td = self._torchrl_rb.sample(batch_size)
        return tensordict_to_transition_batch(td)

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
        return tensordict_to_transition_batch(td)

    def __len__(self):
        return self.num_stored

    def save(self, save_dir: Union[pathlib.Path, str]):
        """Saves the replay buffer to a given directory."""
        path = pathlib.Path(save_dir) / "replay_buffer.npz"
        all_td = self._torchrl_rb.storage[: self.num_stored]
        batch = tensordict_to_transition_batch(all_td)
        np.savez(
            path,
            obs=batch.obs,
            next_obs=batch.next_obs,
            action=batch.act,
            reward=batch.rewards,
            terminated=batch.terminateds,
            truncated=batch.truncateds,
            num_stored=self.num_stored,
            cur_idx=self.cur_idx,
            trajectory_indices=np.array(self.trajectory_indices, dtype=object) if self.trajectory_indices is not None else [],
        )

    def load(self, load_dir: Union[pathlib.Path, str]):
        """Loads the replay buffer from a given directory."""
        path = pathlib.Path(load_dir) / "replay_buffer.npz"
        data = np.load(path, allow_pickle=True)
        self.num_stored = data["num_stored"]
        self.cur_idx = data["cur_idx"]
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

    def get_all(self, shuffle: bool = False) -> TransitionBatch:
        """Returns all transitions stored in the replay buffer."""
        if self.num_stored == 0:
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
        return tensordict_to_transition_batch(td)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng
