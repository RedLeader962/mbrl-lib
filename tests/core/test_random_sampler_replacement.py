# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for the ``RandomSampler`` replacement semantics.

Permanent regression test suite. Introduced by action ``F-C0-sampler``
(stage 1) of the RLRC Training Speed & Efficiency stage-1 follow-up
``.junie`` plan
(``performance_training_speed_efficiency_stage1_followup_plan_20260421.md``).
Pins the fix applied to ``mbrl.util.replay_buffer.ReplayBuffer.__init__``
where the torchrl sampler is switched from ``RandomSampler`` (hardcoded
*with* replacement in ``torchrl >= 0.11``) to
``SamplerWithoutReplacement`` so that ``ReplayBuffer.sample(batch_size)``
preserves pre-refactor mbrl-lib upstream semantics (sampling without
replacement, i.e. ``np.random.choice(..., replace=False)``).

See also: ``report_randomsampler_replacement_landmine_20260421.md``.
"""
import numpy as np
import pytest
import torch

from torchrl.data import RandomSampler, SamplerWithoutReplacement, TensorDictReplayBuffer

import mbrl.util.replay_buffer as replay_buffer


def _make_buffer_with_unique_obs(n: int) -> replay_buffer.ReplayBuffer:
    """Fill a buffer of capacity ``n`` with ``n`` distinct observations.

    The first channel of ``obs`` equals the transition index ``i``, which
    lets tests recover the sampled indices from the returned batch.
    """
    rb = replay_buffer.ReplayBuffer(
        capacity=n,
        obs_shape=(2,),
        action_shape=(1,),
        obs_type=torch.float32,
        action_type=torch.float32,
        reward_type=torch.float32,
        rng=np.random.default_rng(0),
    )
    obs = np.stack(
        [np.array([float(i), 0.0], dtype=np.float32) for i in range(n)], axis=0
    )
    act = np.zeros((n, 1), dtype=np.float32)
    next_obs = obs.copy()
    reward = np.zeros((n,), dtype=np.float32)
    terminated = np.zeros((n,), dtype=bool)
    truncated = np.zeros((n,), dtype=bool)
    rb.add_batch(obs, act, next_obs, reward, terminated, truncated)
    assert rb.num_stored == n
    return rb


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_replay_buffer_sample_without_replacement_full_batch():
    """Sampling ``batch_size == num_stored`` must cover every index exactly once.

    With ``replacement=False`` (the fix), a single ``sample(batch_size=N)``
    call on a buffer with ``N`` distinct transitions must return a
    permutation of ``[0, N)`` — no duplicates.
    """
    n = 128
    _seed_all(1234)
    rb = _make_buffer_with_unique_obs(n)

    batch = rb.sample(batch_size=n)
    obs = batch.obs
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    indices = obs[:, 0].astype(np.int64)

    assert len(indices) == n
    assert len(set(indices.tolist())) == n, (
        "RandomSampler returned duplicates — replacement=True leaked back in. "
        "This contradicts pre-refactor mbrl-lib semantics; revisit "
        "ReplayBuffer.__init__ (action F-C0-sampler)."
    )
    assert sorted(indices.tolist()) == list(range(n))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_replay_buffer_sample_no_duplicates_partial_batch(seed):
    """Partial-batch draws must also be unique under ``replacement=False``."""
    n = 64
    batch_size = 32
    _seed_all(seed)
    rb = _make_buffer_with_unique_obs(n)

    batch = rb.sample(batch_size=batch_size)
    obs = batch.obs
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    indices = obs[:, 0].astype(np.int64)

    assert len(indices) == batch_size
    assert len(set(indices.tolist())) == batch_size, (
        f"Duplicate index detected under seed={seed}; expected uniqueness "
        "because RandomSampler must be instantiated with replacement=False."
    )


def test_replay_buffer_uses_random_sampler_without_replacement():
    """Structural assertion: the installed sampler is ``SamplerWithoutReplacement``.

    Protects the fix against silent regressions (e.g. a future edit that
    reverts to ``RandomSampler``, which in ``torchrl >= 0.11`` is hardcoded
    *with* replacement and therefore contradicts pre-refactor mbrl-lib
    semantics).
    """
    rb = _make_buffer_with_unique_obs(8)
    sampler = rb._torchrl_rb._sampler
    assert isinstance(sampler, SamplerWithoutReplacement), (
        f"Expected torchrl SamplerWithoutReplacement, got "
        f"{type(sampler).__name__}. See action F-C0-sampler."
    )
    assert not isinstance(sampler, RandomSampler), (
        "Sampler must not be RandomSampler (hardcoded with-replacement in "
        "torchrl >= 0.11). See action F-C0-sampler."
    )


def test_negative_control_random_sampler_produces_duplicates():
    """Negative control: ``RandomSampler`` (with-replacement) yields duplicates.

    Sanity check that the test apparatus would actually catch a regression
    to the old (landmine) sampler. Probability of 32 unique draws out of
    64 with replacement across 5 seeds is effectively zero
    (≈ 1e-7 per seed → ≈ 5e-7 cumulative).
    """
    n = 64
    batch_size = 32
    saw_duplicate = False
    for seed in range(5):
        _seed_all(seed)
        rb = _make_buffer_with_unique_obs(n)
        # Swap in the hardcoded-with-replacement sampler on the same storage.
        rb._torchrl_rb = TensorDictReplayBuffer(
            storage=rb._storage,
            sampler=RandomSampler(),
        )
        batch = rb.sample(batch_size=batch_size)
        obs = batch.obs
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        indices = obs[:, 0].astype(np.int64)
        if len(set(indices.tolist())) < batch_size:
            saw_duplicate = True
            break
    assert saw_duplicate, (
        "Negative control failed: no duplicates under RandomSampler across "
        "5 seeds. Test apparatus may be broken."
    )
