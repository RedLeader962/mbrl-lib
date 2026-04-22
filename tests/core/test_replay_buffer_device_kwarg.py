# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Permanent submodule regression suite. Introduced by actions ``C1`` of the
RLRC Training Speed & Efficiency ``.junie`` plan
(``performance_training_speed_efficiency_plan_20260421.md``) and ``F-C1b``
of the stage-1 follow-up ``.junie`` plan
(``performance_training_speed_efficiency_stage1_followup_plan_20260421.md``)
— merged into a single submodule patch.

Scope:
    - ``device=None`` preserves pre-patch CPU semantics (bit-exact default).
    - ``device="cpu"`` explicit CPU path yields identical results to the
      default and matches legacy storage device.
    - ``device=<cuda>`` (gated on ``torch.cuda.is_available()``) lands every
      underlying tensor on CUDA and sampling returns CUDA tensors.
    - Cross-device ``load(...)`` transfers from a CPU-written ``.pt`` file
      into a device-resident buffer without corruption.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
import torch

from mbrl.util.replay_buffer import ReplayBuffer


_OBS_SHAPE = (4,)
_ACT_SHAPE = (2,)
_CAPACITY = 64


def _fill(buf: ReplayBuffer, *, seed: int = 0, n: int = 40) -> np.random.Generator:
    """Deterministically populate ``buf`` with ``n`` transitions."""
    rng = np.random.default_rng(seed)
    obs_all = rng.standard_normal((n, *_OBS_SHAPE)).astype(np.float32)
    act_all = rng.standard_normal((n, *_ACT_SHAPE)).astype(np.float32)
    nobs_all = rng.standard_normal((n, *_OBS_SHAPE)).astype(np.float32)
    rew_all = rng.standard_normal(n).astype(np.float32)
    for i in range(n):
        buf.add(
            obs=obs_all[i],
            action=act_all[i],
            next_obs=nobs_all[i],
            reward=float(rew_all[i]),
            terminated=False,
            truncated=False,
        )
    return rng


def _make(device=None) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=_CAPACITY,
        obs_shape=_OBS_SHAPE,
        action_shape=_ACT_SHAPE,
        device=device,
    )


class TestDeviceKwargCPU:
    def test_device_none_default_is_legacy_cpu(self):
        buf = _make(device=None)
        assert buf.device is None
        _fill(buf)
        td = buf._storage[:buf.num_stored]
        assert td["observation"].device.type == "cpu"

    def test_device_cpu_explicit_is_normalized_to_torch_device(self):
        buf = _make(device="cpu")
        assert buf.device == torch.device("cpu")
        _fill(buf)
        td = buf._storage[:buf.num_stored]
        assert td["observation"].device.type == "cpu"

    def test_device_none_and_device_cpu_produce_equal_storage(self):
        buf_a = _make(device=None)
        buf_b = _make(device="cpu")
        _fill(buf_a, seed=1234)
        _fill(buf_b, seed=1234)
        td_a = buf_a._storage[:buf_a.num_stored]
        td_b = buf_b._storage[:buf_b.num_stored]
        for key in (
            ("observation",),
            ("action",),
            ("next", "observation"),
            ("next", "reward"),
            ("next", "terminated"),
            ("next", "truncated"),
        ):
            # TensorDict accepts nested tuples via __getitem__
            t_a = td_a.get(key) if len(key) == 1 else td_a[key]
            t_b = td_b.get(key) if len(key) == 1 else td_b[key]
            assert torch.equal(t_a, t_b), f"mismatch on {key}"

    def test_save_load_roundtrip_cpu_default(self, tmp_path: pathlib.Path):
        buf_w = _make(device=None)
        _fill(buf_w, seed=7)
        buf_w.save(tmp_path)

        buf_r = _make(device=None)
        buf_r.load(tmp_path)
        assert buf_r.num_stored == buf_w.num_stored

        td_w = buf_w._storage[: buf_w.num_stored]
        td_r = buf_r._storage[: buf_r.num_stored]
        assert torch.equal(td_w["observation"], td_r["observation"])
        assert torch.equal(td_w["action"], td_r["action"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only path")
class TestDeviceKwargCUDA:
    def test_device_cuda_places_storage_on_device(self):
        buf = _make(device="cuda")
        assert buf.device is not None and buf.device.type == "cuda"
        _fill(buf)
        td = buf._storage[:buf.num_stored]
        assert td["observation"].device.type == "cuda"
        assert td["action"].device.type == "cuda"
        assert td["next", "observation"].device.type == "cuda"
        assert td["next", "reward"].device.type == "cuda"

    def test_sample_returns_cuda_tensors_when_storage_is_cuda(self):
        buf = _make(device="cuda")
        _fill(buf, n=32)
        batch = buf.sample(8)
        # TransitionBatch fields should inherit the storage device
        assert isinstance(batch.obs, torch.Tensor)
        assert batch.obs.device.type == "cuda"
        assert batch.act.device.type == "cuda"
        assert batch.next_obs.device.type == "cuda"

    def test_cpu_save_cuda_load_roundtrip(self, tmp_path: pathlib.Path):
        # Save from a CPU buffer, load into a CUDA-resident buffer.
        buf_w = _make(device=None)
        _fill(buf_w, seed=42)
        buf_w.save(tmp_path)

        buf_r = _make(device="cuda")
        buf_r.load(tmp_path)
        assert buf_r.num_stored == buf_w.num_stored

        td_w = buf_w._storage[: buf_w.num_stored]
        td_r = buf_r._storage[: buf_r.num_stored]
        assert td_r["observation"].device.type == "cuda"
        # Values equal after cross-device comparison
        assert torch.equal(td_w["observation"], td_r["observation"].cpu())
        assert torch.equal(td_w["action"], td_r["action"].cpu())
