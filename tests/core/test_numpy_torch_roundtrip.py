# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the numpy→torch roundtrip elimination refactoring."""
import pathlib
import tempfile
import warnings

import numpy as np
import pytest
import torch

import mbrl.models
import mbrl.models.util as model_util
import mbrl.util.replay_buffer as replay_buffer
from mbrl.types import TransitionBatch
from mbrl.util.torchrl_util import (
    tensordict_to_transition_batch,
    transition_batch_to_tensordict,
)

_OBS_SHAPE = (4,)
_ACT_SHAPE = (2,)


# ------------------------------------------------------------------ #
#  4.1 — TransitionBatch Tensor Support
# ------------------------------------------------------------------ #
class TestTransitionBatchTensorSupport:
    def test_transition_batch_torch_fields(self):
        n = 10
        batch = TransitionBatch(
            obs=torch.randn(n, *_OBS_SHAPE),
            act=torch.randn(n, *_ACT_SHAPE),
            next_obs=torch.randn(n, *_OBS_SHAPE),
            rewards=torch.randn(n),
            terminateds=torch.zeros(n, dtype=torch.bool),
            truncateds=torch.zeros(n, dtype=torch.bool),
        )
        assert len(batch) == n
        tup = batch.astuple()
        assert len(tup) == 6
        sub = batch[:5]
        assert len(sub) == 5
        assert isinstance(sub.obs, torch.Tensor)

    def test_transition_batch_numpy_fields(self):
        n = 10
        batch = TransitionBatch(
            obs=np.random.randn(n, *_OBS_SHAPE).astype(np.float32),
            act=np.random.randn(n, *_ACT_SHAPE).astype(np.float32),
            next_obs=np.random.randn(n, *_OBS_SHAPE).astype(np.float32),
            rewards=np.random.randn(n).astype(np.float32),
            terminateds=np.zeros(n, dtype=bool),
            truncateds=np.zeros(n, dtype=bool),
        )
        assert len(batch) == n
        sub = batch[:5]
        assert isinstance(sub.obs, np.ndarray)

    def test_transition_batch_mixed_fields(self):
        n = 10
        batch = TransitionBatch(
            obs=torch.randn(n, *_OBS_SHAPE),
            act=np.random.randn(n, *_ACT_SHAPE).astype(np.float32),
            next_obs=torch.randn(n, *_OBS_SHAPE),
            rewards=np.random.randn(n).astype(np.float32),
            terminateds=torch.zeros(n, dtype=torch.bool),
            truncateds=np.zeros(n, dtype=bool),
        )
        assert len(batch) == n

    def test_transition_batch_add_new_batch_dim_torch(self):
        n = 12
        batch = TransitionBatch(
            obs=torch.randn(n, *_OBS_SHAPE),
            act=torch.randn(n, *_ACT_SHAPE),
            next_obs=torch.randn(n, *_OBS_SHAPE),
            rewards=torch.randn(n),
            terminateds=torch.zeros(n, dtype=torch.bool),
            truncateds=torch.zeros(n, dtype=torch.bool),
        )
        reshaped = batch.add_new_batch_dim(3)
        assert reshaped.obs.shape[0] == 3
        assert reshaped.obs.shape[1] == 4


# ------------------------------------------------------------------ #
#  4.2 — tensordict_to_transition_batch
# ------------------------------------------------------------------ #
class TestTensordictConversion:
    def _make_td(self, n=10):
        batch = TransitionBatch(
            obs=np.random.randn(n, *_OBS_SHAPE).astype(np.float32),
            act=np.random.randn(n, *_ACT_SHAPE).astype(np.float32),
            next_obs=np.random.randn(n, *_OBS_SHAPE).astype(np.float32),
            rewards=np.random.randn(n).astype(np.float32),
            terminateds=np.zeros(n, dtype=bool),
            truncateds=np.zeros(n, dtype=bool),
        )
        return transition_batch_to_tensordict(batch)

    def test_tensordict_to_batch_as_torch_true(self):
        td = self._make_td()
        batch = tensordict_to_transition_batch(td, as_torch=True)
        assert isinstance(batch.obs, torch.Tensor)
        assert isinstance(batch.rewards, torch.Tensor)

    def test_tensordict_to_batch_as_torch_false(self):
        td = self._make_td()
        batch = tensordict_to_transition_batch(td, as_torch=False)
        assert isinstance(batch.obs, np.ndarray)
        assert isinstance(batch.rewards, np.ndarray)

    def test_tensordict_to_batch_default_is_torch(self):
        td = self._make_td()
        batch = tensordict_to_transition_batch(td)
        assert isinstance(batch.obs, torch.Tensor)

    def test_tensordict_roundtrip_values(self):
        n = 5
        orig = TransitionBatch(
            obs=np.random.randn(n, *_OBS_SHAPE).astype(np.float32),
            act=np.random.randn(n, *_ACT_SHAPE).astype(np.float32),
            next_obs=np.random.randn(n, *_OBS_SHAPE).astype(np.float32),
            rewards=np.random.randn(n).astype(np.float32),
            terminateds=np.zeros(n, dtype=bool),
            truncateds=np.zeros(n, dtype=bool),
        )
        td = transition_batch_to_tensordict(orig)
        recovered = tensordict_to_transition_batch(td, as_torch=False)
        np.testing.assert_allclose(orig.obs, recovered.obs, atol=1e-6)
        np.testing.assert_allclose(orig.rewards, recovered.rewards, atol=1e-6)


# ------------------------------------------------------------------ #
#  4.3 — ReplayBuffer Output Mode
# ------------------------------------------------------------------ #
class TestReplayBufferOutputMode:
    def _make_buffer(self, n=20, **kwargs):
        buf = replay_buffer.ReplayBuffer(
            100, _OBS_SHAPE, _ACT_SHAPE, **kwargs
        )
        for i in range(n):
            buf.add(
                np.random.randn(*_OBS_SHAPE).astype(np.float32),
                np.random.randn(*_ACT_SHAPE).astype(np.float32),
                np.random.randn(*_OBS_SHAPE).astype(np.float32),
                float(i),
                False,
                False,
            )
        return buf

    def test_replay_buffer_get_all_returns_torch(self):
        buf = self._make_buffer()
        batch = buf.get_all()
        assert isinstance(batch.obs, torch.Tensor)
        assert isinstance(batch.rewards, torch.Tensor)

    def test_replay_buffer_sample_returns_torch(self):
        buf = self._make_buffer()
        batch = buf.sample(5)
        assert isinstance(batch.obs, torch.Tensor)

    def test_replay_buffer_batch_from_indices_returns_torch(self):
        buf = self._make_buffer()
        batch = buf._batch_from_indices(np.arange(5))
        assert isinstance(batch.obs, torch.Tensor)

    def test_replay_buffer_output_torch_false(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            buf = self._make_buffer(output_torch=False)
        batch = buf.get_all()
        assert isinstance(batch.obs, np.ndarray)

    def test_replay_buffer_output_torch_false_warning(self):
        with pytest.warns(FutureWarning, match="numpy arrays"):
            self._make_buffer(output_torch=False)

    def test_replay_buffer_add_accepts_torch(self):
        buf = replay_buffer.ReplayBuffer(100, _OBS_SHAPE, _ACT_SHAPE)
        buf.add(
            torch.randn(*_OBS_SHAPE),
            torch.randn(*_ACT_SHAPE),
            torch.randn(*_OBS_SHAPE),
            0.5,
            False,
            False,
        )
        assert buf.num_stored == 1

    def test_replay_buffer_add_batch_accepts_torch(self):
        buf = replay_buffer.ReplayBuffer(100, _OBS_SHAPE, _ACT_SHAPE)
        n = 5
        buf.add_batch(
            torch.randn(n, *_OBS_SHAPE),
            torch.randn(n, *_ACT_SHAPE),
            torch.randn(n, *_OBS_SHAPE),
            torch.randn(n),
            torch.zeros(n, dtype=torch.bool),
            torch.zeros(n, dtype=torch.bool),
        )
        assert buf.num_stored == n

    def test_replay_buffer_add_accepts_numpy(self):
        buf = replay_buffer.ReplayBuffer(100, _OBS_SHAPE, _ACT_SHAPE)
        buf.add(
            np.random.randn(*_OBS_SHAPE).astype(np.float32),
            np.random.randn(*_ACT_SHAPE).astype(np.float32),
            np.random.randn(*_OBS_SHAPE).astype(np.float32),
            0.5,
            False,
            False,
        )
        assert buf.num_stored == 1

    def test_replay_buffer_save_load_roundtrip(self):
        buf = self._make_buffer(n=10)
        orig = buf.get_all()
        with tempfile.TemporaryDirectory() as d:
            buf.save(d)
            buf2 = replay_buffer.ReplayBuffer(100, _OBS_SHAPE, _ACT_SHAPE)
            buf2.load(d)
            loaded = buf2.get_all()
        torch.testing.assert_close(orig.obs, loaded.obs)
        torch.testing.assert_close(orig.rewards, loaded.rewards)

    def test_replay_buffer_save_creates_pt_file(self):
        buf = self._make_buffer(n=5)
        with tempfile.TemporaryDirectory() as d:
            buf.save(d)
            assert (pathlib.Path(d) / "replay_buffer.pt").exists()
            assert not (pathlib.Path(d) / "replay_buffer.npz").exists()

    def test_replay_buffer_load_legacy_npz_fallback(self):
        buf = self._make_buffer(n=5)
        with tempfile.TemporaryDirectory() as d:
            # Manually save in legacy npz format
            batch = tensordict_to_transition_batch(
                buf._torchrl_rb.storage[:buf.num_stored], as_torch=False
            )
            np.savez(
                pathlib.Path(d) / "replay_buffer.npz",
                obs=batch.obs,
                next_obs=batch.next_obs,
                action=batch.act,
                reward=batch.rewards,
                terminated=batch.terminateds,
                truncated=batch.truncateds,
                num_stored=buf.num_stored,
                cur_idx=buf.cur_idx,
            )
            buf2 = replay_buffer.ReplayBuffer(100, _OBS_SHAPE, _ACT_SHAPE)
            with pytest.warns(FutureWarning, match="legacy numpy format"):
                buf2.load(d)
            assert buf2.num_stored == buf.num_stored

    def test_replay_buffer_load_missing_file_raises(self):
        buf = replay_buffer.ReplayBuffer(100, _OBS_SHAPE, _ACT_SHAPE)
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(FileNotFoundError):
                buf.load(d)

    def test_replay_buffer_save_load_with_trajectories(self):
        buf = replay_buffer.ReplayBuffer(
            100, _OBS_SHAPE, _ACT_SHAPE, max_trajectory_length=20
        )
        for i in range(15):
            terminated = (i + 1) % 5 == 0
            buf.add(
                np.random.randn(*_OBS_SHAPE).astype(np.float32),
                np.random.randn(*_ACT_SHAPE).astype(np.float32),
                np.random.randn(*_OBS_SHAPE).astype(np.float32),
                float(i),
                terminated,
                False,
            )
        orig_traj = list(buf.trajectory_indices)
        with tempfile.TemporaryDirectory() as d:
            buf.save(d)
            buf2 = replay_buffer.ReplayBuffer(
                100, _OBS_SHAPE, _ACT_SHAPE, max_trajectory_length=20
            )
            buf2.load(d)
        assert buf2.trajectory_indices == orig_traj


# ------------------------------------------------------------------ #
#  4.4 — model_util.to_tensor Fast Path
# ------------------------------------------------------------------ #
class TestToTensorFastPath:
    def test_to_tensor_torch_passthrough(self):
        t = torch.randn(5)
        result = model_util.to_tensor(t)
        assert result is t  # same object, no copy

    def test_to_tensor_numpy_converts(self):
        a = np.random.randn(5).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model_util.to_tensor(a)
        assert isinstance(result, torch.Tensor)

    def test_to_tensor_numpy_warning(self):
        a = np.random.randn(5).astype(np.float32)
        with pytest.warns(UserWarning, match="numpy array"):
            model_util.to_tensor(a)


# ------------------------------------------------------------------ #
#  4.5 — OneDTransitionRewardModel Torch Path
# ------------------------------------------------------------------ #
_DEVICE = "cpu"


class _SimpleMLP(mbrl.models.Model):
    def __init__(self, in_size, out_size):
        super().__init__(_DEVICE)
        self.in_size = in_size
        self.linear = torch.nn.Linear(in_size, out_size)

    def forward(self, x, **kwargs):
        return (self.linear(x),)

    def loss(self, model_in, target=None):
        return torch.tensor(0.0), {}

    def eval_score(self, model_in, target=None):
        return torch.zeros(1), {}

    def save(self, save_dir):
        pass

    def load(self, load_dir):
        pass


class TestOneDTransitionRewardModelTorchPath:
    def _make_wrapper(self, normalize=True):
        obs_dim, act_dim = _OBS_SHAPE[0], _ACT_SHAPE[0]
        model = _SimpleMLP(obs_dim + act_dim, obs_dim + 1)
        return mbrl.models.OneDTransitionRewardModel(
            model, target_is_delta=True, normalize=normalize
        )

    def test_update_normalizer_torch_input(self):
        wrapper = self._make_wrapper()
        batch = TransitionBatch(
            obs=torch.randn(20, *_OBS_SHAPE),
            act=torch.randn(20, *_ACT_SHAPE),
            next_obs=torch.randn(20, *_OBS_SHAPE),
            rewards=torch.randn(20),
            terminateds=torch.zeros(20, dtype=torch.bool),
            truncateds=torch.zeros(20, dtype=torch.bool),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should NOT emit numpy warning since input is torch
            wrapper.update_normalizer(batch)

    def test_update_normalizer_numpy_input(self):
        wrapper = self._make_wrapper()
        batch = TransitionBatch(
            obs=np.random.randn(20, *_OBS_SHAPE).astype(np.float32),
            act=np.random.randn(20, *_ACT_SHAPE).astype(np.float32),
            next_obs=np.random.randn(20, *_OBS_SHAPE).astype(np.float32),
            rewards=np.random.randn(20).astype(np.float32),
            terminateds=np.zeros(20, dtype=bool),
            truncateds=np.zeros(20, dtype=bool),
        )
        # Should still work (backward compat)
        wrapper.update_normalizer(batch)

    def test_get_model_input_torch(self):
        wrapper = self._make_wrapper(normalize=False)
        obs = torch.randn(5, *_OBS_SHAPE)
        act = torch.randn(5, *_ACT_SHAPE)
        result = wrapper._get_model_input(obs, act)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (5, _OBS_SHAPE[0] + _ACT_SHAPE[0])

    def test_process_batch_torch(self):
        wrapper = self._make_wrapper(normalize=False)
        batch = TransitionBatch(
            obs=torch.randn(5, *_OBS_SHAPE),
            act=torch.randn(5, *_ACT_SHAPE),
            next_obs=torch.randn(5, *_OBS_SHAPE),
            rewards=torch.randn(5),
            terminateds=torch.zeros(5, dtype=torch.bool),
            truncateds=torch.zeros(5, dtype=torch.bool),
        )
        model_in, target = wrapper._process_batch(batch)
        assert isinstance(model_in, torch.Tensor)
        assert isinstance(target, torch.Tensor)

    def test_ensure_tensor_helper(self):
        wrapper = self._make_wrapper(normalize=False)
        # torch input
        t = torch.randn(3, 4)
        result = wrapper._ensure_tensor(t)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == _DEVICE
        # numpy input
        a = np.random.randn(3, 4).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = wrapper._ensure_tensor(a)
        assert isinstance(result, torch.Tensor)


# ------------------------------------------------------------------ #
#  4.6 — ModelEnv Torch Path
# ------------------------------------------------------------------ #
class _DummyModel(mbrl.models.Model):
    def __init__(self):
        super().__init__(_DEVICE)
        self.param = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, **kwargs):
        obs = x[:, :_OBS_SHAPE[0]]
        return (torch.cat([obs, obs[:, :1]], dim=1),)

    def reset_1d(self, _obs, rng=None):
        return {}

    def sample_1d(self, x, _, deterministic=False, rng=None):
        return self.forward(x)[0], {}

    def loss(self, _input, target=None):
        return 0.0 * self.param, {}

    def eval_score(self, _input, target=None):
        return torch.zeros_like(_input), {}

    def set_elite(self, _indices):
        pass

    def save(self, d):
        pass

    def load(self, d):
        pass


class TestModelEnvTorchPath:
    def _make_model_env(self):
        import gymnasium as gym

        class MockEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(
                    -np.inf, np.inf, shape=_OBS_SHAPE
                )
                self.action_space = gym.spaces.Box(-1, 1, shape=_ACT_SHAPE)

            def reset(self, **kwargs):
                return np.zeros(_OBS_SHAPE, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(_OBS_SHAPE, dtype=np.float32), 0.0, False, False, {}

        from mbrl.env.termination_fns import no_termination

        model = _DummyModel()
        wrapper = mbrl.models.OneDTransitionRewardModel(
            model, target_is_delta=False
        )
        return mbrl.models.ModelEnv(
            MockEnv(), wrapper, no_termination, generator=torch.Generator()
        )

    def test_evaluate_action_sequences_no_numpy_tile(self):
        model_env = self._make_model_env()
        action_sequences = torch.randn(3, 2, *_ACT_SHAPE)
        returns = model_env.evaluate_action_sequences(
            action_sequences, np.zeros(_OBS_SHAPE), num_particles=2
        )
        assert isinstance(returns, torch.Tensor)
        assert returns.shape == (3,)

    def test_reset_accepts_torch(self):
        model_env = self._make_model_env()
        obs = torch.zeros(1, *_OBS_SHAPE)
        state = model_env.reset(obs, return_as_np=False)
        assert isinstance(state, dict)

    def test_reset_accepts_numpy(self):
        model_env = self._make_model_env()
        obs = np.zeros((1, *_OBS_SHAPE), dtype=np.float32)
        state = model_env.reset(obs, return_as_np=True)
        assert isinstance(state, dict)

    def test_step_returns_torch_when_configured(self):
        model_env = self._make_model_env()
        obs = np.zeros((1, *_OBS_SHAPE), dtype=np.float32)
        state = model_env.reset(obs, return_as_np=False)
        action = torch.zeros(1, *_ACT_SHAPE)
        next_obs, reward, done, _ = model_env.step(action, state)
        assert isinstance(next_obs, torch.Tensor)
