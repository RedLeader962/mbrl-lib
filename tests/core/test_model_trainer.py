# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for ModelTrainer state isolation and bootstrap toggle safety.

These tests verify that:
- Calling ``ModelTrainer.train()`` multiple times produces correct validation
  losses each time (no stale state leaks between calls).
- The bootstrap toggle is safely restored even when validation doesn't
  complete cleanly.
"""
import torch
import torch.nn as nn

import mbrl.models
import mbrl.util.replay_buffer
from mbrl.types import TransitionBatch
from mbrl.util.replay_buffer import BootstrapIterator

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
_OBS_DIM = 1
_ACT_DIM = 1


class _DummyModel(mbrl.models.Model):
    """Minimal model for trainer tests that returns a trivial loss."""

    def __init__(self):
        super().__init__(torch.device(_DEVICE))
        self.param = nn.Parameter(torch.ones(1))

    def forward(self, x, **kwargs):
        obs = x[:, :_OBS_DIM]
        act = x[:, _OBS_DIM:]
        new_obs = obs + act.mean(axis=1, keepdim=True)
        return torch.cat([new_obs, new_obs], dim=1)

    def reset_1d(self, _obs, rng=None):
        return {}

    def sample_1d(self, x, _, deterministic=False, rng=None):
        return self.forward(x), {}

    def loss(self, _input, target=None):
        return 0.0 * self.param, {"loss": 0}

    def eval_score(self, _input, target=None):
        return torch.zeros_like(_input), {"score": 0}

    def set_elite(self, _indices):
        pass


def _make_dataset(num_samples):
    """Create a TransitionIterator with *num_samples* dummy transitions."""
    data = torch.zeros(num_samples, _OBS_DIM)
    return mbrl.util.replay_buffer.TransitionIterator(
        TransitionBatch(
            data,
            data,
            data,
            data.squeeze(1),
            data.squeeze(1),
            data.squeeze(1),
        ),
        batch_size=1,
    )


def _make_bootstrap_dataset(num_samples, ensemble_size=2):
    """Create a BootstrapIterator with *num_samples* dummy transitions."""
    data = torch.zeros(num_samples, _OBS_DIM)
    return BootstrapIterator(
        TransitionBatch(
            data,
            data,
            data,
            data.squeeze(1),
            data.squeeze(1),
            data.squeeze(1),
        ),
        batch_size=1,
        ensemble_size=ensemble_size,
    )


# ------------------------------------------------------------------ #
#  Step 4.1: Regression test for trainer state isolation
# ------------------------------------------------------------------ #
class TestModelTrainerStateIsolation:
    """Verify that calling ``ModelTrainer.train()`` multiple times produces
    correct and consistent validation losses without stale state leaks.
    """

    def test_multiple_train_calls_return_consistent_val_losses(self):
        """Call train() three times with identical data and verify that
        validation loss lists have the expected length each time.

        Expected outcome: each call returns *num_epochs* val loss entries and
        the values are consistent (no corruption from prior calls).
        """
        model = _DummyModel()
        wrapper = mbrl.models.OneDTransitionRewardModel(
            model, target_is_delta=False
        )
        trainer = mbrl.models.ModelTrainer(wrapper)

        num_epochs = 5
        for call_idx in range(3):
            ds_train = _make_dataset(10)
            ds_val = _make_dataset(10)
            train_losses, val_losses = trainer.train(
                ds_train,
                dataset_val=ds_val,
                num_epochs=num_epochs,
                evaluate=True,
                silent=True,
            )
            assert len(train_losses) == num_epochs, (
                f"Call {call_idx}: expected {num_epochs} train losses, "
                f"got {len(train_losses)}"
            )
            assert len(val_losses) == num_epochs, (
                f"Call {call_idx}: expected {num_epochs} val losses, "
                f"got {len(val_losses)}"
            )

    def test_different_dataset_sizes_across_calls(self):
        """Train with datasets of varying sizes across successive calls to
        ensure no stale dataloader references affect batch processing.

        Expected outcome: each call completes successfully and returns the
        correct number of loss entries regardless of prior dataset sizes.
        """
        model = _DummyModel()
        wrapper = mbrl.models.OneDTransitionRewardModel(
            model, target_is_delta=False
        )
        trainer = mbrl.models.ModelTrainer(wrapper)

        num_epochs = 3
        for size in [5, 20, 8]:
            ds_train = _make_dataset(size)
            ds_val = _make_dataset(size)
            train_losses, val_losses = trainer.train(
                ds_train,
                dataset_val=ds_val,
                num_epochs=num_epochs,
                evaluate=True,
                silent=True,
            )
            assert len(train_losses) == num_epochs
            assert len(val_losses) == num_epochs

    def test_val_losses_are_deterministic_across_calls(self):
        """Two consecutive train() calls with identical data and reset model
        should produce identical val losses (no stale state influence).

        Expected outcome: val losses from call 1 equal val losses from call 2.
        """
        torch.manual_seed(42)
        model = _DummyModel()
        wrapper = mbrl.models.OneDTransitionRewardModel(
            model, target_is_delta=False
        )
        trainer = mbrl.models.ModelTrainer(wrapper)

        num_epochs = 3
        ds_train = _make_dataset(10)
        ds_val = _make_dataset(10)

        _, val_losses_1 = trainer.train(
            ds_train,
            dataset_val=ds_val,
            num_epochs=num_epochs,
            evaluate=True,
            silent=True,
        )
        _, val_losses_2 = trainer.train(
            ds_train,
            dataset_val=ds_val,
            num_epochs=num_epochs,
            evaluate=True,
            silent=True,
        )
        assert val_losses_1 == val_losses_2, (
            f"Val losses differ between identical calls: "
            f"{val_losses_1} vs {val_losses_2}"
        )


# ------------------------------------------------------------------ #
#  Step 4.2: Bootstrap toggle safety test
# ------------------------------------------------------------------ #
class TestBootstrapToggleSafety:
    """Verify bootstrap iterator state is correctly restored after
    validation, including edge cases.
    """

    def test_bootstrap_restored_after_normal_training(self):
        """After a normal train() call with a BootstrapIterator as
        validation data, the bootstrap flag should be restored to its
        original state (enabled).

        Expected outcome: ``_bootstrap_iter`` is True before and after
        the train() call.
        """
        model = _DummyModel()
        wrapper = mbrl.models.OneDTransitionRewardModel(
            model, target_is_delta=False
        )
        trainer = mbrl.models.ModelTrainer(wrapper)

        ds_train = _make_dataset(10)
        ds_val = _make_bootstrap_dataset(10, ensemble_size=2)

        assert ds_val._bootstrap_iter is True, (
            "Bootstrap should be enabled before training"
        )

        trainer.train(
            ds_train,
            dataset_val=ds_val,
            num_epochs=3,
            evaluate=True,
            silent=True,
        )

        assert ds_val._bootstrap_iter is True, (
            "Bootstrap should be restored after training completes"
        )

    def test_bootstrap_restored_after_multiple_train_calls(self):
        """After multiple successive train() calls the bootstrap flag
        should remain in its original state each time.

        Expected outcome: ``_bootstrap_iter`` is True after every call.
        """
        model = _DummyModel()
        wrapper = mbrl.models.OneDTransitionRewardModel(
            model, target_is_delta=False
        )
        trainer = mbrl.models.ModelTrainer(wrapper)

        ds_train = _make_dataset(10)
        ds_val = _make_bootstrap_dataset(10, ensemble_size=2)

        for call_idx in range(3):
            trainer.train(
                ds_train,
                dataset_val=ds_val,
                num_epochs=2,
                evaluate=True,
                silent=True,
            )
            assert ds_val._bootstrap_iter is True, (
                f"Call {call_idx}: bootstrap should be restored after training"
            )

    def test_bootstrap_restored_with_early_stopping(self):
        """When early stopping triggers during training, the bootstrap
        flag should still be correctly restored.

        Expected outcome: ``_bootstrap_iter`` is True after early stopping.
        """
        model = _DummyModel()
        wrapper = mbrl.models.OneDTransitionRewardModel(
            model, target_is_delta=False
        )
        trainer = mbrl.models.ModelTrainer(wrapper)

        ds_train = _make_dataset(10)
        ds_val = _make_bootstrap_dataset(10, ensemble_size=2)

        trainer.train(
            ds_train,
            dataset_val=ds_val,
            num_epochs=100,
            patience=2,
            evaluate=True,
            silent=True,
        )

        assert ds_val._bootstrap_iter is True, (
            "Bootstrap should be restored after early stopping"
        )
