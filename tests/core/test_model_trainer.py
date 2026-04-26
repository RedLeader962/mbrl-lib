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


# ------------------------------------------------------------------ #
#  RLRP-606: Validate gradients are zeroed at the end of the legacy
#  callback's ``on_train_epoch_end`` hook.
#
#  Background
#  ----------
#  ``_LegacyCallback`` snapshots ``param.grad`` in ``on_before_zero_grad``
#  (Lightning zeros the optimizer's gradients after every training step) so
#  that the user-provided ``legacy_callback`` can inspect them at the end of
#  each epoch.  Restoring ``param.grad = param._last_grad`` leaves the model
#  parameters carrying a *stale* gradient tensor after the callback returns.
#  If those stale tensors leak into the next epoch (or into any other
#  optimizer-driven code path) they corrupt subsequent updates.
#
#  The fix in ``on_train_epoch_end`` calls
#  ``self.model_trainer.optimizer.zero_grad()`` after the legacy callback has
#  inspected the gradients, ensuring no stale values survive past the hook.
#
#  These tests verify that:
#    1. The legacy callback can still observe non-``None`` gradients during
#       its invocation (i.e. the snapshot/restore path is preserved).
#    2. After ``train()`` returns, every ``param.grad`` is either ``None`` or
#       a zero tensor (i.e. the fix actually zeros them back).
# ------------------------------------------------------------------ #
class _GradModel(mbrl.models.Model):
    """Model with a non-trivial loss that produces real, non-zero grads."""

    def __init__(self):
        super().__init__(torch.device(_DEVICE))
        # Non-zero initialization so ``loss = (param ** 2).mean()`` gives a
        # non-zero gradient on every backward pass.
        self.param = nn.Parameter(torch.full((4,), 0.5))

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
        # Differentiable, parameter-dependent loss → non-zero gradients.
        return (self.param ** 2).mean(), {"loss": 0}

    def eval_score(self, _input, target=None):
        return torch.zeros_like(_input), {"score": 0}

    def set_elite(self, _indices):
        pass


class TestLegacyCallbackZeroGradFix:
    """Regression tests for RLRP-606.

    Validate that ``_LegacyCallback.on_train_epoch_end`` zeros the trainer's
    optimizer gradients after invoking the user-provided legacy callback so
    that no stale gradient tensors survive past the hook.
    """

    def test_legacy_callback_observes_non_none_grad(self):
        """The legacy callback must still see ``param.grad is not None`` at
        epoch end (this is the contract that the snapshot/restore path
        protects).

        Expected outcome: ``observed_has_grad`` is ``True`` for every epoch.
        """
        model = _GradModel()
        wrapper = mbrl.models.OneDTransitionRewardModel(
            model, target_is_delta=False
        )
        trainer = mbrl.models.ModelTrainer(wrapper)

        observed_has_grad: list = []

        def legacy_cb(model_, _it, _epoch, _train_loss, _val_score, _best):
            # Inspect parameter gradients exactly as a real legacy callback
            # would (e.g. for gradient histograms).
            grads_present = [
                p.grad is not None
                for p in model_.parameters()
                if p.requires_grad
            ]
            observed_has_grad.append(all(grads_present) and len(grads_present) > 0)

        ds_train = _make_dataset(8)
        ds_val = _make_dataset(8)
        trainer.train(
            ds_train,
            dataset_val=ds_val,
            num_epochs=3,
            evaluate=True,
            silent=True,
            callback=legacy_cb,
        )

        assert len(observed_has_grad) == 3, (
            f"Expected 3 callback invocations, got {len(observed_has_grad)}"
        )
        assert all(observed_has_grad), (
            "Legacy callback must observe non-None param.grad each epoch; "
            f"got {observed_has_grad}"
        )

    def test_param_grad_zeroed_after_training(self):
        """After ``train()`` returns, every parameter gradient must be either
        ``None`` or a zero tensor.

        Without the fix, ``param.grad`` would still hold the (non-zero)
        snapshot restored in ``on_train_epoch_end``.

        Expected outcome: every trainable parameter's gradient is zero/None.
        """
        model = _GradModel()
        wrapper = mbrl.models.OneDTransitionRewardModel(
            model, target_is_delta=False
        )
        trainer = mbrl.models.ModelTrainer(wrapper)

        ds_train = _make_dataset(8)
        ds_val = _make_dataset(8)
        trainer.train(
            ds_train,
            dataset_val=ds_val,
            num_epochs=2,
            evaluate=True,
            silent=True,
        )

        for p in model.parameters():
            if not p.requires_grad:
                continue
            assert (p.grad is None) or torch.all(p.grad == 0.0), (
                "param.grad must be zeroed after train() (RLRP-606); "
                f"found non-zero grad: {p.grad}"
            )

    def test_param_grad_zeroed_across_multiple_train_calls(self):
        """The grad-zeroing invariant must hold across successive
        ``train()`` calls (no stale grads leaking between calls).

        Expected outcome: every parameter has a zero/None grad after each call.
        """
        model = _GradModel()
        wrapper = mbrl.models.OneDTransitionRewardModel(
            model, target_is_delta=False
        )
        trainer = mbrl.models.ModelTrainer(wrapper)

        for call_idx in range(3):
            ds_train = _make_dataset(8)
            ds_val = _make_dataset(8)
            trainer.train(
                ds_train,
                dataset_val=ds_val,
                num_epochs=2,
                evaluate=True,
                silent=True,
            )
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                assert (p.grad is None) or torch.all(p.grad == 0.0), (
                    f"Call {call_idx}: param.grad must be zeroed after "
                    f"train() (RLRP-606); found {p.grad}"
                )
