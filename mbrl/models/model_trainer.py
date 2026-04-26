# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
import sys
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import optim as optim
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.callbacks import EarlyStopping, RichModelSummary
from pytorch_lightning.utilities.model_summary import summarize as pl_summarize
import pytorch_lightning as pl


from mbrl.util.logger import Logger
from mbrl.util.replay_buffer import BootstrapIterator, TransitionIterator

from .model import Ensemble, Model
from .one_dim_tr_model import OneDTransitionRewardModel

MODEL_LOG_FORMAT = [
    ("train_iteration", "I", "int"),
    ("epoch", "E", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
]


class _IteratorDataset(IterableDataset):
    def __init__(self, it: TransitionIterator):
        self.it = it

    def __iter__(self):
        return iter(self.it)

    def __len__(self):
        return len(self.it)


class _LegacyCallback(pl.Callback):
    """Bridge between Lightning training events and the legacy callback system.

    This callback collects per-epoch training losses and validation scores,
    invokes the user-provided ``legacy_callback`` and ``legacy_batch_callback`` at the
    appropriate times, and logs metrics via the mbrl :class:`Logger`.

    It also implements the legacy best-weight tracking logic using
    :meth:`ModelTrainer.maybe_get_best_weights`, storing both the best
    validation score (per ensemble member) and the corresponding model weights
    in memory so that :meth:`ModelTrainer.train` can restore them after
    training without relying on ``ModelCheckpoint``.
    """

    def __init__(
        self,
        model_trainer: "ModelTrainer",
        legacy_callback,
        batch_callback,
        logger=None,
        evaluate: bool = True,
        improvement_threshold: float = 0.01,
    ):
        self.model_trainer = model_trainer
        self.legacy_callback = legacy_callback
        self.legacy_batch_callback = batch_callback
        self.logger = logger
        self.evaluate = evaluate
        self.improvement_threshold = improvement_threshold

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        # Best-weight tracking (replaces ModelCheckpoint)
        self.best_val_score: Optional[torch.Tensor] = None
        self.current_epoch_val_score: Optional[torch.Tensor] = None
        self.best_weights: Optional[Dict] = None

    # ------------------------------------------------------------------ #
    #  Validation score tracking (per-member, for ensembles)
    # ------------------------------------------------------------------ #
    def on_validation_epoch_start(self, trainer, pl_module):
        self._epoch_val_scores: List[torch.Tensor] = []
        self._bootstrap_was_toggled = False
        # Toggle bootstrap off for validation, matching legacy evaluate() behavior
        val_dl = trainer.val_dataloaders
        if val_dl is not None:
            ds = val_dl.dataset if hasattr(val_dl, "dataset") else None
            if (
                ds is not None
                and hasattr(ds, "it")
                and isinstance(ds.it, BootstrapIterator)
            ):
                ds.it.toggle_bootstrap()
                self._bootstrap_was_toggled = True

        return None

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # Collect raw (non-reduced) val scores for per-member tracking
        val_score = None
        if isinstance(outputs, dict):
            val_score = outputs.get("val_score")

        if self.legacy_batch_callback:
            score = outputs
            meta = {}
            if isinstance(outputs, dict):
                score = outputs.get("score", outputs)
                meta = {
                    k: v for k, v in outputs.items() if k not in ["score", "val_score"]
                }

            if isinstance(score, torch.Tensor):
                score = score.detach().cpu().numpy()

            meta["score"] = score
            self.legacy_batch_callback(trainer.current_epoch, score, meta, "eval")

        if val_score is not None:
            self._epoch_val_scores.append(val_score.detach().cpu())

        return None

    def on_validation_epoch_end(self, trainer, pl_module):
        # Always restore bootstrap if it was toggled, regardless of scores
        if self._bootstrap_was_toggled:
            val_dl = trainer.val_dataloaders
            if val_dl is not None:
                ds = val_dl.dataset if hasattr(val_dl, "dataset") else None
                if (
                    ds is not None
                    and hasattr(ds, "it")
                    and isinstance(ds.it, BootstrapIterator)
                ):
                    ds.it.toggle_bootstrap()
            self._bootstrap_was_toggled = False

        if not self._epoch_val_scores:
            return

        first = self._epoch_val_scores[0]
        if first.ndim == 3:  # Ensemble (E, B, Od)
            all_scores = torch.cat(self._epoch_val_scores, dim=1)
            # Average over batch (1) and output dim (2) to get (E,)
            epoch_avg_scores = all_scores.mean(dim=(1, 2))
        elif first.ndim == 2:  # Non-ensemble (B, Od)
            all_scores = torch.cat(self._epoch_val_scores, dim=0)
            epoch_avg_scores = all_scores.mean(
                dim=0 if all_scores.ndim == 1 else (0, 1)
            ).unsqueeze(0)
        else:
            epoch_avg_scores = torch.stack(self._epoch_val_scores).mean().unsqueeze(0)

        # Initialize best_val_score on first epoch
        if self.best_val_score is None:
            self.best_val_score = epoch_avg_scores

        # Use the legacy relative-improvement logic
        maybe_best = self.model_trainer.maybe_get_best_weights(
            self.best_val_score, epoch_avg_scores, self.improvement_threshold
        )
        if maybe_best is not None:
            self.best_val_score = torch.minimum(self.best_val_score, epoch_avg_scores)
            self.best_weights = maybe_best

        self.current_epoch_val_score = epoch_avg_scores
        self._epoch_val_scores = []

        return None

    # ------------------------------------------------------------------ #
    #  Training batch callback
    # ------------------------------------------------------------------ #
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.legacy_batch_callback:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            meta = (
                {k: v for k, v in outputs.items() if k != "loss"}
                if isinstance(outputs, dict)
                else {}
            )

            if isinstance(loss, torch.Tensor):
                loss = loss.detach().cpu().item()

            meta["loss"] = loss
            self.legacy_batch_callback(trainer.current_epoch, loss, meta, "train")

        return None

    # ------------------------------------------------------------------ #
    #  Gradient snapshot – capture before Lightning zeros them
    # ------------------------------------------------------------------ #
    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        """Store a snapshot of the current gradients.

        Lightning calls ``optimizer.zero_grad()`` after each training step,
        which sets ``param.grad`` to ``None``.  The legacy callback (invoked
        at epoch end) expects gradients to still be available for monitoring.
        We therefore clone them here so they survive the zero-grad call.

        Uses in-place copy to reuse the same buffer across steps,
        eliminating repeated memory allocation overhead.
        """
        for param in pl_module.parameters():
            if param.requires_grad and param.grad is not None:
                if hasattr(param, "_last_grad") and param._last_grad is not None:
                    param._last_grad.copy_(param.grad)  # In-place, no allocation
                else:
                    param._last_grad = param.grad.clone()  # First-time allocation only

        return None

    # ------------------------------------------------------------------ #
    #  End-of-epoch logging and legacy callback
    # ------------------------------------------------------------------ #
    def on_train_epoch_end(self, trainer, pl_module):
        # Restore the last-seen gradients so the legacy callback can inspect
        # ``param.grad`` (e.g. for gradient monitoring / histograms).
        for param in pl_module.parameters():
            if param.requires_grad and hasattr(param, "_last_grad"):
                param.grad = param._last_grad
        metrics = trainer.callback_metrics
        train_loss = metrics.get(
            "train/loss", metrics.get("train_loss", torch.tensor(0.0))
        ).item()
        val_loss = metrics.get(
            "val/loss", metrics.get("val_loss", torch.tensor(0.0))
        ).item()

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        eval_score = (
            self.current_epoch_val_score
            if self.current_epoch_val_score is not None
            else self.best_val_score
        )
        best_val_score = self.best_val_score

        if self.legacy_callback:
            self.legacy_callback(
                pl_module,
                self.model_trainer._train_iteration,
                trainer.current_epoch,
                train_loss,
                eval_score,
                best_val_score,
            )

        if self.logger:
            log_dict = {
                "train_iteration": self.model_trainer._train_iteration,
                "epoch": trainer.current_epoch,
                "train_dataset_size": len(trainer.train_dataloader.dataset.it),
                "val_dataset_size": (
                    len(trainer.val_dataloaders.dataset.it)
                    if trainer.val_dataloaders
                    else 0
                ),
                "model_loss": train_loss,
                "model_val_score": val_loss,
                "model_best_val_score": (
                    best_val_score.mean().item()
                    if best_val_score is not None
                    else val_loss
                ),
            }

            for k, v in metrics.items():
                if k not in ["train/loss", "val/loss", "train_loss", "val_loss"]:
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    log_dict[k] = v

            self.logger.log_data("model_train", log_dict)
            self.logger._dump("model_train")

        # Safety net: force-restore bootstrap if validation didn't clean up
        if hasattr(self, "_bootstrap_was_toggled") and self._bootstrap_was_toggled:
            val_dl = trainer.val_dataloaders
            if val_dl is not None:
                ds = val_dl.dataset if hasattr(val_dl, "dataset") else None
                if (
                    ds is not None
                    and hasattr(ds, "it")
                    and isinstance(ds.it, BootstrapIterator)
                ):
                    ds.it.toggle_bootstrap()
            self._bootstrap_was_toggled = False

        return None


class ModelTrainer:
    """Trainer for dynamics models.

    Args:
        model (:class:`mbrl.models.Model`): a model to train.
        optim_lr (float): the learning rate for the optimizer (using Adam).
        weight_decay (float): the weight decay to use.
        optim_eps (float): the epsilon for the optimizer.
        logger (:class:`mbrl.util.Logger`, optional): the logger to use.
    """

    _LOG_GROUP_NAME = "model_train"

    def __init__(
        self,
        model: Model,
        optim_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        optim_eps: float = 1e-8,
        logger: Optional[Logger] = None,
        use_preallocated_best_weights_buffer: bool = False,
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: bool = False,
        dataloader_persistent_workers: bool = False,
        dataloader_prefetch_factor: Optional[int] = None,
    ):
        """
        ``use_preallocated_best_weights_buffer`` (opt-in, default False):
            When True, :meth:`maybe_get_best_weights` stops doing
            ``copy.deepcopy(self.model.state_dict())`` on every improved
            inner epoch and instead keeps a single preallocated buffer
            (a clone of the model's ``state_dict`` on first improvement)
            that is updated in place via
            ``tensor.copy_(..., non_blocking=True)``. The buffer is
            returned by reference so the downstream ``load_state_dict``
            call path is unchanged.
            Introduced by the RLRC Training Speed & Efficiency plan
            (stage 1, Batch 2 — action B0-bis). Default False keeps
            the legacy ``copy.deepcopy`` path bit-exact.

        ``dataloader_num_workers`` / ``dataloader_pin_memory`` /
        ``dataloader_persistent_workers`` / ``dataloader_prefetch_factor``
        (opt-in, default legacy: ``0 / False / False / None``):
            Wiring for :class:`torch.utils.data.DataLoader` knobs on the
            train and validation loaders built inside :meth:`train`.
            Introduced by the RLRC Training Speed & Efficiency stage-1
            follow-up plan (action F-C2). ``pin_memory`` is auto-gated
            on CUDA only — it is silently forced to False on CPU / MPS
            devices. ``persistent_workers`` and ``prefetch_factor`` are
            only passed through when ``num_workers > 0`` (PyTorch
            raises otherwise). Legacy defaults preserve the historical
            behaviour bit-exact (single-process data loading, no
            pinned host memory, no prefetch).
        """
        self.model = model
        self._train_iteration = 0
        # Training Speed & Efficiency plan (B0-bis): opt-in
        # preallocated best-weights buffer. Legacy default False →
        # ``copy.deepcopy(state_dict)`` behaviour unchanged.
        self._use_preallocated_best_weights_buffer = bool(
            use_preallocated_best_weights_buffer
        )
        # Lazily populated on first improvement when the opt-in flag
        # is True. ``None`` both in the legacy path and before the
        # first improvement in the new path.
        self._best_weights_buffer: Optional[Dict[str, torch.Tensor]] = None
        # Training Speed & Efficiency stage-1 follow-up plan (F-C2):
        # opt-in DataLoader knobs. Stored on the instance so
        # :meth:`train` can thread them into both DataLoader ctors
        # without re-reading the cfg. ``pin_memory`` is auto-gated on
        # CUDA at :meth:`train` time (model device may not be set at
        # ``__init__`` time for lazily-built models).
        self._dataloader_num_workers: int = int(dataloader_num_workers)
        self._dataloader_pin_memory: bool = bool(dataloader_pin_memory)
        self._dataloader_persistent_workers: bool = bool(dataloader_persistent_workers)
        self._dataloader_prefetch_factor: Optional[int] = (
            int(dataloader_prefetch_factor)
            if dataloader_prefetch_factor is not None
            else None
        )

        self.logger = logger
        if self.logger:
            self.logger.register_group(
                self._LOG_GROUP_NAME,
                MODEL_LOG_FORMAT,
                color="blue",
                dump_frequency=1,
            )

        self.optim_lr = optim_lr
        self.weight_decay = weight_decay
        self.optim_eps = optim_eps

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.optim_lr,
            weight_decay=self.weight_decay,
            eps=self.optim_eps,
        )

        # Monkey-patch ``configure_optimizers`` on the model instance so that
        # Lightning reuses the same optimizer instance.  This allows users to
        # attach LR schedulers to ``self.optimizer`` before calling ``train()``.
        def configure_optimizers():
            return self.optimizer

        self.model.configure_optimizers = configure_optimizers

        # Determine accelerator once (device type does not change between
        # ``train()`` calls) so we avoid re-computing it every iteration.
        if self.model.device.type == "cuda":
            self._accelerator = "gpu"
        elif self.model.device.type == "mps":
            self._accelerator = "mps"
        else:
            self._accelerator = "cpu"

        # Eagerly initialize CUDA and flush streams so that subsequent
        # ``pl.Trainer`` constructions in ``train()`` do not trigger
        # repeated diagnostics or lazy-init overhead.
        if torch.cuda.is_available():
            torch.cuda.init()
        sys.stderr.flush()
        sys.stdout.flush()

        self._model_summary_printed = False

    def print_model_summary_once(self):
        """Print the Lightning model summary table to stdout.

        The summary is printed only on the first call; subsequent calls are
        no-ops.  Call this **before** creating any external progress bar so
        that the table appears above the bar rather than interleaved with it.
        """
        if not self._model_summary_printed:
            model = self.model
            # print(model)
            if hasattr(model, "model"):
                # Show the model that learns something instead of the OneDTransitionRewardModel wrapper
                model = self.model.model

            if hasattr(model, "description"):
                print(model.description)

            summary = pl_summarize(model, max_depth=1)
            RichModelSummary.summarize(
                summary_data=summary._get_summary_data(),
                total_parameters=summary.total_parameters,
                trainable_parameters=summary.trainable_parameters,
                model_size=summary.model_size,
                total_training_modes=summary.total_training_modes,
                total_flops=getattr(summary, "total_flops", 0),
            )
            print()  # blank line after the summary
            self._model_summary_printed = True

    def train(
        self,
        dataset_train: TransitionIterator,
        dataset_val: Optional[TransitionIterator] = None,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        improvement_threshold: float = 0.01,
        callback: Optional[Callable] = None,
        batch_callback: Optional[Callable] = None,
        evaluate: bool = True,
        silent: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Trains the model for some number of epochs.

        This method iterates over the stored train dataset, one batch of
        transitions at a time, updates the model using
        ``pytorch_lightning.Trainer``.

        If a validation dataset is provided, this method will also evaluate
        the model over the validation data once per training epoch. The method
        will keep track of the weights with the best validation score, and
        after training the weights of the model will be set to the best
        weights. If no validation dataset is provided, the method will keep
        the model with the best loss over training data.

        Args:
            dataset_train (:class:`mbrl.util.TransitionIterator`): the iterator
                to use for the training data.
            dataset_val (:class:`mbrl.util.TransitionIterator`, optional):
                an iterator to use for the validation data.
            num_epochs (int, optional): if provided, the maximum number of
                epochs to train for. Default is ``None``, which indicates
                there is no limit.
            patience (int, optional): if provided, the patience to use for
                training. That is, training will stop after ``patience``
                number of epochs without improvement.
                Ignored if ``evaluate=False``.
            improvement_threshold (float): The threshold in relative decrease
                of the evaluation score at which the model is seen as having
                improved. Ignored if ``evaluate=False``.
            callback (callable, optional): if provided, this function will be
                called after every training epoch with the following positional
                arguments::

                    - the model that's being trained
                    - total number of calls made to ``trainer.train()``
                    - current epoch
                    - training loss
                    - validation score (for ensembles, factored per member)
                    - best validation score so far

            batch_callback (callable, optional): if provided, this function
                will be called for every batch with the output of
                ``model.loss()`` (during training) and
                ``model.eval_score()`` (during evaluation). It will be called
                with four arguments
                ``(epoch_index, loss/score, meta, mode)``, where ``mode`` is
                one of ``"train"`` or ``"eval"``.
            evaluate (bool, optional): if ``True``, the trainer will use
                ``model.eval_score()`` to keep track of the best model.
                Defaults to ``True``.
            silent (bool): if ``True`` logging and progress bar are
                deactivated. Defaults to ``False``.

        Returns:
            (tuple of two list(float)): the history of training losses and
                validation losses.
        """
        self._train_iteration += 1

        if not silent:
            self.print_model_summary_once()

        # Ensure model is in train mode before fitting.  A previous
        # ``pl.Trainer.fit()`` call may leave the model in eval mode after
        # its validation loop, and creating a new Trainer does not restore it.
        self.model.train()

        # Bridge TransitionIterator to Lightning DataLoader.
        # F-C2 (stage-1 follow-up): resolve DataLoader knobs with the
        # legacy defaults preserved bit-exact when the flags are left
        # at their defaults. ``pin_memory`` auto-downgrades on any
        # non-CUDA device (CPU / MPS) so a cfg flag accidentally left
        # on does not crash or warn on those runners.
        _num_workers = self._dataloader_num_workers
        _pin_memory = self._dataloader_pin_memory and self.model.device.type == "cuda"
        _dl_kwargs = {
            "batch_size": None,
            "num_workers": _num_workers,
            "pin_memory": _pin_memory,
        }
        if _num_workers > 0:
            # ``persistent_workers`` / ``prefetch_factor`` are only
            # valid when ``num_workers > 0``; PyTorch raises otherwise.
            if self._dataloader_persistent_workers:
                _dl_kwargs["persistent_workers"] = True
            if self._dataloader_prefetch_factor is not None:
                _dl_kwargs["prefetch_factor"] = self._dataloader_prefetch_factor
        train_loader = DataLoader(_IteratorDataset(dataset_train), **_dl_kwargs)
        val_loader = None
        if evaluate:
            eval_dataset = dataset_train if dataset_val is None else dataset_val
            val_loader = DataLoader(_IteratorDataset(eval_dataset), **_dl_kwargs)

        # Lightning Callbacks
        callbacks = []
        if evaluate and dataset_val and patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val/loss",
                    patience=patience,
                    min_delta=0.0,
                    mode="min",
                    check_on_train_epoch_end=False,
                )
            )

        legacy_cb = _LegacyCallback(
            model_trainer=self,
            legacy_callback=callback,
            batch_callback=batch_callback,
            logger=self.logger,
            evaluate=evaluate,
            improvement_threshold=improvement_threshold,
        )
        callbacks.append(legacy_cb)

        max_epochs = num_epochs if num_epochs is not None else 1000

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*GPU available but not used.*")
            warnings.filterwarnings("ignore", ".*does not have many workers.*")
            warnings.filterwarnings("ignore", ".*IterableDataset.*__len__.*")

            # Suppress Lightning's "Trainer.fit stopped: max_epochs=N
            # reached." info message that clutters ERLL progress output.
            _pl_logger = logging.getLogger("pytorch_lightning")
            _prev_level = _pl_logger.level
            _pl_logger.setLevel(logging.WARNING)
            try:
                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    callbacks=callbacks,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    devices="auto",
                    accelerator=self._accelerator,
                    logger=False,
                    num_sanity_val_steps=0,
                    enable_checkpointing=False,
                )
                trainer.fit(self.model, train_loader, val_loader)
            finally:
                _pl_logger.setLevel(_prev_level)

        # Restore best weights and select elite models
        if evaluate:
            self._maybe_set_best_weights_and_elite(
                legacy_cb.best_weights, legacy_cb.best_val_score
            )

        return legacy_cb.train_losses, legacy_cb.val_losses

    def evaluate(
        self, dataset: TransitionIterator, batch_callback: Optional[Callable] = None
    ) -> torch.Tensor:
        # (CRITICAL) ToDo: assess deprecating >> the validate method is not used anymore.
        """Evaluates the model on the validation dataset.

        Iterates over the dataset, one batch at a time, and calls
        :meth:`mbrl.models.Model.eval_score` to compute the model score
        over the batch. The method returns the average score over the whole
        dataset.

        Args:
            dataset (bool): the transition iterator to use.
            batch_callback (callable, optional): if provided, this function
                will be called for every batch with the output of
                ``model.eval_score()`` (the score will be passed as a float,
                reduced using mean()). It will be called with four arguments
                ``(epoch_index, loss/score, meta, mode)``, where ``mode`` is
                the string ``"eval"``.

        Returns:
            (tensor): The average score of the model over the dataset (and for
                ensembles, per ensemble member).
        """
        if isinstance(dataset, BootstrapIterator):
            dataset.toggle_bootstrap()

        batch_scores_list = []
        for batch in dataset:
            batch_score, meta = self.model.eval_score(batch)
            batch_scores_list.append(batch_score)
            if batch_callback:
                # (CRITICAL) ToDo: validate it is suppose to be called with four argument. Missing `epoch_index`!
                batch_callback(batch_score.mean(), meta, "eval")
        try:
            batch_scores = torch.cat(
                batch_scores_list, dim=batch_scores_list[0].ndim - 2
            )
        except RuntimeError as e:
            print(
                f"There was an error calling ModelTrainer.evaluate(). "
                f"Note that model.eval_score() should be non-reduced. "
                f"Error was: {e}"
            )
            raise e
        if isinstance(dataset, BootstrapIterator):
            dataset.toggle_bootstrap()

        mean_axis = 1 if batch_scores.ndim == 2 else (1, 2)
        batch_scores = batch_scores.mean(dim=mean_axis)

        return batch_scores

    def maybe_get_best_weights(
        self,
        best_val_score: torch.Tensor,
        val_score: torch.Tensor,
        threshold: float = 0.01,
    ) -> Optional[Dict]:
        """Return the current model state dict if the validation score improves.

        For ensembles, this checks the validation for each ensemble member
        separately.

        Args:
            best_val_score (tensor): the current best validation losses per
                model.
            val_score (tensor): the new validation loss per model.
            threshold (float): the threshold for relative improvement.

        Returns:
            (dict, optional): if the validation score's relative improvement
                over the best validation score is higher than the threshold,
                returns the state dictionary of the stored model, otherwise
                returns ``None``.
        """
        improvement = (best_val_score - val_score) / torch.abs(best_val_score)
        improved = (improvement > threshold).any().item()
        if not improved:
            return None

        # Legacy path (default; bit-exact with pre-B0-bis behaviour):
        # a brand-new deepcopy of the full state_dict is returned on
        # every improved inner epoch. This is O(parameters) allocations
        # + D→H copies on CUDA and is the hot-spot the B0-bis opt-in
        # removes when turned on explicitly.
        if not self._use_preallocated_best_weights_buffer:
            return copy.deepcopy(self.model.state_dict())

        # Opt-in path (Training Speed & Efficiency plan — B0-bis):
        # keep a single preallocated clone of ``state_dict`` and
        # update it in place via ``tensor.copy_(...)``. Bit-exact by
        # construction since ``tensor.copy_`` is a value-for-value
        # assignment and ``load_state_dict`` downstream still sees the
        # same numeric values. The dict object is stable across calls
        # so the callback's ``best_weights = maybe_best`` semantics
        # remain unchanged.
        current_state_dict = self.model.state_dict()
        if self._best_weights_buffer is None:
            # First improvement: clone once so the buffer is
            # independent from the live model parameters. ``detach``
            # avoids any autograd history leakage.
            self._best_weights_buffer = {
                k: v.detach().clone() for k, v in current_state_dict.items()
            }
        else:
            # Subsequent improvements: overwrite the buffer in place.
            # ``non_blocking=True`` is a no-op on CPU but lets CUDA
            # D↔D copies overlap with the next inner-epoch forward
            # when the buffer lives on-device.
            for name, src_tensor in current_state_dict.items():
                dst_tensor = self._best_weights_buffer[name]
                dst_tensor.copy_(src_tensor, non_blocking=True)
        return self._best_weights_buffer

    def _maybe_set_best_weights_and_elite(
        self, best_weights: Optional[Dict], best_val_score: Optional[torch.Tensor]
    ):
        """Restore best weights and select elite models.

        Args:
            best_weights (dict, optional): the state dict of the best model.
                If ``None``, the model weights are left as-is.
            best_val_score (tensor, optional): the best validation score per
                ensemble member. Used for elite selection.
        """
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        if best_val_score is not None and len(best_val_score) > 1:
            num_elites = getattr(self.model, "num_elites", None)
            if num_elites is None and isinstance(self.model, OneDTransitionRewardModel):
                num_elites = getattr(self.model.model, "num_elites", None)
            if num_elites is not None:
                sorted_indices = np.argsort(best_val_score.tolist())
                elite_models = sorted_indices[:num_elites]
                self.model.set_elite(elite_models)
