# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import functools
import itertools
import tempfile
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from torch import optim as optim
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from mbrl.util.logger import Logger
from mbrl.util.replay_buffer import BootstrapIterator, TransitionIterator
from mbrl.util.torchrl_util import transition_batch_to_tensordict

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
    def __init__(self, train_iteration, legacy_callback, batch_callback, logger=None):
        self.train_iteration = train_iteration
        self.legacy_callback = legacy_callback
        self.batch_callback = batch_callback
        self.logger = logger
        self.train_losses = []
        self.val_losses = []
        self.epoch_val_scores = []
        self.best_val_loss = float("inf")
        self.best_avg_scores = None

    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch_val_scores = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get(
            "train/loss", metrics.get("train_loss", torch.tensor(0.0))
        ).item()
        val_loss = metrics.get(
            "val/loss", metrics.get("val_loss", torch.tensor(0.0))
        ).item()

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if self.legacy_callback:
            # We need to match legacy callback arguments
            self.legacy_callback(
                pl_module, self.train_iteration, trainer.current_epoch
            )

        if self.logger:
            log_dict = {
                "train_iteration": self.train_iteration,
                "epoch": trainer.current_epoch,
                "train_dataset_size": len(trainer.train_dataloader.dataset.it),
                "val_dataset_size": len(trainer.val_dataloaders.dataset.it)
                if trainer.val_dataloaders
                else 0,
                "model_loss": train_loss,
                "model_val_score": val_loss,
                "model_best_val_score": self.best_val_loss
                if self.best_val_loss != float("inf")
                else val_loss,
            }

            for k, v in metrics.items():
                if k not in ["train/loss", "val/loss", "train_loss", "val_loss"]:
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    log_dict[k] = v

            self.logger.log_data("model_train", log_dict)
            self.logger._dump("model_train")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.batch_callback:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            meta = (
                {k: v for k, v in outputs.items() if k != "loss"}
                if isinstance(outputs, dict)
                else {}
            )

            if isinstance(loss, torch.Tensor):
                loss = loss.detach().cpu().item()

            meta["loss"] = loss
            self.batch_callback(trainer.current_epoch, loss, meta, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        val_score = None
        if isinstance(outputs, dict):
            val_score = outputs.get("val_score")

        if self.batch_callback:
            score = outputs
            meta = {}
            if isinstance(outputs, dict):
                score = outputs.get("score", outputs)
                meta = {
                    k: v
                    for k, v in outputs.items()
                    if k not in ["score", "val_score"]
                }

            if isinstance(score, torch.Tensor):
                score = score.detach().cpu().numpy()

            meta["score"] = score
            self.batch_callback(trainer.current_epoch, score, meta, "eval")

        if val_score is not None:
            self.epoch_val_scores.append(val_score.detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epoch_val_scores:
            first = self.epoch_val_scores[0]
            if first.ndim == 3:  # Ensemble (E, B, Od)
                # Concatenate along batch dimension (dim 1)
                all_scores = torch.cat(self.epoch_val_scores, dim=1)
                # Average over batch (1) and output dim (2) to get (E,)
                avg_scores = all_scores.mean(dim=(1, 2))

                current_loss = avg_scores.mean().item()
                if current_loss < self.best_val_loss:
                    self.best_val_loss = current_loss
                    self.best_avg_scores = avg_scores

            # Clear memory
            self.epoch_val_scores = []


class ModelTrainer:
    """Trainer for dynamics models.

    Args:
        model (:class:`mbrl.models.Model`): a model to train.
        optim_lr (float): the learning rate for the optimizer (using Adam).
        weight_decay (float): the weight decay to use.
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
    ):
        self.model = model
        self._train_iteration = 0

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

        # Optimizer configuration is now handled via monkeypatching configure_optimizers
        # on the model instance, which Lightning will call during fit().
        def configure_optimizers():
            return optim.Adam(
                self.model.parameters(),
                lr=self.optim_lr,
                weight_decay=self.weight_decay,
                eps=self.optim_eps,
            )

        self.model.configure_optimizers = configure_optimizers

    def train(
        self,
        dataset_train: TransitionIterator,
        dataset_val: Optional[TransitionIterator] = None,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        improvement_threshold: float = 0.0,
        callback: Optional[Callable] = None,
        batch_callback: Optional[Callable] = None,
        evaluate: bool = True,
        silent: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Trains the model for some number of epochs.

        This method refactors the original training loop to use pytorch_lightning.Trainer.
        """
        self._train_iteration += 1

        # Bridge TransitionIterator to Lightning DataLoader
        train_loader = DataLoader(_IteratorDataset(dataset_train), batch_size=None)
        val_loader = None
        if evaluate:
            eval_dataset = dataset_train if dataset_val is None else dataset_val
            val_loader = DataLoader(_IteratorDataset(eval_dataset), batch_size=None)

        # Lightning Callbacks
        callbacks = []
        if evaluate and dataset_val and patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val/loss",
                    patience=patience,
                    min_delta=improvement_threshold,
                    mode="min",
                    check_on_train_epoch_end=False,
                )
            )

        # We can also add a custom callback for legacy callbacks
        legacy_callback = _LegacyCallback(
            self._train_iteration, callback, batch_callback, logger=self.logger
        )
        callbacks.append(legacy_callback)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_callback = ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_weights_only=True,
                dirpath=temp_dir,
                filename="best_model",
            )
            callbacks.append(checkpoint_callback)

            if self.model.device.type == "cuda":
                accelerator = "gpu"
                devices = 1
            elif self.model.device.type == "mps":
                accelerator = "mps"
                devices = 1
            else:
                accelerator = "cpu"
                devices = "auto"

            trainer = pl.Trainer(
                max_epochs=num_epochs,
                callbacks=callbacks,
                enable_progress_bar=not silent,
                devices=devices,
                accelerator=accelerator,
                logger=False,  # We use the custom logger
                num_sanity_val_steps=0,
            )

            trainer.fit(self.model, train_loader, val_loader)

            # Restore best weights
            if evaluate and checkpoint_callback.best_model_path:
                self.model.load_state_dict(
                    torch.load(checkpoint_callback.best_model_path)["state_dict"]
                )

            # Select elite models if it's an Ensemble or OneDTransitionRewardModel wrapping an Ensemble
            is_ensemble = isinstance(self.model, Ensemble)
            is_oned_ensemble = isinstance(
                self.model, OneDTransitionRewardModel
            ) and isinstance(self.model.model, Ensemble)

            if is_ensemble or is_oned_ensemble:
                avg_scores = legacy_callback.best_avg_scores

                # Fallback to manual evaluation if scores weren't captured (e.g. evaluate=False)
                if avg_scores is None and evaluate:
                    # Should not happen if evaluate=True unless no validation batches ran
                    pass
                elif avg_scores is None:
                    # If evaluate=False, we explicitly check if we need to run it now?
                    # Original logic implied if is_ensemble is True, we evaluate.
                    # But if evaluate=False, we don't have val_loader.
                    # We create one now.
                    eval_dataset = (
                        dataset_train if dataset_val is None else dataset_val
                    )
                    val_score = self.evaluate(eval_dataset)
                    if val_score.ndim > 0:
                        avg_scores = (
                            val_score.mean(dim=tuple(range(1, val_score.ndim)))
                            if val_score.ndim > 1
                            else val_score
                        )
                
                # Move to device if needed
                if avg_scores is not None:
                     avg_scores = avg_scores.to(self.model.device)

                if avg_scores is not None:
                    num_elites = getattr(self.model, "num_elites", None)
                    if is_oned_ensemble and num_elites is None:
                        num_elites = getattr(self.model.model, "num_elites", None)

                    if num_elites:
                        elite_indices = torch.argsort(avg_scores)[:num_elites]
                        self.model.set_elite(elite_indices.tolist())

        return legacy_callback.train_losses, legacy_callback.val_losses

    def evaluate(
        self, dataset: TransitionIterator, batch_callback: Optional[Callable] = None
    ) -> torch.Tensor:
        """Evaluates the model on the validation dataset.

        Iterates over the dataset, one batch at a time, and calls
        :meth:`mbrl.models.Model.eval_score` to compute the model score
        over the batch. The method returns the average score over the whole dataset.

        Args:
            dataset (bool): the transition iterator to use.
            batch_callback (callable, optional): if provided, this function will be called
                for every batch with the output of ``model.eval_score()`` (the score will
                be passed as a float, reduced using mean()). It will be called
                with four arguments ``(epoch_index, loss/score, meta, mode)``, where
                ``mode`` is the string ``"eval"``.

        Returns:
            (tensor): The average score of the model over the dataset (and for ensembles, per
                ensemble member).
        """
        if isinstance(dataset, BootstrapIterator):
            dataset.toggle_bootstrap()

        batch_scores_list = []
        for batch in dataset:
            batch_score, meta = self.model.eval_score(batch)
            batch_scores_list.append(batch_score)
            if batch_callback:
                batch_callback(batch_score.mean(), meta, "eval")
        try:
            batch_scores = torch.cat(
                batch_scores_list, dim=batch_scores_list[0].ndim - 2
            )
        except RuntimeError as e:
            print(
                f"There was an error calling ModelTrainer.evaluate(). "
                f"Note that model.eval_score() should be non-reduced. Error was: {e}"
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
        """Return the current model state dict  if the validation score improves.

        For ensembles, this checks the validation for each ensemble member separately.

        Args:
            best_val_score (tensor): the current best validation losses per model.
            val_score (tensor): the new validation loss per model.
            threshold (float): the threshold for relative improvement.

        Returns:
            (dict, optional): if the validation score's relative improvement over the
            best validation score is higher than the threshold, returns the state dictionary
            of the stored model, otherwise returns ``None``.
        """
        improvement = (best_val_score - val_score) / torch.abs(best_val_score)
        improved = (improvement > threshold).any().item()
        return copy.deepcopy(self.model.state_dict()) if improved else None

    def _maybe_set_best_weights_and_elite(
        self, best_weights: Optional[Dict], best_val_score: torch.Tensor
    ):
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        if len(best_val_score) > 1 and hasattr(self.model, "num_elites"):
            sorted_indices = np.argsort(best_val_score.tolist())
            elite_models = sorted_indices[: self.model.num_elites]
            self.model.set_elite(elite_models)
