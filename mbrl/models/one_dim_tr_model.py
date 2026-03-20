# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from deprecated import deprecated
import torch

import mbrl.models.util as model_util
import mbrl.types
import mbrl.util.math
import mbrl.util.normalization

from .model import Ensemble, Model

MODEL_LOG_FORMAT = [
    ("train_iteration", "I", "int"),
    ("epoch", "E", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_score", "MSCORE", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
]


class OneDTransitionRewardModel(Model):
    """Wrapper class for 1-D dynamics models.

    This model functions as a wrapper for another model to convert transition
    batches into 1-D transition reward models. It also provides
    data manipulations that are common when using dynamics models with 1-D observations
    and actions, so that users don't have to manipulate the underlying model's
    inputs and outputs directly (e.g., predicting delta observations, input
    normalization).

    The wrapper assumes that the wrapped model inputs/outputs will be consistent with

        [pred_obs_{t+1}, pred_rewards_{t+1} (optional)] = model([obs_t, action_t]).

    To use with :class:mbrl.models.ModelEnv`, the wrapped model must define methods
    ``reset_1d`` and ``sample_1d``.

    Args:
        model (:class:`mbrl.model.Model`): the model to wrap.
        target_is_delta (bool): if ``True``, the predicted observations will represent
            the difference respect to the input observations.
            That is, ignoring rewards, pred_obs_{t + 1} = obs_t + model([obs_t, act_t]).
            Defaults to ``True``. Can be deactivated per dimension using ``no_delta_list``.
        normalize (bool): if ``True``, an input normalizer is created (type selected
            by ``normalizer_type``).  The user must call :meth:`update_normalizer`
            before using the model.  Defaults to ``False``.
        normalize_double_precision (bool): if ``True``, the normalizer will work with
            double precision.
        learned_rewards (bool): if ``True``, the wrapper considers the last output of the model
            to correspond to rewards predictions, and will use it to construct training
            targets for the model and when returning model predictions. Defaults to ``True``.
        obs_process_fn (callable, optional): if provided, observations will be passed through
            this function before being given to the model (and before the normalizer also).
            The processed observations should have the same dimensions as the original.
            Defaults to ``None``.
        no_delta_list (list(int), optional): if provided, represents a list of dimensions over
            which the model predicts the actual observation and not just a delta.
        num_elites (int, optional): if provided, only the best ``num_elites`` models according
            to validation score are used when calling :meth:`predict`. Defaults to
            ``None`` which means that all models will always be included in the elite set.
        normalizer_type (str): ``"winsorized"`` (default), ``"quantile"`` or
            ``"standard"``.  When ``"standard"`` a single concatenated
            ``input_normalizer`` is used; otherwise separate ``obs_normalizer``
            and ``act_normalizer`` are created.
        obs_dim (int, optional): single-step observation dimensionality.  Inferred
            from the model when ``None``.
        act_dim (int, optional): single-step action dimensionality.  Inferred from
            the model when ``None``.
        normalizer_kwargs (dict, optional): extra keyword arguments forwarded to the
            normalizer factory (e.g. ``clip_range``, ``winsor_percentile``,
            ``soft_clip_iqr_mult``).
    """

    def __init__(
        self,
        model: Model,
        target_is_delta: bool = True,
        normalize: bool = False,
        normalize_double_precision: bool = False,
        learned_rewards: bool = True,
        obs_process_fn: Optional[mbrl.types.ObsProcessFnType] = None,
        no_delta_list: Optional[List[int]] = None,
        num_elites: Optional[int] = None,
        normalizer_type: str = "winsorized",
        obs_dim: Optional[int] = None,
        act_dim: Optional[int] = None,
        normalizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model.device)
        self.model = model
        self.normalizer_type = normalizer_type
        self._obs_dim = obs_dim
        self._act_dim = act_dim

        # Normalizer setup
        self.input_normalizer: Optional[mbrl.util.normalization.Normalizer] = None
        self.obs_normalizer: Optional[mbrl.util.normalization.Normalizer] = None
        self.act_normalizer: Optional[mbrl.util.normalization.Normalizer] = None
        norm_dtype = torch.double if normalize_double_precision else torch.float
        norm_kwargs = normalizer_kwargs or {}

        if normalize:
            if normalizer_type == "standard":
                # Legacy behavior: single normalizer for concatenated [obs, act]
                self.input_normalizer = mbrl.util.normalization.ZScoreNormalizer(
                    self.model.in_size,
                    self.model.device,
                    dtype=norm_dtype,
                )
            else:
                # Robust normalizer: separate obs and act normalizers
                if obs_dim is None or act_dim is None:
                    # Attempt to infer from model attributes
                    if hasattr(model, "singlestep_obs_len") and hasattr(model, "singlestep_act_len"):
                        obs_dim = obs_dim or model.singlestep_obs_len
                        act_dim = act_dim or model.singlestep_act_len
                    elif hasattr(model, "in_size") and hasattr(model, "out_size"):
                        # Single-step model: out_size = obs_dim, in_size = obs_dim + act_dim
                        obs_dim = obs_dim or model.out_size
                        act_dim = act_dim or (model.in_size - model.out_size)
                    else:
                        raise ValueError(
                            f"normalizer_type='{normalizer_type}' requires obs_dim and act_dim "
                            "to be specified (or the wrapped model must expose "
                            "singlestep_obs_len / singlestep_act_len or in_size / out_size)."
                        )
                self._obs_dim = obs_dim
                self._act_dim = act_dim
                self.obs_normalizer = mbrl.util.normalization.create_normalizer(
                    normalizer_type, obs_dim, self.model.device, dtype=norm_dtype, **norm_kwargs
                )
                self.act_normalizer = mbrl.util.normalization.create_normalizer(
                    normalizer_type, act_dim, self.model.device, dtype=norm_dtype, **norm_kwargs
                )

        self.learned_rewards = learned_rewards
        self.target_is_delta = target_is_delta
        self.no_delta_list = no_delta_list if no_delta_list else []
        self.obs_process_fn = obs_process_fn

        self.num_elites = num_elites
        if not num_elites and isinstance(self.model, Ensemble):
            self.num_elites = self.model.num_members

    def _ensure_tensor(self, val: mbrl.types.TensorType) -> torch.Tensor:
        """Convert to tensor on model device, handling MPS float64."""
        if not isinstance(val, torch.Tensor):
            val = model_util.to_tensor(val)
        if self.device.type == "mps" and val.dtype == torch.float64:
            val = val.float()
        return val.to(self.device)

    @property
    def _uses_robust_normalizer(self) -> bool:
        """True when using separate obs/act normalizers (winsorized or quantile)."""
        return self.obs_normalizer is not None

    @property
    def _is_multistep(self) -> bool:
        return hasattr(self.model, "history_len") and hasattr(self.model, "singlestep_obs_len")

    def _normalize_composed_obs(self, composed_obs: torch.Tensor) -> torch.Tensor:
        """Normalize a composed observation by decomposing into obs/act blocks.

        For multi-step models, ``composed_obs`` has shape
        ``(..., Do*H + Da*L)`` where *L* is the number of act timesteps
        (may be H-1 for batch.obs or H for model_in).  The obs block
        ``[0, Do*H)`` is reshaped to ``(-1, Do)``, normalized with
        ``obs_normalizer``, then reshaped back.  The act block
        ``[Do*H, end)`` is similarly processed with ``act_normalizer``.

        For single-step models, ``obs_normalizer.normalize`` is applied
        directly (no decomposition needed).
        """
        if not self._is_multistep:
            return self.obs_normalizer.normalize(composed_obs)

        Do = self.model.singlestep_obs_len
        Da = self.model.singlestep_act_len
        H = self.model.history_len
        leading = composed_obs.shape[:-1]

        obs_part = composed_obs[..., : Do * H]
        act_part = composed_obs[..., Do * H :]

        obs_norm = self.obs_normalizer.normalize(
            obs_part.reshape(-1, Do)
        ).reshape(*leading, Do * H)

        if act_part.shape[-1] > 0:
            act_norm = self.act_normalizer.normalize(
                act_part.reshape(-1, Da)
            ).reshape(*leading, act_part.shape[-1])
            return torch.cat([obs_norm, act_norm], dim=-1)

        return obs_norm

    def _normalize_composed_act(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize an action tensor that may span multiple timesteps.

        When the last dimension of *action* exceeds ``Da`` (single-step
        action dim), the tensor is reshaped to ``(-1, Da)``, normalized,
        and reshaped back.  For single-step actions the normalizer is
        applied directly.
        """
        Da = getattr(self.model, "singlestep_act_len", None)
        if Da is not None and action.shape[-1] > Da:
            leading = action.shape[:-1]
            return self.act_normalizer.normalize(
                action.reshape(-1, Da)
            ).reshape(*leading, action.shape[-1])
        return self.act_normalizer.normalize(action)

    def _denormalize_composed_obs(self, composed_obs_norm: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_normalize_composed_obs`."""
        if not self._is_multistep:
            return self.obs_normalizer.denormalize(composed_obs_norm)

        Do = self.model.singlestep_obs_len
        Da = self.model.singlestep_act_len
        H = self.model.history_len
        leading = composed_obs_norm.shape[:-1]

        obs_part = composed_obs_norm[..., : Do * H]
        act_part = composed_obs_norm[..., Do * H :]

        obs_denorm = self.obs_normalizer.denormalize(
            obs_part.reshape(-1, Do)
        ).reshape(*leading, Do * H)

        if act_part.shape[-1] > 0:
            act_denorm = self.act_normalizer.denormalize(
                act_part.reshape(-1, Da)
            ).reshape(*leading, act_part.shape[-1])
            return torch.cat([obs_denorm, act_denorm], dim=-1)

        return obs_denorm

    def _get_model_input(
        self,
        obs: mbrl.types.TensorType,
        action: mbrl.types.TensorType,
    ) -> torch.Tensor:
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        obs = self._ensure_tensor(obs)
        action = self._ensure_tensor(action)
        if self._uses_robust_normalizer:
            obs = self._normalize_composed_obs(obs).float().to(self.device)
            action = self._normalize_composed_act(action).float().to(self.device)
            model_in = torch.cat([obs, action], dim=obs.ndim - 1)
        else:
            model_in = torch.cat([obs, action], dim=obs.ndim - 1)
            if self.input_normalizer:
                model_in = self.input_normalizer.normalize(model_in).float().to(self.device)
        return model_in

    def _process_batch(
        self, batch: mbrl.types.TransitionBatch, _as_float: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, action, next_obs, reward, _, _ = batch.astuple()
        obs_t = self._ensure_tensor(obs)
        next_obs_t = self._ensure_tensor(next_obs)
        if self.target_is_delta:
            if self._uses_robust_normalizer:
                # Compute delta in normalized space for consistent scaling
                target_obs = self._normalize_composed_obs(next_obs_t) - self._normalize_composed_obs(obs_t)
                for dim in self.no_delta_list:
                    target_obs[..., dim] = self._normalize_composed_obs(next_obs_t)[..., dim]
            else:
                target_obs = next_obs_t - obs_t
                for dim in self.no_delta_list:
                    target_obs[..., dim] = next_obs_t[..., dim]
        else:
            if self._uses_robust_normalizer:
                target_obs = self._normalize_composed_obs(next_obs_t)
            else:
                target_obs = next_obs_t

        model_in = self._get_model_input(obs, action)
        if self.learned_rewards:
            reward_t = self._ensure_tensor(reward)
            reward_t = reward_t.unsqueeze(reward_t.ndim)
            target = torch.cat([target_obs, reward_t], dim=obs_t.ndim - 1)
        else:
            target = target_obs

        return model_in.float(), target.float()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Calls forward method of base model with the given input and args."""
        x = self._ensure_tensor(x)
        return self.model.forward(x, *args, **kwargs)

    def update_normalizer(self, batch: mbrl.types.TransitionBatch):
        """Updates the normalizer statistics using the batch of transition data.

        The normalizer will compute mean and standard deviation the obs and action in
        the transition. If an observation processing function has been provided, it will
        be called on ``obs`` before updating the normalizer.

        Args:
            batch (:class:`mbrl.types.TransitionBatch`): The batch of transition data.
                Only obs and action will be used, since these are the inputs to the model.
        """
        if self.input_normalizer is None and not self._uses_robust_normalizer:
            return
        obs = self._ensure_tensor(batch.obs)
        action = self._ensure_tensor(batch.act)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            action = action.unsqueeze(0)
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)

        if self._uses_robust_normalizer:
            # Update separate obs and act normalizers
            # For multi-step models, pool across timesteps (Section 3.3.6)
            if self._is_multistep:
                Do = self.model.singlestep_obs_len
                Da = self.model.singlestep_act_len
                H = self.model.history_len
                obs_block = obs[..., : Do * H].reshape(-1, Do)  # (N*H, Do)
                # Act block in composed obs has (H-1) timesteps; pool with batch.act
                act_from_composed = obs[..., Do * H :].reshape(-1, Da)  # (N*(H-1), Da)
                act_current = action.reshape(-1, Da)  # (N, Da)
                act_pooled = torch.cat([act_from_composed, act_current], dim=0)
                self.obs_normalizer.update_stats(obs_block)
                self.act_normalizer.update_stats(act_pooled)
            else:
                self.obs_normalizer.update_stats(obs)
                self.act_normalizer.update_stats(action)
        else:
            model_in = torch.cat([obs, action], dim=obs.ndim - 1)
            self.input_normalizer.update_stats(model_in)

    def loss(
        self,
        batch: mbrl.types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the model loss over a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.loss()` on them and returns the value and the metadata
        as returned by the model.

        Args:
            batch (transition batch): a batch of transition to train the model.

        Returns:
            (tensor and optional dict): as returned by `model.loss().`
        """
        assert target is None
        model_in, target = self._process_batch(batch)
        return self.model.loss(model_in, target=target)

    @deprecated(
        reason=(
            "Model.update is deprecated and will be removed in a future version. "
            "Please use `training_step` or `pytorch_lightning.Trainer` instead."
        )
    )
    def update(
        self,
        batch: mbrl.types.TransitionBatch,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Updates the model given a batch of transitions and an optimizer.

        Args:
            batch (transition batch): a batch of transition to train the model.
            optimizer (torch optimizer): the optimizer to use to update the model.

        Returns:
            (tensor and optional dict): as returned by `model.loss().`
        """
        assert target is None
        model_in, target = self._process_batch(batch)
        return self.model.update(model_in, optimizer, target=target)

    def eval_score(
        self,
        batch: mbrl.types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Evaluates the model score over a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.eval_score()` on them and returns the value.

        Args:
            batch (transition batch): a batch of transition to train the model.

        Returns:
            (tensor): as returned by `model.eval_score().`
        """
        assert target is None
        with torch.no_grad():
            model_in, target = self._process_batch(batch)
            return self.model.eval_score(model_in, target=target)

    def get_output_and_targets(
        self, batch: mbrl.types.TransitionBatch
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Returns the model output and the target tensors given a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.forward()` on them and returns the value.
        No gradient information will be kept.

        Args:
            batch (transition batch): a batch of transition to train the model.

        Returns:
            (tuple(tensor), tensor): the model outputs and the target for this batch.
        """
        with torch.no_grad():
            model_in, target = self._process_batch(batch)
            output = self.model.forward(model_in)
        return output, target

    def sample(
        self,
        act: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        """Samples next observations and rewards from the underlying 1-D model.

        This wrapper assumes that the underlying model's sample method returns a tuple
        with just one tensor, which concatenates next_observation and reward.

        Args:
            act (tensor): the action at.
            model_state (tensor): the model state st.
            deterministic (bool): if ``True``, the model returns a deterministic
                "sample" (e.g., the mean prediction). Defaults to ``False``.
            rng (random number generator): a rng to use for sampling.

        Returns:
            (tuple of two tensors): predicted next_observation (o_{t+1}) and rewards (r_{t+1}).
        """
        obs = self._ensure_tensor(model_state["obs"])
        model_in = self._get_model_input(model_state["obs"], act)
        if not hasattr(self.model, "sample_1d"):
            raise RuntimeError(
                "OneDTransitionRewardModel requires wrapped model to define method sample_1d"
            )
        preds, next_model_state = self.model.sample_1d(
            model_in, model_state, rng=rng, deterministic=deterministic
        )
        next_observs = preds[:, :-1] if self.learned_rewards else preds
        if self._uses_robust_normalizer:
            # Model output is in normalized space; denormalize at the output boundary
            if self.target_is_delta:
                # Delta is in normalized space; add to normalized obs, then denormalize
                norm_obs = self.obs_normalizer.normalize(obs)
                next_observs_norm = next_observs + norm_obs
                for dim in self.no_delta_list:
                    next_observs_norm[:, dim] = next_observs[:, dim]
                next_observs = self.obs_normalizer.denormalize(next_observs_norm)
            else:
                next_observs = self.obs_normalizer.denormalize(next_observs)
        else:
            if self.target_is_delta:
                tmp_ = next_observs + obs
                for dim in self.no_delta_list:
                    tmp_[:, dim] = next_observs[:, dim]
                next_observs = tmp_
        rewards = preds[:, -1:] if self.learned_rewards else None
        next_model_state["obs"] = next_observs
        return next_observs, rewards, None, next_model_state

    def reset(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Calls reset on the underlying model.

        Args:
            obs (tensor): the observation from which the trajectory will be
                started. The actual value is ignore, only the shape is used.
            rng (`torch.Generator`, optional): an optional random number generator
                to use.

        Returns:
            (dict(str, tensor)): the model state necessary to continue the simulation.
        """
        if not hasattr(self.model, "reset_1d"):
            raise RuntimeError(
                "OneDTransitionRewardModel requires wrapped model to define method reset_1d"
            )
        obs = self._ensure_tensor(obs)
        model_state = {"obs": obs}
        model_state.update(self.model.reset_1d(obs, rng=rng))
        return model_state

    def save(self, save_dir: Union[str, pathlib.Path]):
        self.model.save(save_dir)
        if self.input_normalizer:
            self.input_normalizer.save(save_dir)
        if self._uses_robust_normalizer:
            save_dir = pathlib.Path(save_dir)
            obs_dir = save_dir / "obs_normalizer"
            act_dir = save_dir / "act_normalizer"
            obs_dir.mkdir(parents=True, exist_ok=True)
            act_dir.mkdir(parents=True, exist_ok=True)
            self.obs_normalizer.save(obs_dir)
            self.act_normalizer.save(act_dir)

    def load(self, load_dir: Union[str, pathlib.Path]):
        self.model.load(load_dir)
        if self.input_normalizer:
            self.input_normalizer.load(load_dir)
        if self._uses_robust_normalizer:
            load_dir = pathlib.Path(load_dir)
            self.obs_normalizer.load(load_dir / "obs_normalizer")
            self.act_normalizer.load(load_dir / "act_normalizer")

    def set_elite(self, elite_indices: Sequence[int]):
        self.model.set_elite(elite_indices)

    def __len__(self):
        return len(self.model)

    def set_propagation_method(self, propagation_method: Optional[str] = None):
        if isinstance(self.model, Ensemble):
            self.model.set_propagation_method(propagation_method)
