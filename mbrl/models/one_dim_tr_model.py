# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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


class _InputOutputNormalizerFacade(torch.nn.Module):
    """Unified normalizer facade hiding the standard / robust duality.

    Permanent helper introduced by stage S1 of the RLRP-684 normalization
    consolidation `.junie` plan
    (``refactor_normalization_denormalization_consolidation_RLRP-684.md``).

    Two physical configurations are supported behind a single interface:

    - **single**: one :class:`~mbrl.util.normalization.Normalizer` operating on
      the whole concatenated vector (the legacy ``standard`` input path).
    - **block**: a per-single-step ``obs_sub`` (size ``Do``) and ``act_sub``
      (size ``Da``) pair (the robust ``winsorized`` / ``quantile`` path).  The
      composed-tensor reshape/split is performed by the owning wrapper
      (:class:`OneDTransitionRewardModel`) via its ``_normalize_output`` /
      ``_denormalize_output`` / ``_apply_input_normalizer`` primitives.

    Save layout (per RLRP-684 §3.3 — no legacy back-compat):

    - single: one normalizer serialized flat under ``<dir>/``.
    - block:  ``<dir>/obs_sub/`` and ``<dir>/act_sub/``.
    """

    def __init__(
        self,
        single: Optional[mbrl.util.normalization.Normalizer] = None,
        obs_sub: Optional[mbrl.util.normalization.Normalizer] = None,
        act_sub: Optional[mbrl.util.normalization.Normalizer] = None,
    ):
        super().__init__()
        self.single = single
        self.obs_sub = obs_sub
        self.act_sub = act_sub

    @property
    def is_block(self) -> bool:
        return self.single is None

    # Convenience delegations for the ``single`` (standard) configuration so that
    # ``wrapper.input_normalizer.normalize(x)`` keeps working at call sites that
    # operate on the concatenated standard input.
    def normalize(self, val):
        if self.single is None:
            raise RuntimeError(
                "normalize() is only defined for a single-normalizer facade; "
                "use the wrapper _normalize_output / _apply_input_normalizer "
                "primitives for the robust block facade."
            )
        return self.single.normalize(val)

    def denormalize(self, val):
        if self.single is None:
            raise RuntimeError(
                "denormalize() is only defined for a single-normalizer facade; "
                "use the wrapper _denormalize_output primitive for the robust "
                "block facade."
            )
        return self.single.denormalize(val)

    def update_stats(self, data):
        if self.single is None:
            raise RuntimeError(
                "update_stats() is only defined for a single-normalizer facade; "
                "the robust block facade is updated per-block by the wrapper's "
                "update_normalizer()."
            )
        return self.single.update_stats(data)

    @property
    def mean(self):
        return self.single.mean if self.single is not None else None

    @property
    def std(self):
        return self.single.std if self.single is not None else None

    def save(self, save_dir: Union[str, pathlib.Path]):
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.single is not None:
            self.single.save(save_dir)
        else:
            obs_dir = save_dir / "obs_sub"
            act_dir = save_dir / "act_sub"
            obs_dir.mkdir(parents=True, exist_ok=True)
            act_dir.mkdir(parents=True, exist_ok=True)
            self.obs_sub.save(obs_dir)
            self.act_sub.save(act_dir)

    def load(self, load_dir: Union[str, pathlib.Path]):
        load_dir = pathlib.Path(load_dir)
        if self.single is not None:
            self.single.load(load_dir)
        else:
            self.obs_sub.load(load_dir / "obs_sub")
            self.act_sub.load(load_dir / "act_sub")


class OneDTransitionRewardModel(Model):
    """Wrapper class for 1-D dynamics models.

    This model functions as a wrapper for another model to convert transition
    batches into 1-D transition reward models. It also provides data
    manipulations common when using dynamics models with 1-D observations and
    actions (delta prediction, input/output normalization).

    Refactored by stage S1 of the RLRP-684 normalization consolidation `.junie`
    plan (``refactor_normalization_denormalization_consolidation_RLRP-684.md``):
    the previous ``(input_normalizer | obs_normalizer + act_normalizer)`` duality
    is unified behind a single ``(input_normalizer, output_normalizer)`` pair of
    :class:`_InputOutputNormalizerFacade` objects.

    Multi-step composed batch shape contract:
        The last axis of the model input is a flattened combination
        ``(..., O[1:Do]_{t-Hi} + ... + O[1:Do]_t + A[1:Da]_{t-Hi} + ... +
        A[1:Da]_t)`` with O=observation, A=action and Hi=history len.  Model
        output predictions are composed as ``(..., O[1:Do]_{t+1} + ... +
        O[1:Do]_{t+Ho} + A[1:Da]_{t+1} + ... + A[1:Da]_{t+Ho-1})`` with
        Ho=horizon len.

    Space-domain contract (RLRP-684 §3.5): the wrapper public API
    consumes/produces UN-NORMALIZED tensors; the wrapped dynamic model
    consumes/produces NORMALIZED tensors; the wrapper is the only normalization
    boundary.

    Normalizer design rationale (single-step vs multi-step, symmetric vs
    asymmetric) — read this before touching the normalization code:

    * Two facade *shapes* exist behind the ``input_normalizer`` /
      ``output_normalizer`` pair:

      - **single** facade: ONE ``ZScoreNormalizer`` fitted over the *whole
        concatenated* model input vector (per-position statistics).  Used only
        by ``normalizer_type="standard"``.
      - **block** facade: a pair of *block-shared, per-single-step*
        sub-normalizers ``obs_sub`` (width ``Do``) and ``act_sub`` (width
        ``Da``), shared between the input and output facades.  Used by
        ``"standard_symmetric"`` (plain Z-score) and the robust ``"winsorized"``
        / ``"quantile"`` variants.

    * Single-step vs multi-step is handled by the SAME block sub-normalizers and
      is *not* a separate code path: a composed tensor of layout
      ``[obs(Do*k) | act(Da*m)]`` is reshaped to a per-single-step ``(-1, Do)`` /
      ``(-1, Da)`` view before the sub-normalizer is applied, then reshaped
      back.  Hence training-time *multi-step* predictions (``k=H``, ``m=H-1``)
      and deploy-time *single-step* observations (``k=1``, ``m=0``) are denorm-
      alized by the exact same statistics — the multi-step result is bit-
      identical to denormalizing each single step independently
      (see ``_output_primitive`` and ``_output_layout_steps``, and the
      ``TestRobustDenormMultiVsSingleStep`` regression suite).

    * Symmetric vs asymmetric refers to whether the regression *target* is
      normalized:

      - **asymmetric** (``"standard"``): input is normalized, target stays in
        RAW space (``output_normalizer is None``).  This preserves upstream
        mbrl PETS / MBPO behaviour.  Because input and target live in different
        spaces, downstream auto-regressive (AR) loops MUST run the
        ``denormalize -> shift -> renormalize`` round-trip (fed by the
        ``input_normalizer`` handle propagated to the wrapped model).
      - **symmetric** (``"standard_symmetric"`` / ``"winsorized"`` /
        ``"quantile"``): both input and target are normalized in the same block
        space, so the wrapped model lives entirely in normalized space, the
        wrapper denormalizes predictions at its boundary, and the AR round-trip
        collapses to a no-op (no normalizer handle is propagated).

    Residual (``target_is_delta``) targets — single-step and multi-step
    (RLRP-686):

    * The wrapper converts ``(obs, action, next_obs)`` into the regression
      target.  Under ``target_is_delta`` the obs target is the per-single-step
      RESIDUAL taken in RAW space, ``next_obs - obs`` (``no_delta_list`` feature
      columns are kept ABSOLUTE), then normalized by a SEPARATE output
      ``obs_sub`` fitted on the residual distribution (see ``_build_normalizers``
      / ``update_normalizer``).  Fitting the residual as a single warped quantity
      preserves the bounded / outlier-robust guarantee of winsorized / quantile
      (it is NOT a difference of two warped z-scores).  This works identically
      for single-step (``Ho=1``) and multi-step (``MultiStepMLP``) models because
      ``obs``/``next_obs`` are the same ``Hi`` window shifted by one step and the
      wrapped model slices the horizon internally.
    * Deploy reconstruction (``sample``): the prediction is a residual, so the
      raw next obs is ``obs + denormalize_output(residual)`` (``no_delta`` dims
      absolute).
    * Auto-regressive feedback (block facades only): the AR loop must map a
      predicted RESIDUAL back to the next ABSOLUTE obs slot before splicing it
      into the sliding window — see :meth:`reconstruct_ar_step_obs` and its
      wiring via ``ExponentialFamilyMLP.set_ar_step_obs_reconstructor``.  The
      asymmetric ``standard`` path keeps ``target_is_delta`` single-step-only
      (upstream PETS / MBPO), handled by the raw-space AR round-trip above.
    """
    # (CRITICAL) ToDo: Rethink the target is delta logic. (ref task RLRP-686)
    # Implementation is not working as expected. Results are actulay worst. Might be because
    # the observations are velocity so target delta become acceleration which amplifie the noise.
    # Also the implementation is error prone and it compute target delta only on the observation
    # part of the composed observation (i.e., it does not compute delta target on the action part)
    # which mean that the observation components are realy not on the same scale as the action part.
    # We need to rethink this!
    #

    _LEGACY_NORMALIZER_DIRS = ("obs_normalizer", "act_normalizer")

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

        self.input_normalizer: Optional[_InputOutputNormalizerFacade] = None
        self.output_normalizer: Optional[_InputOutputNormalizerFacade] = None
        norm_dtype = torch.double if normalize_double_precision else torch.float
        norm_kwargs = normalizer_kwargs or {}

        self.learned_rewards = learned_rewards
        self.target_is_delta = target_is_delta
        self.no_delta_list = no_delta_list if no_delta_list else []
        self.obs_process_fn = obs_process_fn

        # RLRP-686: target_is_delta/no_delta_list must be known before building
        # normalizers so the block facade can allocate a SEPARATE residual-fitted
        # output obs sub-normalizer for the delta case (see _build_normalizers).
        if normalize:
            self._build_normalizers(normalizer_type, obs_dim, act_dim, norm_dtype, norm_kwargs)

        self.num_elites = num_elites
        if not num_elites and isinstance(self.model, Ensemble):
            self.num_elites = self.model.num_members

    def _build_normalizers(self, normalizer_type, obs_dim, act_dim, norm_dtype, norm_kwargs):
        """Construct the unified ``input_normalizer`` / ``output_normalizer`` facades.

        - ``standard`` (asymmetric Z-score): a single concatenated
          ``ZScoreNormalizer`` is used as the input normalizer; the output
          normalizer is ``None`` (target stays in raw space, matching the
          legacy standard behaviour and upstream mbrl PETS / MBPO). Downstream
          AR loops keep the ``denormalize -> shift -> renormalize`` round-trip
          for this path.
        - ``standard_symmetric`` (NEW, RLRP-684 A1): block-shared per-single-step
          ``obs_sub`` (size ``Do``) / ``act_sub`` (size ``Da``)
          ``ZScoreNormalizer``s shared by both the input and output facades, so
          the target is normalized and predictions are denormalized at the
          wrapper boundary. The AR round-trip is a no-op for this path.
        - ``winsorized`` / ``quantile``: same block facade as
          ``standard_symmetric`` but with the robust normalizer variants.
        """
        if normalizer_type == "standard":
            single = mbrl.util.normalization.ZScoreNormalizer(
                self.model.in_size, self.model.device, dtype=norm_dtype,
            )
            self.input_normalizer = _InputOutputNormalizerFacade(single=single)
            self.output_normalizer = None
            return

        obs_dim, act_dim = self._resolve_obs_act_dim(normalizer_type, obs_dim, act_dim)
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        obs_norm_kwargs, act_norm_kwargs = self._split_normalizer_kwargs(
            norm_kwargs, obs_dim, act_dim
        )
        obs_sub = mbrl.util.normalization.create_normalizer(
            normalizer_type, obs_dim, self.model.device, dtype=norm_dtype, **obs_norm_kwargs,
        )
        act_sub = mbrl.util.normalization.create_normalizer(
            normalizer_type, act_dim, self.model.device, dtype=norm_dtype, **act_norm_kwargs,
        )
        self.input_normalizer = _InputOutputNormalizerFacade(obs_sub=obs_sub, act_sub=act_sub)
        if self.target_is_delta:
            # RLRP-686 (Q3 resolution): the OUTPUT/target obs normalizer is fitted
            # on the per-single-step RESIDUAL distribution (next_obs - obs), kept
            # SEPARATE from the input obs normalizer.  This keeps the delta target a
            # single winsorized/quantile-warped quantity (bounded / outlier-robust)
            # rather than a difference of two warped z-scores.  ``no_delta_list``
            # feature columns are instead fitted on the absolute-obs distribution
            # (see update_normalizer).  The act sub-normalizer is shared (actions in
            # the composed output are absolute, never residualized).
            out_obs_sub = mbrl.util.normalization.create_normalizer(
                normalizer_type, obs_dim, self.model.device, dtype=norm_dtype, **obs_norm_kwargs,
            )
            self.output_normalizer = _InputOutputNormalizerFacade(
                obs_sub=out_obs_sub, act_sub=act_sub
            )
        else:
            # Symmetric absolute target: input and output share the same physical
            # sub-normalizers (unchanged RLRP-684 behaviour).
            self.output_normalizer = _InputOutputNormalizerFacade(
                obs_sub=obs_sub, act_sub=act_sub
            )

    def _resolve_obs_act_dim(self, normalizer_type, obs_dim, act_dim):
        if obs_dim is not None and act_dim is not None:
            return obs_dim, act_dim
        model = self.model
        if hasattr(model, "singlestep_obs_len") and hasattr(model, "singlestep_act_len"):
            return (obs_dim or model.singlestep_obs_len), (act_dim or model.singlestep_act_len)
        if hasattr(model, "in_size") and hasattr(model, "out_size"):
            return (obs_dim or model.out_size), (act_dim or (model.in_size - model.out_size))
        raise ValueError(
            f"normalizer_type='{normalizer_type}' requires obs_dim and act_dim "
            "to be specified (or the wrapped model must expose "
            "singlestep_obs_len / singlestep_act_len or in_size / out_size)."
        )

    @staticmethod
    def _split_normalizer_kwargs(
        norm_kwargs: Dict[str, Any], obs_dim: int, act_dim: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split normalizer kwargs into obs/act subsets for the per-`feature_dim`
        config API (:class:`SoftWinsorizedNormalizer`).

        Recognised keys (all optional, all back-compatible):

        - ``feature_dim_names``: ordered list of length ``obs_dim + act_dim``
          split into the first ``obs_dim`` names for the obs sub-normalizer and
          the remaining ``act_dim`` names for the act sub-normalizer.
        - ``obs_feature_dim_names`` / ``act_feature_dim_names``: explicit
          per-normalizer lists; take precedence over ``feature_dim_names``.
        - ``winsor_percentile`` / ``soft_clip_iqr_mult``: scalars are forwarded
          verbatim; ``Mapping`` values are split key-by-key using the resolved
          obs/act ``feature_dim_names`` subsets.
        """
        try:
            from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore

            _OMEGA_AVAILABLE = True
        except ImportError:  # pragma: no cover - omegaconf is a hard dep here
            _OMEGA_AVAILABLE = False
            DictConfig = ListConfig = OmegaConf = None  # type: ignore

        def _to_plain(v):
            if _OMEGA_AVAILABLE and isinstance(v, (DictConfig, ListConfig)):
                return OmegaConf.to_container(v, resolve=True)
            return v

        def _is_mapping(v):
            return isinstance(v, Mapping) or (
                _OMEGA_AVAILABLE and isinstance(v, DictConfig)
            )

        combined_names = _to_plain(norm_kwargs.get("feature_dim_names"))
        obs_names = _to_plain(norm_kwargs.get("obs_feature_dim_names"))
        act_names = _to_plain(norm_kwargs.get("act_feature_dim_names"))

        if combined_names is not None and (obs_names is None or act_names is None):
            if len(combined_names) != obs_dim + act_dim:
                raise ValueError(
                    f"normalizer_kwargs.feature_dim_names has length "
                    f"{len(combined_names)} but obs_dim + act_dim = "
                    f"{obs_dim + act_dim}.  Provide the concatenation of "
                    f"`obs_dims + act_dims` from your simulator config."
                )
            if obs_names is None:
                obs_names = list(combined_names[:obs_dim])
            if act_names is None:
                act_names = list(combined_names[obs_dim:])

        wrapper_only_keys = {
            "feature_dim_names",
            "obs_feature_dim_names",
            "act_feature_dim_names",
        }

        obs_out: Dict[str, Any] = {}
        act_out: Dict[str, Any] = {}
        for k, v in norm_kwargs.items():
            if k in wrapper_only_keys:
                continue
            plain_v = _to_plain(v)
            if _is_mapping(v) and obs_names is not None and act_names is not None:
                obs_out[k] = {n: plain_v[n] for n in obs_names if n in plain_v}
                act_out[k] = {n: plain_v[n] for n in act_names if n in plain_v}
            else:
                obs_out[k] = plain_v
                act_out[k] = plain_v

        if obs_names is not None:
            obs_out["feature_dim_names"] = list(obs_names)
        if act_names is not None:
            act_out["feature_dim_names"] = list(act_names)

        return obs_out, act_out

    # ---- tensor / layout helpers --------------------------------------------

    def _ensure_tensor(self, val: mbrl.types.TensorType) -> torch.Tensor:
        """Convert to tensor on model device, handling MPS float64."""
        if not isinstance(val, torch.Tensor):
            val = model_util.to_tensor(val)
        if self.device.type == "mps" and val.dtype == torch.float64:
            val = val.float()
        return val.to(self.device)

    @property
    def _is_multistep(self) -> bool:
        return hasattr(self.model, "history_len") and hasattr(self.model, "singlestep_obs_len")

    @property
    def _uses_block_normalizer(self) -> bool:
        """True when a *block* facade is active (``standard_symmetric`` /
        ``winsorized`` / ``quantile``), i.e. a per-single-step ``obs_sub`` /
        ``act_sub`` normalizer pair with a symmetric input+output facade.

        This is False for the asymmetric ``"standard"`` path (single
        per-position ``ZScoreNormalizer``, raw target, ``output_normalizer`` is
        ``None``).  Renamed from ``_uses_robust_normalizer`` (RLRP-685): the
        block facade now also covers ``standard_symmetric``, which is *not* a
        robust/bounded normalizer, so "block" is the accurate distinction.
        """
        return self.output_normalizer is not None and self.output_normalizer.is_block

    @property
    def obs_normalizer(self):
        """Backward-compat accessor: the robust obs sub-normalizer (or ``None``)."""
        f = self.output_normalizer
        return f.obs_sub if (f is not None and f.is_block) else None

    @property
    def act_normalizer(self):
        """Backward-compat accessor: the robust act sub-normalizer (or ``None``)."""
        f = self.output_normalizer
        return f.act_sub if (f is not None and f.is_block) else None

    @property
    def _Do(self) -> int:
        if self._obs_dim is not None:
            return self._obs_dim
        if hasattr(self.model, "singlestep_obs_len"):
            return self.model.singlestep_obs_len
        return self.model.out_size

    @property
    def _Da(self) -> int:
        if self._act_dim is not None:
            return self._act_dim
        if hasattr(self.model, "singlestep_act_len"):
            return self.model.singlestep_act_len
        return self.model.in_size - self.model.out_size

    def _output_layout_steps(self) -> Tuple[int, int]:
        """Return ``(obs_steps, act_steps)`` for the wrapped model's output.

        ``(horizon_len, horizon_len - 1)`` for multi-step models, else
        ``(1, 0)``.  This is the only place the output layout is decided; every
        other caller passes the tuple explicitly.
        """
        if hasattr(self.model, "horizon_len"):
            ho = self.model.horizon_len
            return ho, max(ho - 1, 0)
        return 1, 0

    # ---- normalization primitives (single source of truth) ------------------

    def _apply_input_normalizer(self, model_in: torch.Tensor) -> torch.Tensor:
        """Normalize a composed model input.

        For the ``standard`` single facade the whole vector is normalized.  For
        the robust block facade the input is split into a ``Do``-wide obs block
        (``history_len`` steps) and the remaining act block (``Da``-wide steps),
        each normalized per single-step feature, then reassembled.
        """
        facade = self.input_normalizer
        if facade is None:
            return model_in
        if facade.single is not None:
            return facade.single.normalize(model_in).float().to(self.device)

        Do, Da = self._Do, self._Da
        leading = model_in.shape[:-1]
        if self._is_multistep:
            H = self.model.history_len
            obs_part = model_in[..., : Do * H]
            act_part = model_in[..., Do * H :]
        else:
            obs_part = model_in[..., :Do]
            act_part = model_in[..., Do:]
        obs_norm = facade.obs_sub.normalize(obs_part.reshape(-1, Do)).reshape(
            *leading, obs_part.shape[-1]
        )
        if act_part.shape[-1] > 0:
            act_norm = facade.act_sub.normalize(act_part.reshape(-1, Da)).reshape(
                *leading, act_part.shape[-1]
            )
            return torch.cat([obs_norm, act_norm], dim=-1).float().to(self.device)
        return obs_norm.float().to(self.device)

    def _normalize_output(
        self, y: torch.Tensor, *, obs_steps: int, act_steps: int
    ) -> torch.Tensor:
        """Forward output primitive: normalize a composed prediction/target of
        layout ``(..., Do * obs_steps + Da * act_steps)``.

        Splits the last axis at ``Do * obs_steps``; the obs block is reshaped to
        ``(-1, Do)`` and normalized by the output obs sub-normalizer, the act
        block (if ``act_steps > 0``) by the act sub-normalizer.
        """
        return self._output_primitive(
            y, obs_steps=obs_steps, act_steps=act_steps, denorm=False
        )

    def _denormalize_output(
        self, y_norm: torch.Tensor, *, obs_steps: int, act_steps: int
    ) -> torch.Tensor:
        """Inverse of :meth:`_normalize_output` (single source of truth for
        all inference-time denormalization)."""
        return self._output_primitive(
            y_norm, obs_steps=obs_steps, act_steps=act_steps, denorm=True
        )

    def _output_primitive(self, y, *, obs_steps, act_steps, denorm):
        """Single source of truth for output (de)normalization.

        Layout contract: ``y`` has last-axis layout
        ``[obs(Do*obs_steps) | act(Da*act_steps)]``.  Each block is reshaped to a
        per-single-step ``(-1, Do)`` / ``(-1, Da)`` view, run through the
        block-shared ``obs_sub`` / ``act_sub`` sub-normalizer (``denormalize``
        when ``denorm`` else ``normalize``), then reshaped back to the original
        leading dims.  Because the transform is per-single-step, the
        multi-step training case (``obs_steps=H, act_steps=H-1``) and the
        single-step deploy case (``obs_steps=1, act_steps=0``) share identical
        statistics — see the class-docstring "Normalizer design rationale".
        """
        facade = self.output_normalizer
        if facade is None:
            return y
        Do, Da = self._Do, self._Da
        leading = y.shape[:-1]
        obs_len = Do * obs_steps
        obs_part = y[..., :obs_len]
        act_part = y[..., obs_len:]
        op_obs = facade.obs_sub.denormalize if denorm else facade.obs_sub.normalize
        obs_out = op_obs(obs_part.reshape(-1, Do))
        if len(leading) > 0:
            obs_out = obs_out.reshape(*leading, obs_len)
        else:
            obs_out = obs_out.reshape(obs_len)
        if act_steps > 0 and act_part.shape[-1] > 0:
            op_act = facade.act_sub.denormalize if denorm else facade.act_sub.normalize
            act_out = op_act(act_part.reshape(-1, Da)).reshape(
                *leading, act_part.shape[-1]
            )
            return torch.cat([obs_out, act_out], dim=-1)
        return obs_out

    # ---- public deployment-time denorm helpers (thin wrappers) --------------

    def denormalize_predicted_obs(
        self,
        obs_pred_norm: torch.Tensor,
        horizon_steps: Optional[int] = None,
        add_obs_baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denormalize a (possibly sliced) multi-step obs prediction.

        ``obs_pred_norm`` has shape ``(..., Do * k)`` in NORMALIZED space.  When
        ``self.target_is_delta`` and ``add_obs_baseline`` (a single-step ``O_t``
        of shape ``(..., Do)``) is provided, the prediction is treated as a
        per-single-step RESIDUAL and reconstructed in RAW space following the
        RLRP-686 (Q3) contract — the SAME contract used by :meth:`sample` and
        :meth:`reconstruct_ar_step_obs`::

            res_raw    = output_denormalize(obs_pred_norm)
            O_next_raw = add_obs_baseline + res_raw     # delta dims (baseline
                                                        # broadcast across the
                                                        # k horizon slots)
            O_next_raw = res_raw                        # no_delta dims (absolute)

        The baseline is added in RAW (not normalized) space so the
        reconstruction is exact for the NONLINEAR ``winsorized`` / ``quantile``
        warps as well as the linear ``standard_symmetric`` Z-score
        (``denorm(norm(b) + r) != b + denorm(r)`` for nonlinear normalizers).
        """
        if self.output_normalizer is None:
            return obs_pred_norm
        k = horizon_steps if horizon_steps is not None else (obs_pred_norm.shape[-1] // self._Do)
        res_raw = self._denormalize_output(obs_pred_norm, obs_steps=k, act_steps=0)
        if self.target_is_delta and add_obs_baseline is not None:
            Do = self._Do
            leading = res_raw.shape[:-1]
            res_view = res_raw.reshape(*leading, k, Do)
            out_view = res_view + add_obs_baseline.unsqueeze(-2)
            for dim in self.no_delta_list:
                out_view[..., dim] = res_view[..., dim]
            return out_view.reshape(*leading, Do * k)
        return res_raw

    def denormalize_predicted_act(
        self,
        act_pred_norm: torch.Tensor,
        horizon_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Symmetric helper for the act side of the composed output."""
        if self.output_normalizer is None:
            return act_pred_norm
        k = horizon_steps if horizon_steps is not None else (act_pred_norm.shape[-1] // self._Da)
        # Reuse the primitive with the act block only (obs_steps=0).
        facade = self.output_normalizer
        leading = act_pred_norm.shape[:-1]
        out = facade.act_sub.denormalize(act_pred_norm.reshape(-1, self._Da))
        return out.reshape(*leading, self._Da * k) if len(leading) > 0 else out.reshape(self._Da * k)

    def reconstruct_ar_step_obs(
        self,
        prev_obs_norm: torch.Tensor,
        predicted_residual: torch.Tensor,
    ) -> torch.Tensor:
        """RLRP-686: map an AR-predicted RESIDUAL to the next ABSOLUTE obs slot.

        Used by the auto-regressive feedback loop (block-facade
        ``target_is_delta`` path) to splice the model prediction back into the
        sliding-window input.  Both ``prev_obs_norm`` (the latest single-step obs
        slot of the window) and the return value live in INPUT-normalized space;
        ``predicted_residual`` is the model output, a per-single-step residual in
        OUTPUT-normalized space.  All tensors are ``Do``-wide on the last axis.

        Contract (per §3.7)::

            O_prev_raw = input_denormalize(prev_obs_norm)
            res_raw    = output_denormalize(predicted_residual)
            O_next_raw = O_prev_raw + res_raw        # no_delta dims: = res_raw
            return input_normalize(O_next_raw)

        For ``target_is_delta=False`` the prediction is already an absolute obs in
        (shared) normalized space, so it is returned unchanged.
        """
        if not self.target_is_delta:
            return predicted_residual
        Do = self._Do
        prev_shape = prev_obs_norm.shape
        res_shape = predicted_residual.shape
        O_prev_raw = self.input_normalizer.obs_sub.denormalize(
            prev_obs_norm.reshape(-1, Do)
        ).reshape(prev_shape)
        res_raw = self.output_normalizer.obs_sub.denormalize(
            predicted_residual.reshape(-1, Do)
        ).reshape(res_shape)
        O_next_raw = O_prev_raw + res_raw
        for dim in self.no_delta_list:
            O_next_raw[..., dim] = res_raw[..., dim]
        return self.input_normalizer.obs_sub.normalize(
            O_next_raw.reshape(-1, Do)
        ).reshape(res_shape)

    # ---- robust composed-tensor shims (RLRP-684 S1 interim) -----------------
    # These delegate to the unified obs/act sub-normalizers behind the
    # ``output_normalizer`` block facade.  They preserve the pre-refactor
    # composed-obs / composed-act semantics bit-for-bit so existing call sites
    # (notably ``OneDTransitionRewardModelV2``) keep working while the broader
    # call-site sweep to the ``_normalize_output`` / ``_denormalize_output``
    # primitives is completed in a follow-up.

    def _normalize_composed_obs(self, composed_obs: torch.Tensor) -> torch.Tensor:
        facade = self.output_normalizer
        Do, Da = self._Do, self._Da
        if not self._is_multistep:
            return facade.obs_sub.normalize(composed_obs)
        H = self.model.history_len
        leading = composed_obs.shape[:-1]
        obs_part = composed_obs[..., : Do * H]
        act_part = composed_obs[..., Do * H :]
        obs_norm = facade.obs_sub.normalize(obs_part.reshape(-1, Do)).reshape(
            *leading, Do * H
        )
        if act_part.shape[-1] > 0:
            act_norm = facade.act_sub.normalize(act_part.reshape(-1, Da)).reshape(
                *leading, act_part.shape[-1]
            )
            return torch.cat([obs_norm, act_norm], dim=-1)
        return obs_norm

    def _normalize_composed_act(self, action: torch.Tensor) -> torch.Tensor:
        facade = self.output_normalizer
        Da = self._Da
        if action.shape[-1] > Da:
            leading = action.shape[:-1]
            return facade.act_sub.normalize(action.reshape(-1, Da)).reshape(
                *leading, action.shape[-1]
            )
        return facade.act_sub.normalize(action)

    def _normalize_composed_obs_input(self, composed_obs: torch.Tensor) -> torch.Tensor:
        """Normalize a composed obs tensor with the INPUT (absolute) facade.

        RLRP-686: this is the correct normalizer for the *model input* obs.  It
        must NOT route through ``output_normalizer`` because, for the
        ``target_is_delta`` block path, the output obs sub-normalizer is fitted
        on the per-single-step RESIDUAL distribution (a different statistic),
        whereas the model input carries ABSOLUTE observations and must use the
        input obs sub-normalizer.  For the non-delta path the input and output
        sub-normalizers are physically shared, so this is bit-equivalent to
        :meth:`_normalize_composed_obs`.
        """
        facade = self.input_normalizer
        Do, Da = self._Do, self._Da
        if not self._is_multistep:
            return facade.obs_sub.normalize(composed_obs)
        H = self.model.history_len
        leading = composed_obs.shape[:-1]
        obs_part = composed_obs[..., : Do * H]
        act_part = composed_obs[..., Do * H :]
        obs_norm = facade.obs_sub.normalize(obs_part.reshape(-1, Do)).reshape(
            *leading, Do * H
        )
        if act_part.shape[-1] > 0:
            act_norm = facade.act_sub.normalize(act_part.reshape(-1, Da)).reshape(
                *leading, act_part.shape[-1]
            )
            return torch.cat([obs_norm, act_norm], dim=-1)
        return obs_norm

    def _denormalize_composed_obs(self, composed_obs_norm: torch.Tensor) -> torch.Tensor:
        facade = self.output_normalizer
        Do, Da = self._Do, self._Da
        if not self._is_multistep:
            return facade.obs_sub.denormalize(composed_obs_norm)
        H = self.model.history_len
        leading = composed_obs_norm.shape[:-1]
        obs_part = composed_obs_norm[..., : Do * H]
        act_part = composed_obs_norm[..., Do * H :]
        obs_denorm = facade.obs_sub.denormalize(obs_part.reshape(-1, Do))
        if len(leading) > 0:
            obs_denorm = obs_denorm.reshape(*leading, Do * H)
        if act_part.shape[-1] > 0:
            act_denorm = facade.act_sub.denormalize(act_part.reshape(-1, Da)).reshape(
                *leading, act_part.shape[-1]
            )
            return torch.cat([obs_denorm, act_denorm], dim=-1)
        return obs_denorm

    # ---- model input / batch processing -------------------------------------

    def _get_model_input(
        self,
        obs: mbrl.types.TensorType,
        action: mbrl.types.TensorType,
    ) -> torch.Tensor:
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        obs = self._ensure_tensor(obs)
        action = self._ensure_tensor(action)
        model_in = torch.cat([obs, action], dim=obs.ndim - 1)
        if self.input_normalizer is not None:
            model_in = self._apply_input_normalizer(model_in)
        return model_in

    def _obs_block_width(self, composed: torch.Tensor) -> int:
        """Width (last-axis) of the observation block in a composed obs tensor.

        For multi-step models the composed tensor is ``[obs(Do*H) | act(...)]``;
        otherwise the whole tensor is the (single-step) obs block.
        """
        if self._is_multistep:
            return self._Do * self.model.history_len
        return composed.shape[-1]

    def _raw_residual_obs_target(
        self, obs_t: torch.Tensor, next_obs_t: torch.Tensor
    ) -> torch.Tensor:
        """Build the RAW per-single-step residual target over the obs block.

        RLRP-686: ``mix`` equals ``next_obs - obs`` on the observation block
        (consecutive single-step residual, since ``obs`` / ``next_obs`` are the
        same ``Hi``-composed window shifted by one step), keeps the absolute
        ``next_obs`` value on ``no_delta_list`` feature columns (single-step
        relative, broadcast across all slots via ``dim::Do``), and leaves any
        trailing act block as the absolute ``next_obs`` (actions are never
        residualized).  The caller normalizes this with the residual-fitted
        output obs sub-normalizer.
        """
        Do = self._Do
        ow = self._obs_block_width(next_obs_t)
        mix = next_obs_t.clone()
        mix[..., :ow] = next_obs_t[..., :ow] - obs_t[..., :ow]
        for dim in self.no_delta_list:
            mix[..., dim:ow:Do] = next_obs_t[..., dim:ow:Do]
        return mix

    def _process_batch(
        self, batch: mbrl.types.TransitionBatch, _as_float: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, action, next_obs, reward, _, _ = batch.astuple()
        obs_t = self._ensure_tensor(obs)
        next_obs_t = self._ensure_tensor(next_obs)

        if self.output_normalizer is not None:
            # Robust / symmetric block path: target lives in NORMALIZED space,
            # keeping the full composed observation layout (obs block + act
            # block) of next_obs.
            if self.target_is_delta:
                # RLRP-686: residual taken in RAW space, then normalized by the
                # residual-fitted output obs sub-normalizer (single bounded
                # warped value for winsorized/quantile).
                target_obs = self._normalize_composed_obs(
                    self._raw_residual_obs_target(obs_t, next_obs_t)
                )
            else:
                target_obs = self._normalize_composed_obs(next_obs_t)
        else:
            # Standard / no-normalize path: target stays in RAW space.
            if self.target_is_delta:
                target_obs = next_obs_t - obs_t
                for dim in self.no_delta_list:
                    target_obs[..., dim] = next_obs_t[..., dim]
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

        Only obs and action are used (the model inputs).  For the robust block
        facade the obs / act sub-normalizers are shared between the input and
        output facades, so a single update keeps both in sync.
        """
        if self.input_normalizer is None:
            return
        obs = self._ensure_tensor(batch.obs)
        action = self._ensure_tensor(batch.act)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            action = action.unsqueeze(0)
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)

        if self.input_normalizer.single is not None:
            model_in = torch.cat([obs, action], dim=obs.ndim - 1)
            self.input_normalizer.single.update_stats(model_in)
            return

        Do, Da = self._Do, self._Da
        if self._is_multistep:
            H = self.model.history_len
            obs_block = obs[..., : Do * H].reshape(-1, Do)
            act_from_composed = obs[..., Do * H :].reshape(-1, Da)
            act_current = action.reshape(-1, Da)
            act_pooled = torch.cat([act_from_composed, act_current], dim=0)
            self.input_normalizer.obs_sub.update_stats(obs_block)
            self.input_normalizer.act_sub.update_stats(act_pooled)
        else:
            self.input_normalizer.obs_sub.update_stats(obs.reshape(-1, Do))
            self.input_normalizer.act_sub.update_stats(action.reshape(-1, Da))

        # RLRP-686: when the output obs sub-normalizer is a SEPARATE residual-fitted
        # normalizer (target_is_delta block path), fit it on the per-single-step
        # residual distribution (next_obs - obs over the obs block; absolute for
        # no_delta_list feature columns), so the delta target stays a single
        # bounded warped quantity.
        if (
            self.target_is_delta
            and self.output_normalizer is not None
            and self.output_normalizer.obs_sub is not self.input_normalizer.obs_sub
        ):
            next_obs = self._ensure_tensor(batch.next_obs)
            if next_obs.ndim == 1:
                next_obs = next_obs.unsqueeze(0)
            if self.obs_process_fn:
                next_obs = self.obs_process_fn(next_obs)
            ow = self._obs_block_width(next_obs)
            res_mix = self._raw_residual_obs_target(obs, next_obs)
            self.output_normalizer.obs_sub.update_stats(
                res_mix[..., :ow].reshape(-1, Do)
            )

    def loss(
        self,
        batch: mbrl.types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the model loss over a batch of transitions."""
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
        """Updates the model given a batch of transitions and an optimizer."""
        assert target is None
        model_in, target = self._process_batch(batch)
        return self.model.update(model_in, optimizer, target=target)

    def eval_score(
        self,
        batch: mbrl.types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Evaluates the model score over a batch of transitions."""
        assert target is None
        with torch.no_grad():
            model_in, target = self._process_batch(batch)
            return self.model.eval_score(model_in, target=target)

    def get_output_and_targets(
        self, batch: mbrl.types.TransitionBatch
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Returns the model output and the target tensors given a batch."""
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

        The wrapper boundary is the only space crossing: ``model_state["obs"]``
        enters UN-NORMALIZED, the wrapped model runs in NORMALIZED space, and the
        returned ``next_observs`` is UN-NORMALIZED (RLRP-684 §3.5).
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
        next_obs_pred = preds[:, :-1] if self.learned_rewards else preds

        if self.target_is_delta:
            # RLRP-686 deploy reconstruction (Q3 contract): the prediction is a
            # RESIDUAL; denormalize it (residual-fitted output normalizer, or a
            # no-op for the standard/raw path) then add the RAW baseline obs.
            # ``no_delta_list`` dims are absolute (skip the baseline add).
            res_raw = self._denormalize_output(
                next_obs_pred, obs_steps=1, act_steps=0
            )
            next_observs = obs + res_raw
            for dim in self.no_delta_list:
                next_observs[..., dim] = res_raw[..., dim]
        else:
            next_observs = self._denormalize_output(
                next_obs_pred, obs_steps=1, act_steps=0
            )

        rewards = preds[:, -1:] if self.learned_rewards else None
        next_model_state["obs"] = next_observs
        return next_observs, rewards, None, next_model_state

    def reset(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Calls reset on the underlying model."""
        if not hasattr(self.model, "reset_1d"):
            raise RuntimeError(
                "OneDTransitionRewardModel requires wrapped model to define method reset_1d"
            )
        obs = self._ensure_tensor(obs)
        model_state = {"obs": obs}
        model_state.update(self.model.reset_1d(obs, rng=rng))
        return model_state

    def _check_no_legacy_normalizer_layout(self, load_dir: pathlib.Path):
        """Raise on pre-RLRP-684 checkpoint layouts (Q1 — no back-compat)."""
        has_legacy = any((load_dir / d).exists() for d in self._LEGACY_NORMALIZER_DIRS)
        has_new = (load_dir / "input_normalizer").exists() or (
            load_dir / "output_normalizer"
        ).exists()
        if has_legacy and not has_new:
            raise RuntimeError(
                f"Detected a legacy normalizer save layout under {load_dir} "
                f"({'/'.join(self._LEGACY_NORMALIZER_DIRS)} or flat normalizer files). "
                "RLRP-684 dropped back-compat for old checkpoints. Re-train or "
                "re-save the model with the current code to produce the "
                "'input_normalizer/' (and 'output_normalizer/') layout."
            )

    def save(self, save_dir: Union[str, pathlib.Path]):
        self.model.save(save_dir)
        save_dir = pathlib.Path(save_dir)
        if self.input_normalizer is not None:
            self.input_normalizer.save(save_dir / "input_normalizer")
        if self.output_normalizer is not None:
            self.output_normalizer.save(save_dir / "output_normalizer")

    def load(self, load_dir: Union[str, pathlib.Path]):
        self.model.load(load_dir)
        load_dir = pathlib.Path(load_dir)
        self._check_no_legacy_normalizer_layout(load_dir)
        if self.input_normalizer is not None:
            self.input_normalizer.load(load_dir / "input_normalizer")
        if self.output_normalizer is not None:
            self.output_normalizer.load(load_dir / "output_normalizer")

    def set_elite(self, elite_indices: Sequence[int]):
        self.model.set_elite(elite_indices)

    def __len__(self):
        return len(self.model)

    def set_propagation_method(self, propagation_method: Optional[str] = None):
        if isinstance(self.model, Ensemble):
            self.model.set_propagation_method(propagation_method)
