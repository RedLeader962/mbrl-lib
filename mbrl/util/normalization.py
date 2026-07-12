# coding=utf-8
"""Normalization utilities for model-based reinforcement learning.

This module provides three normalizer classes with increasing outlier
robustness.  All share the same public API (``update_stats``, ``normalize``,
``denormalize``, ``save``/``load``) and can be instantiated through the
:func:`create_normalizer` factory.

Quick comparison of normalizers:
--------------------------------

- ``ZScoreNormalizer`` — Standard running-mean z-score.  No outlier protection.
  Differentiable, linearly invertible.  Cost: O(N).
- ``SoftWinsorizedNormalizer`` — Winsorized z-score + adaptive asinh soft-clip.
  Moderate outlier robustness.  Differentiable, closed-form invertible.
  Cost: O(N log N).
- ``QuantileNormalizer`` — Empirical-CDF mapping to standard normal.
  Strong outlier robustness.  Differentiable, invertible via lookup +
  interpolation.  Cost: O(N log N).

Rule of thumb — choosing a normalizer:
--------------------------------------

- Use ``ZScoreNormalizer`` when data is well-behaved (roughly Gaussian, no extreme
  outliers) and you want the fastest, simplest option.
- Use ``SoftWinsorizedNormalizer`` when data has moderate outliers or heavy tails
  but the core distribution is roughly symmetric.  Good default choice for
  robotic state/action spaces.
- Use ``QuantileNormalizer`` when the distribution is highly skewed,
  multi-modal, or when outlier robustness is critical and you can afford
  storing per-feature quantile boundaries.

Rule of thumb — parameter configuration:
----------------------------------------

*ZScoreNormalizer*

- ``clip_range`` (default ``None``): Leave at ``None`` for well-behaved data.
  Set to ``5.0``–``10.0`` when occasional spikes can produce z-scores that
  destabilize downstream layers.  A tighter range (e.g. ``3.0``) aggressively
  truncates tails and may discard useful signal.

*SoftWinsorizedNormalizer*

- ``winsor_percentile`` (default ``0.05``): Fraction of each tail clamped
  before computing mean/std.  ``0.05`` (5 %) suits most robotic data.
  Lower it to ``0.01``–``0.02`` if outliers are rare; raise to ``0.10`` for
  heavily contaminated streams.
- ``soft_clip_iqr_mult`` (default ``3.0``): IQR multiplier for the per-feature
  asinh clip threshold (``c_i ≈ mult × IQR/σ ≈ mult × 1.35`` for Gaussian
  data).  ``3.0`` keeps ≈ ±4 σ in the identity region — a safe default.
  Reduce to ``1.5``–``2.0`` for aggressive tail compression; values below
  ``0.75`` hit the internal floor of 1.0 and trigger a warning.
  Set to ``None`` to **disable** the asinh soft-clip stage entirely and
  recover the classic ``WinsorizedNormalizer`` behavior (pure winsorized
  z-score, no soft-clipping).

*QuantileNormalizer*

- ``n_bins`` (default ``1000``): Number of quantile bins.  ``1000`` is fine
  for datasets with ≥ 1 000 samples.  For very small datasets (< 500
  samples), lower to ``~N/2`` to avoid under-populated bins.  For very large
  datasets with fine-grained structure, ``2000``–``5000`` can improve
  resolution at negligible extra cost.
- ``tail_policy`` (default ``"linear"``): Currently only ``"linear"`` is
  supported — it extrapolates beyond the observed range using the slope of
  the outermost bin.  Keep the default.
"""
import abc
import difflib
import pathlib
import warnings
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import mbrl.types
import numpy as np
import torch
import torch.compiler
import torch.nn

# Type aliases for the per-`feature_dim` configuration API of
# :class:`SoftWinsorizedNormalizer` (see its docstring for details).
WinsorPercentileConfig = Union[float, Mapping[str, float], Sequence[float]]
SoftClipIqrMultConfig = Union[
    float, None, Mapping[str, Optional[float]], Sequence[Optional[float]]
]


class Normalizer(torch.nn.Module, abc.ABC):
    """Abstract base class for all normalizers in this module.

    Every concrete normalizer (``ZScoreNormalizer``, ``SoftWinsorizedNormalizer``,
    ``QuantileNormalizer``) inherits from this class and implements the
    required interface: :meth:`update_stats`, :meth:`normalize`,
    :meth:`denormalize`, :meth:`save`, and :meth:`load`.

    Subclasses must also expose ``mean`` and ``std`` attributes (either
    buffers or properties) and a ``device`` property so that downstream
    code can inspect normalizer statistics uniformly.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Device on which the normalizer operates."""

    @property
    @abc.abstractmethod
    def mean(self) -> torch.Tensor:
        """Alias for compatibility with code expecting a ``mean`` attribute."""

    @property
    @abc.abstractmethod
    def std(self) -> torch.Tensor:
        """Alias for compatibility with code expecting a ``std`` attribute."""

    @abc.abstractmethod
    def update_stats(self, data: mbrl.types.TensorType) -> None:
        """Compute and store normalization statistics from *data*.

        Args:
            data: shape ``(N, in_size)``.
        """

    @abc.abstractmethod
    def normalize(
        self,
        val: Union[float, mbrl.types.TensorType],
        strict_finite: Optional[bool] = None,
    ) -> torch.Tensor:
        """Normalize *val* according to stored statistics.

        Args:
            val: value(s) to normalize.
            strict_finite: optional per-call override of the instance-level
                :attr:`strict_finite` fail-fast policy. ``None`` (default) uses
                the instance setting. Pass ``False`` when normalizing values
                that are *model outputs* (e.g. auto-regressive test-time rollout
                predictions fed back as inputs), which may legitimately be
                non-finite / diverged early in training and should be clamped
                rather than raise; genuine *input data* keeps the strict default.

        Returns:
            Normalized tensor (same dtype as input).
        """

    @abc.abstractmethod
    def denormalize(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        """Invert the normalization applied by :meth:`normalize`.

        Args:
            val: normalized value(s) to de-normalize.

        Returns:
            De-normalized tensor (same dtype as input).
        """

    @abc.abstractmethod
    def save(self, save_dir: Union[str, pathlib.Path]) -> None:
        """Persist normalizer statistics to *save_dir*."""

    @abc.abstractmethod
    def load(self, load_dir: Union[str, pathlib.Path]) -> None:
        """Restore normalizer statistics from *load_dir*."""

    def _to_tensor(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        """Convert input to a tensor on the correct device, handling MPS dtype."""
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        if self.device.type == "mps" and val.dtype == torch.float64:
            val = val.float()
        return val.to(self.device)

    # ------------------------------------------------------------------
    # Per-feature-dimension customization (shared by ALL variants)
    # ------------------------------------------------------------------
    #
    # Introduced by stage 1 (action S1.2a) of the Per-Environment Feature
    # Handling ``.junie`` plan
    # (``rlrp-736-per-environment-feature-handling-plan-20260711.md``,
    # YouTrack RLRP-736). Lifts per-``feature_dim`` naming
    # (``feature_dim_names``) and a per-dimension enable/disable mask
    # (``normalize_dims``) to the abstract base so every concrete variant
    # (``ZScoreNormalizer``, ``SoftWinsorizedNormalizer``,
    # ``QuantileNormalizer``, and any future type) supports them uniformly.
    #
    # Back-compat: the default (``normalize_dims=True`` /
    # ``feature_dim_names=None``) yields an all-``True`` mask, so
    # :meth:`_apply_norm_mask` short-circuits and the output is byte-identical
    # to the pre-plan behaviour for every variant.

    #: File name used to persist the per-dimension mask alongside the
    #: variant-specific statistics (optional; absent for legacy checkpoints).
    _FEATURE_MASK_FNAME = "norm_feature_mask.pt"

    def _setup_feature_dim_mask(
        self,
        in_size: int,
        feature_dim_names: Optional[Sequence[str]] = None,
        normalize_dims: Union[bool, Sequence[bool], Mapping[str, bool]] = True,
    ) -> None:
        """Resolve and register the per-dimension normalization mask.

        Args:
            in_size: feature dimension of the normalizer.
            feature_dim_names: optional ordered names of length ``in_size``.
            normalize_dims: ``bool`` (all on/off), a per-dim ``bool`` sequence of
                length ``in_size``, or a ``{feature_name: bool}`` mapping (which
                requires ``feature_dim_names``). Dimensions mapped to ``False``
                are passed through untouched (identity) by
                :meth:`normalize`/:meth:`denormalize`.
        """
        names = list(feature_dim_names) if feature_dim_names is not None else None
        if names is not None and len(names) != in_size:
            raise ValueError(
                f"feature_dim_names has length {len(names)} but in_size={in_size}."
            )
        self.feature_dim_names: Optional[List[str]] = names
        mask = self._resolve_normalize_dims(normalize_dims, in_size, names)
        self.register_buffer("_norm_mask", mask)
        self._norm_mask_all_true: bool = bool(torch.all(mask).item())

    @staticmethod
    def _resolve_normalize_dims(
        normalize_dims: Union[bool, Sequence[bool], Mapping[str, bool]],
        in_size: int,
        feature_dim_names: Optional[List[str]],
    ) -> torch.Tensor:
        """Resolve ``normalize_dims`` to a 1-D ``bool`` mask of length ``in_size``.

        ``True`` means "normalize this dimension" (default); ``False`` means
        "pass through untouched". Accepts a scalar ``bool``, a per-dim sequence,
        or a ``{feature_name: bool}`` mapping.
        """
        if isinstance(normalize_dims, bool):
            return torch.full((in_size,), normalize_dims, dtype=torch.bool)
        if isinstance(normalize_dims, Mapping):
            if feature_dim_names is None:
                raise ValueError(
                    "normalize_dims given as a mapping requires feature_dim_names."
                )
            name_to_idx = {name: i for i, name in enumerate(feature_dim_names)}
            mask = torch.ones((in_size,), dtype=torch.bool)
            for key, enabled in normalize_dims.items():
                if key not in name_to_idx:
                    raise ValueError(
                        f"normalize_dims key '{key}' not found in feature_dim_names."
                    )
                mask[name_to_idx[key]] = bool(enabled)
            return mask
        seq = list(normalize_dims)
        if len(seq) != in_size:
            raise ValueError(
                f"normalize_dims sequence has length {len(seq)} but in_size={in_size}."
            )
        return torch.tensor([bool(v) for v in seq], dtype=torch.bool)

    def _apply_norm_mask(
        self, transformed: torch.Tensor, raw_val: torch.Tensor
    ) -> torch.Tensor:
        """Restore disabled dimensions to their raw (pass-through) value.

        Short-circuits (returns ``transformed`` unchanged) when the mask is
        all-``True`` — guaranteeing byte-identical output for the default
        configuration.
        """
        if getattr(self, "_norm_mask_all_true", True):
            return transformed
        raw = self._to_tensor(raw_val).to(transformed.dtype)
        # Store/apply the mask as a 1-D last-dim mask so it broadcasts against
        # any trailing ``in_size`` dimension without inserting a leading axis.
        mask = self._norm_mask.to(device=transformed.device).reshape(-1)
        return torch.where(mask, transformed, raw)

    def _save_feature_mask(self, save_dir: Union[str, pathlib.Path]) -> None:
        """Persist the per-dimension mask next to the variant statistics."""
        if getattr(self, "_norm_mask", None) is None:
            return
        torch.save(
            {
                "norm_mask": self._norm_mask.cpu(),
                "feature_dim_names": getattr(self, "feature_dim_names", None),
            },
            pathlib.Path(save_dir) / self._FEATURE_MASK_FNAME,
        )

    def _load_feature_mask(self, load_dir: Union[str, pathlib.Path]) -> None:
        """Restore the per-dimension mask if present (legacy-tolerant)."""
        path = pathlib.Path(load_dir) / self._FEATURE_MASK_FNAME
        if not path.exists():
            # Legacy checkpoint: keep the mask resolved at construction time.
            return
        payload = torch.load(path, weights_only=False, map_location="cpu")
        mask = payload["norm_mask"].to(device=self.device, dtype=torch.bool).reshape(-1)
        if getattr(self, "_norm_mask", None) is None:
            self.register_buffer("_norm_mask", mask)
        else:
            self._norm_mask = mask
        self.feature_dim_names = payload.get("feature_dim_names", None)
        self._norm_mask_all_true = bool(torch.all(self._norm_mask).item())


class ZScoreNormalizer(Normalizer):
    """Standard running-mean z-score normalizer with optional hard clipping.

    **What it does:**

    A simple one-stage normalization pipeline that computes per-feature mean
    and standard deviation from a calibration dataset, then standardizes
    incoming values as ``(x - mean) / std``.

    1. **Running z-score** — Computes ``mean`` and ``std`` from the full
       calibration data via :meth:`update_stats`, then linearly rescales
       every incoming value to zero-mean, unit-variance space.
    2. **Optional hard clip** — When ``clip_range`` is set, the z-score
       output is clamped to ``[-clip_range, clip_range]``.  This provides
       a coarse safety net against extreme outliers but introduces a
       non-differentiable boundary at the clip edges.

    The transform is linear (affine), exactly invertible via
    :meth:`denormalize`, and has the lowest computational cost of the
    three normalizers in this module.

    .. note::
       The standard deviation is floored at a small epsilon (``1e-5`` for
       ``float32``, ``1e-14`` for ``float64``) to prevent division by zero
       for near-constant features.  Unlike :class:`SoftWinsorizedNormalizer`,
       there is no outlier-aware statistic — a single extreme value can
       inflate ``std`` and under-normalize the remaining data.

    Args:
        in_size (int): the size of the data that will be normalized.
        device (torch.device): the device in which the data will reside.
        dtype (torch.dtype): the data type to use for the normalizer.
        clip_range (float, optional): if set, clamps normalized z-scores to
            ``[-clip_range, clip_range]``.  ``None`` (default) disables clipping.
    """

    _STATS_FNAME = "env_stats.pt"
    _LEGACY_STATS_FNAME = "env_stats.pickle"

    #: When set, ``normalize()`` clamps the z-score output to
    #: ``[-clip_range, clip_range]`` after the standard normalization step.
    #: Useful for heavy-tailed feature distributions (e.g. angular-velocity
    #: outliers in NeuroBem) that would otherwise saturate downstream
    #: activation functions. Effective even with non-saturating activations
    #: such as GELU or LeakyReLU, where extreme values cause large gradient
    #: magnitudes and numerical instability.
    clip_range: Optional[float]

    def __init__(
        self,
        in_size: int,
        device: torch.device,
        dtype=torch.float32,
        clip_range: Optional[float] = None,
        strict_finite: bool = True,
        feature_dim_names: Optional[Sequence[str]] = None,
        normalize_dims: Union[bool, Sequence[bool], Mapping[str, bool]] = True,
    ):
        super().__init__()
        # RLRP-684 WS-C: when True, non-finite *input data* / *statistics*
        # (the ``normalize`` forward path and ``update_stats``) raise instead of
        # being silently clamped/zeroed. This is a fail-fast on genuine data or
        # statistic corruption; it does NOT govern ``denormalize`` (the model-
        # output inverse path), where early-training non-finiteness is tolerated.
        self.strict_finite: bool = strict_finite
        self.register_buffer("_mean", torch.zeros((1, in_size), dtype=dtype))
        self.register_buffer("_std", torch.ones((1, in_size), dtype=dtype))
        # Minimum std floor chosen relative to each dtype's machine epsilon:
        #   float32  machine eps ≈ 1.19e-7  →  eps = 1e-5  (~84× machine eps)
        #   float64  machine eps ≈ 2.22e-16 →  eps = 1e-14 (~45× machine eps)
        # The buffer is stored in the normalizer's own dtype for consistency.
        _eps_value = 1e-14 if dtype == torch.double else 1e-5
        self.register_buffer("eps", torch.tensor(_eps_value, dtype=dtype))
        self.clip_range: Optional[float] = clip_range
        # RLRP-736 S1.2a: per-feature-dim naming + enable/disable mask.
        self._setup_feature_dim_mask(in_size, feature_dim_names, normalize_dims)
        self.to(device)

    @property
    def device(self):
        return self._mean.device

    @property
    def mean(self):
        """Alias for compatibility with code expecting a ``mean`` attribute."""
        return self._mean

    @property
    def std(self):
        """Alias for compatibility with code expecting a ``std`` attribute."""
        return self._std

    def update_stats(self, data: mbrl.types.TensorType) -> None:
        """Updates the stored statistics using the given data.

        Equivalent to ``self.mean = data.mean(0)`` and ``self.std = data.std(0)``.

        Args:
            data (np.ndarray or torch.Tensor): The data used to compute the statistics.
        """
        # (CRITICAL) ToDo: assess support for model ensemble
        assert data.ndim == 2 and data.shape[1] == self._mean.shape[1]
        data = self._to_tensor(data)

        if data.shape[0] < 10:
            warnings.warn(
                f"ZScoreNormalizer.update_stats called with only {data.shape[0]} samples. "
                "Statistics may be unreliable.",
                RuntimeWarning,
            )

        if torch.isnan(data).any() or torch.isinf(data).any():
            if self.strict_finite:
                raise ValueError(
                    "ZScoreNormalizer.update_stats received data containing NaN "
                    "or Inf (strict_finite=True). Fix the upstream data pipeline, "
                    "or pass strict_finite=False to tolerate it (entries are then "
                    "replaced with zeros)."
                )
            warnings.warn(
                "ZScoreNormalizer.update_stats received data containing NaN or Inf. "
                "These entries will be replaced with zeros.",
                RuntimeWarning,
            )
            data = torch.where(torch.isfinite(data), data, torch.zeros_like(data))

        self._mean.copy_(data.mean(0, keepdim=True))
        if data.shape[0] > 1:
            self._std.copy_(data.std(0, keepdim=True))
        else:
            self._std.fill_(1.0)

        self._std.clamp_(min=self.eps.item())
        self._std[torch.isnan(self._std)] = 1.0

        return None

    @torch.compiler.disable
    def normalize(
        self,
        val: Union[float, mbrl.types.TensorType],
        strict_finite: Optional[bool] = None,
    ) -> torch.Tensor:
        """Normalizes the value according to the stored statistics.

        Equivalent to (val - mu) / sigma, where mu and sigma are the stored mean and
        standard deviation, respectively.

        The output tensor preserves the input dtype: if the caller passes a
        ``float32`` tensor the result is ``float32``, even when the normalizer
        stores its statistics in ``float64`` (and vice-versa).  Internal
        arithmetic is always performed at the higher of the two precisions to
        avoid unnecessary loss of significance.

        Args:
            val (float, np.ndarray or torch.Tensor): The value to normalize.
            strict_finite: optional per-call override of :attr:`strict_finite`
                (see :meth:`Normalizer.normalize`). ``None`` uses the instance
                setting. Note that a non-finite ``result`` can also arise from a
                *huge but finite* input overflowing ``(val - mu) / sigma`` (a
                diverged model-output prediction fed back as input); the tolerant
                path clamps it to the finite range instead of raising.

        Returns:
            (torch.Tensor): The normalized value.
        """
        _strict_finite = self.strict_finite if strict_finite is None else strict_finite
        val = self._to_tensor(val)
        input_dtype = val.dtype
        compute_dtype = (
            torch.float64
            if val.dtype == torch.float64 or self._mean.dtype == torch.float64
            else self._mean.dtype
        )
        result = (val.to(compute_dtype) - self._mean.to(compute_dtype)) / self._std.to(
            compute_dtype
        )
        if not torch.isfinite(result).all():
            non_finite_count = (~torch.isfinite(result)).sum().item()
            if _strict_finite:
                raise ValueError(
                    f"ZScoreNormalizer.normalize produced {non_finite_count} "
                    "non-finite values (strict_finite=True). This indicates "
                    "corrupt input data or degenerate statistics (mean/std). "
                    "Pass strict_finite=False to fall back to clamping."
                )
            warnings.warn(
                f"ZScoreNormalizer produced {non_finite_count} non-finite values. "
                "Clamping to finite data range. "
                "Check input data and normalizer statistics.",
                RuntimeWarning,
                stacklevel=2,
            )
            finite_mask = torch.isfinite(result)
            if finite_mask.any():
                lo = result[finite_mask].min()
                hi = result[finite_mask].max()
                result = result.clamp(lo, hi)
            else:
                result = torch.zeros_like(result)
        if self.clip_range is not None:
            result = result.clamp(-self.clip_range, self.clip_range)

        return self._apply_norm_mask(result.to(input_dtype), val)

    @torch.compiler.disable
    def denormalize(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        """De-normalizes the value according to the stored statistics.

        Equivalent to sigma * val + mu, where mu and sigma are the stored mean and
        standard deviation, respectively.

        The output tensor preserves the input dtype (see :meth:`normalize`).

        Args:
            val (float, np.ndarray or torch.Tensor): The value to de-normalize.

        Returns:
            (torch.Tensor): The de-normalized value.
        """
        val = self._to_tensor(val)
        input_dtype = val.dtype
        compute_dtype = (
            torch.float64
            if val.dtype == torch.float64 or self._mean.dtype == torch.float64
            else self._mean.dtype
        )
        result = self._std.to(compute_dtype) * val.to(compute_dtype) + self._mean.to(
            compute_dtype
        )
        if not torch.isfinite(result).all():
            non_finite_count = (~torch.isfinite(result)).sum().item()
            warnings.warn(
                f"ZScoreNormalizer produced {non_finite_count} non-finite values. "
                "Clamping to finite data range. "
                "Check input data and normalizer statistics.",
                RuntimeWarning,
                stacklevel=2,
            )
            finite_mask = torch.isfinite(result)
            if finite_mask.any():
                lo = result[finite_mask].min()
                hi = result[finite_mask].max()
                result = result.clamp(lo, hi)
            else:
                result = torch.zeros_like(result)

        return self._apply_norm_mask(result.to(input_dtype), val)

    def save(self, save_dir: Union[str, pathlib.Path]) -> None:
        """Saves statistics to a torch file."""
        save_dir = pathlib.Path(save_dir)
        torch.save(
            {"mean": self._mean.cpu(), "std": self._std.cpu(), "eps": self.eps.cpu()},
            save_dir / self._STATS_FNAME,
        )
        self._save_feature_mask(save_dir)

        return None

    def load(self, load_dir: Union[str, pathlib.Path]) -> None:
        """Loads statistics from a torch file, with legacy pickle fallback."""
        load_dir = pathlib.Path(load_dir)
        pt_path = load_dir / self._STATS_FNAME
        pickle_path = load_dir / self._LEGACY_STATS_FNAME

        if pt_path.exists():
            from mbrl.util.common import resolve_load_map_location

            stats = torch.load(pt_path, weights_only=True, map_location=resolve_load_map_location())
            self._mean.copy_(stats["mean"].to(self.device))
            self._std.copy_(stats["std"].to(self.device))
            if "eps" in stats:
                self.eps.copy_(stats["eps"].to(self.device))
        elif pickle_path.exists():
            # Support for mbrl-lib legacy pickle format
            warnings.warn(
                f"Loading normalizer from legacy pickle format "
                f"'{self._LEGACY_STATS_FNAME}'. Please re-save to migrate "
                f"to the new '{self._STATS_FNAME}' format.",
                FutureWarning,
            )
            import pickle

            with open(pickle_path, "rb") as f:
                stats = pickle.load(f)
                self._mean.copy_(torch.from_numpy(stats["mean"]).to(self.device))
                self._std.copy_(torch.from_numpy(stats["std"]).to(self.device))
        else:
            raise FileNotFoundError(
                f"No normalizer stats found at '{pt_path}' or '{pickle_path}'."
            )

        self._load_feature_mask(load_dir)

        return None


class SoftWinsorizedNormalizer(Normalizer):
    """Robust normalizer using winsorized z-score with per-`feature_dim`
    adaptive asinh soft-clipping.

    **What it does:**

    A two-stage normalization pipeline designed to handle heavy-tailed and
    outlier-contaminated feature distributions commonly encountered in robotic
    sensory data (e.g. angular-velocity spikes, contact-force transients).

    1. **Winsorized z-score** — Clamps each ``feature_dim`` to its
       ``[q_alpha, q_{1-alpha}]`` quantile range before computing mean and
       standard deviation, then standardizes.  This prevents extreme outliers
       from inflating the statistics.
    2. **Per-`feature_dim` adaptive asinh soft-clip** — Derives an automatic
       clip threshold per ``feature_dim`` from the interquartile range (IQR) as
       ``c_i = soft_clip_iqr_mult_i * IQR_i / sigma_i``, then smoothly
       compresses z-scores that exceed ``c_i`` via ``asinh``.  Values inside
       ``[-c_i, c_i]`` pass through untouched (identity region).

    The transform is differentiable everywhere, monotonic, and has an exact
    closed-form inverse (see :meth:`denormalize`).

    **Per-``feature_dim`` configuration.**  Both ``winsor_percentile`` and
    ``soft_clip_iqr_mult`` accept either:

    - a Python scalar (broadcast to every ``feature_dim`` — back-compatible
      default behaviour); ``None`` is additionally accepted for
      ``soft_clip_iqr_mult`` and disables the asinh stage on every dim,
      recovering the classic ``WinsorizedNormalizer``.
    - a :class:`~typing.Mapping` keyed by ``feature_dim`` names (as listed in
      ``feature_dim_names``), with strict set-equality validation; a ``None``
      value disables soft-clip on that single ``feature_dim``.  Requires
      ``feature_dim_names`` to be provided.
    - a :class:`~typing.Sequence` of length ``in_size`` (internal/test surface,
      not exposed in YAML).

    Mixing forms across the two arguments is allowed (e.g. scalar
    ``winsor_percentile`` + dict ``soft_clip_iqr_mult``).

    .. note::
       The per-``feature_dim`` clip threshold ``c_i`` is floored at ``1.0``
       during :meth:`update_stats` to guarantee that at least ±1 standard
       deviation of the z-score passes through the identity region.

    .. note::
       When every ``feature_dim`` has the same setting, an internal *uniform
       fast path* dispatches to the original scalar code path — zero
       runtime overhead vs. the pre-per-dim API.

    **Deviations from the Soft-Winsorization math formula.**

    The implementation faithfully follows the math of the specification
    (clamp at ``[Q_alpha, Q_{1-alpha}]`` → winsorized mean/std on the clamped
    data → z-score on the **raw** input → adaptive asinh soft-clip with
    ``c_i = tau · IQR_i / sigma_i`` where ``IQR_i = Q_{0.75} - Q_{0.25}``).
    Two intentional numerical-safety deviations are applied:

    1. ``eps`` is added inside the square root of the winsorized standard
       deviation (and ``sigma_i`` is then clamped to a minimum of ``eps``)
       to protect against zero-variance ``feature_dim`` columns (e.g.
       near-constant ``timestamps.delta_stamps``).  The impact on
       well-behaved channels is ``O(eps)``.
    2. The adaptive clip threshold ``c_i`` is floored at ``1.0`` so that
       the identity region of the soft-clip always covers at least ±1
       standard deviation.  This deviates from the bare formula
       ``c_i = tau · IQR_i / sigma_i`` only when ``tau · IQR_i < sigma_i``
       (i.e. very tight-body / heavy-tail ``feature_dim``).

    **Integration caveat — ``target_is_delta``.**

    This normalizer is designed for the *input/output normalization* setting
    described in the paper rationale: ``input -> normalize -> model ->
    denormalize -> output``.  Soft-clip composition is preserved exactly on
    the round-trip ``denormalize(normalize(x)) == x`` (within numerical
    precision).  However, when the downstream transition model uses
    ``target_is_delta=True`` and computes its regression target as
    ``normalize(next_obs) - normalize(obs)``, the bounded-target
    interpretation does **not** carry over: the difference of two soft-clipped
    z-scores is neither itself a soft-clipped z-score nor equal to
    ``normalize(next_obs - obs)``.  Training remains self-consistent (because
    prediction uses the same definition), but the per-``feature_dim``
    "bounded NLL" property documented above is no longer guaranteed for the
    delta target.  See :class:`mbrl.models.OneDTransitionRewardModelV2`
    which emits a :class:`RuntimeWarning` when both options are combined.

    Args:
        in_size: the size of the data that will be normalized.
        device: the device on which the data will reside.
        dtype: the data type to use for the normalizer.
        winsor_percentile: scalar fraction in ``(0, 0.5)`` or a per-``feature_dim``
            mapping / sequence (default ``0.05``).
        soft_clip_iqr_mult: scalar in ``(0, +inf)``, ``None`` to disable, or a
            per-``feature_dim`` mapping / sequence (default ``3.0``).
        feature_dim_names: ordered list of ``feature_dim`` names of length
            ``in_size``.  **Required** whenever a :class:`Mapping` is passed.
    """

    _STATS_FNAME = "winsorized_stats.pt"
    _NORMALIZER_FORMAT_VERSION = 2

    def __init__(
        self,
        in_size: int,
        device: torch.device,
        dtype=torch.float32,
        winsor_percentile: WinsorPercentileConfig = 0.05,
        soft_clip_iqr_mult: SoftClipIqrMultConfig = 3.0,
        feature_dim_names: Optional[Sequence[str]] = None,
        strict_finite: bool = True,
        normalize_dims: Union[bool, Sequence[bool], Mapping[str, bool]] = True,
    ):
        super().__init__()
        # RLRP-684 WS-C: fail-fast on non-finite *data* / *statistics* (the
        # ``normalize`` forward path + ``update_stats``); ``denormalize`` (the
        # model-output inverse) stays tolerant. See ZScoreNormalizer.strict_finite.
        self.strict_finite: bool = strict_finite
        self._in_size = in_size
        self._dtype = dtype
        self._feature_dim_names: Optional[List[str]] = (
            list(feature_dim_names) if feature_dim_names is not None else None
        )
        if self._feature_dim_names is not None and len(self._feature_dim_names) != in_size:
            raise ValueError(
                f"feature_dim_names has length {len(self._feature_dim_names)} "
                f"but in_size={in_size}."
            )

        # Resolve possibly-dict/sequence configs to per-dim float arrays
        # (NaN encodes 'disabled' for ``soft_clip_iqr_mult``).
        winsor_per_dim, winsor_is_scalar, winsor_scalar_val = self._resolve_winsor_percentile(
            winsor_percentile, in_size, self._feature_dim_names
        )
        soft_per_dim, soft_is_scalar, soft_scalar_val = self._resolve_soft_clip_iqr_mult(
            soft_clip_iqr_mult, in_size, self._feature_dim_names
        )

        self.register_buffer("winsorized_mean", torch.zeros((1, in_size), dtype=dtype))
        self.register_buffer("winsorized_std", torch.ones((1, in_size), dtype=dtype))
        self.register_buffer("q_low", torch.zeros((1, in_size), dtype=dtype))
        self.register_buffer("q_high", torch.zeros((1, in_size), dtype=dtype))
        self.register_buffer("iqr", torch.ones((1, in_size), dtype=dtype))

        # Per-`feature_dim` config buffers (shape (1, in_size)).  Buffers
        # auto-migrate to ``device`` via ``self.to(device)`` and are
        # included in ``state_dict``.
        self.register_buffer(
            "_winsor_percentile_per_dim",
            torch.from_numpy(winsor_per_dim).to(dtype=dtype),
        )
        # NaN in this buffer => soft-clip disabled on that dim.
        self.register_buffer(
            "_soft_clip_iqr_mult_per_dim",
            torch.from_numpy(soft_per_dim).to(dtype=dtype),
        )
        self.register_buffer(
            "_soft_clip_active_mask",
            ~torch.isnan(torch.from_numpy(soft_per_dim).to(dtype=dtype)),
        )
        # When a dim is disabled the ``clip_threshold`` entry stays at 0.
        self.register_buffer(
            "clip_threshold", torch.zeros((1, in_size), dtype=dtype)
        )
        _eps_value = 1e-14 if dtype == torch.double else 1e-5
        self.register_buffer("eps", torch.tensor(_eps_value, dtype=dtype))
        self.register_buffer(
            "_normalizer_format_version",
            torch.tensor(self._NORMALIZER_FORMAT_VERSION, dtype=torch.int64),
        )

        # User-facing echoes of the original inputs (back-compat — existing
        # tests assert ``norm.winsor_percentile == 0.05`` etc.).  For dict /
        # sequence input these become the resolved Python value rather than
        # a scalar.
        self.winsor_percentile: Union[float, Dict[str, float], List[float]] = (
            winsor_scalar_val
            if winsor_is_scalar
            else self._echo_user_input(winsor_percentile, in_size, self._feature_dim_names)
        )
        self.soft_clip_iqr_mult: Union[
            float, None, Dict[str, Optional[float]], List[Optional[float]]
        ] = (
            soft_scalar_val
            if soft_is_scalar
            else self._echo_user_input(soft_clip_iqr_mult, in_size, self._feature_dim_names)
        )

        # Fast-path flags (computed once, used in hot path).
        active_mask_np = ~np.isnan(soft_per_dim)
        self._soft_clip_all_disabled: bool = bool(not active_mask_np.any())
        self._soft_clip_all_enabled: bool = bool(active_mask_np.all())
        self._winsor_is_uniform: bool = bool(np.unique(winsor_per_dim).size == 1)
        # Uniform fast path = same scalar code path as today on the hot path.
        self._scalar_fast_path: bool = (
            self._winsor_is_uniform
            and (self._soft_clip_all_disabled or self._soft_clip_all_enabled)
            and (
                self._soft_clip_all_disabled
                or bool(np.unique(soft_per_dim[active_mask_np]).size == 1)
            )
        )

        # Floor warning — iterate per dim so the message quotes the
        # offending ``feature_dim`` name when available.
        _FLOOR_WARNING_THRESHOLD = 0.75
        for i in range(in_size):
            k = soft_per_dim[i]
            if not np.isnan(k) and k < _FLOOR_WARNING_THRESHOLD:
                name = (
                    self._feature_dim_names[i]
                    if self._feature_dim_names is not None
                    else f"dim[{i}]"
                )
                warnings.warn(
                    f"soft_clip_iqr_mult={k} for '{name}' is very low. "
                    f"For Gaussian-like features the per-`feature_dim` clip "
                    f"threshold (≈ {k} × IQR/σ ≈ {k * 1.35:.2f}) falls below "
                    f"the internal floor of 1.0, so the floor will silently "
                    f"dominate. Consider using a value ≥ 0.75.",
                    UserWarning,
                    stacklevel=2,
                )

        # RLRP-736 S1.2a: per-feature-dim naming + enable/disable mask.
        self._setup_feature_dim_mask(
            in_size, self._feature_dim_names, normalize_dims
        )
        self.to(device)

    # ------------------------------------------------------------------ #
    #  Per-`feature_dim` configuration resolution helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_winsor_percentile(
        value: WinsorPercentileConfig,
        in_size: int,
        feature_dim_names: Optional[List[str]],
    ) -> Tuple[np.ndarray, bool, float]:
        """Resolve ``winsor_percentile`` to a per-dim ``float64`` array.

        Returns ``(per_dim_array, is_scalar_input, scalar_value_or_nan)``.
        Validates bounds and (when dict) set-equality with
        ``feature_dim_names``.
        """
        if value is None:
            raise ValueError(
                "winsor_percentile=None is not supported (winsorization is "
                "always on); use a positive fraction in (0, 0.5)."
            )
        if isinstance(value, Mapping):
            arr = SoftWinsorizedNormalizer._mapping_to_array(
                dict(value),
                in_size,
                feature_dim_names,
                arg_name="winsor_percentile",
                allow_none_value=False,
            )
            is_scalar = False
            scalar_val = float("nan")
        elif isinstance(value, (list, tuple)):
            if len(value) != in_size:
                raise ValueError(
                    f"winsor_percentile sequence has length {len(value)} "
                    f"but in_size={in_size}."
                )
            arr = np.asarray([float(v) for v in value], dtype=np.float64)
            is_scalar = False
            scalar_val = float("nan")
        else:
            v = float(value)
            arr = np.full((in_size,), v, dtype=np.float64)
            is_scalar = True
            scalar_val = v

        if np.any(arr <= 0.0) or np.any(arr >= 0.5) or np.any(np.isnan(arr)):
            bad = np.where((arr <= 0.0) | (arr >= 0.5) | np.isnan(arr))[0]
            names = (
                [feature_dim_names[i] for i in bad]
                if feature_dim_names is not None
                else [f"dim[{i}]" for i in bad]
            )
            raise ValueError(
                f"winsor_percentile values must be in (0, 0.5); "
                f"offending entries: {dict(zip(names, arr[bad].tolist()))}"
            )
        return arr, is_scalar, scalar_val

    @staticmethod
    def _resolve_soft_clip_iqr_mult(
        value: SoftClipIqrMultConfig,
        in_size: int,
        feature_dim_names: Optional[List[str]],
    ) -> Tuple[np.ndarray, bool, Optional[float]]:
        """Resolve ``soft_clip_iqr_mult`` to a per-dim ``float64`` array.

        ``NaN`` encodes 'disabled' (recovers classic ``WinsorizedNormalizer``
        behaviour on that dim).  Returns
        ``(per_dim_array, is_scalar_input, scalar_value_or_None)``.
        """
        if value is None:
            arr = np.full((in_size,), np.nan, dtype=np.float64)
            return arr, True, None
        if isinstance(value, Mapping):
            arr = SoftWinsorizedNormalizer._mapping_to_array(
                dict(value),
                in_size,
                feature_dim_names,
                arg_name="soft_clip_iqr_mult",
                allow_none_value=True,
            )
            is_scalar = False
            scalar_val: Optional[float] = None
        elif isinstance(value, (list, tuple)):
            if len(value) != in_size:
                raise ValueError(
                    f"soft_clip_iqr_mult sequence has length {len(value)} "
                    f"but in_size={in_size}."
                )
            arr = np.asarray(
                [np.nan if v is None else float(v) for v in value], dtype=np.float64
            )
            is_scalar = False
            scalar_val = None
        else:
            v = float(value)
            arr = np.full((in_size,), v, dtype=np.float64)
            is_scalar = True
            scalar_val = v

        # Validate: every non-NaN entry must be > 0.
        active = ~np.isnan(arr)
        if np.any(active & (arr <= 0.0)):
            bad = np.where(active & (arr <= 0.0))[0]
            names = (
                [feature_dim_names[i] for i in bad]
                if feature_dim_names is not None
                else [f"dim[{i}]" for i in bad]
            )
            raise ValueError(
                f"soft_clip_iqr_mult values must be > 0 (or None to disable); "
                f"offending entries: {dict(zip(names, arr[bad].tolist()))}"
            )
        return arr, is_scalar, scalar_val

    @staticmethod
    def _mapping_to_array(
        mapping: Dict[str, Optional[float]],
        in_size: int,
        feature_dim_names: Optional[List[str]],
        arg_name: str,
        allow_none_value: bool,
    ) -> np.ndarray:
        """Resolve a ``feature_dim``-keyed mapping to a per-dim ``float64`` array.

        Strict set-equality validation with ``feature_dim_names`` (no missing
        keys, no extra keys); ``difflib`` typo suggestions on extras.
        """
        if feature_dim_names is None:
            raise ValueError(
                f"{arg_name} is a Mapping but `feature_dim_names` was not "
                f"provided to SoftWinsorizedNormalizer; pass the ordered "
                f"obs_dims + act_dims list from your simulator config."
            )
        if len(feature_dim_names) != in_size:
            raise ValueError(
                f"feature_dim_names has length {len(feature_dim_names)} but "
                f"in_size={in_size}."
            )
        expected = set(feature_dim_names)
        provided = set(mapping.keys())
        missing = sorted(expected - provided)
        extras = sorted(provided - expected)
        if missing or extras:
            parts = [f"{arg_name} mapping does not match feature_dim_names."]
            if missing:
                parts.append(f"  Missing keys: {missing}")
            if extras:
                suggestions = {
                    k: difflib.get_close_matches(k, feature_dim_names, n=1)
                    for k in extras
                }
                parts.append(
                    f"  Extra/unknown keys (with closest-match suggestions): "
                    f"{suggestions}"
                )
            raise ValueError("\n".join(parts))

        arr = np.empty((in_size,), dtype=np.float64)
        for i, name in enumerate(feature_dim_names):
            v = mapping[name]
            if v is None:
                if not allow_none_value:
                    raise ValueError(
                        f"{arg_name}['{name}'] is None, which is not allowed "
                        f"(only soft_clip_iqr_mult supports None per dim)."
                    )
                arr[i] = np.nan
            else:
                arr[i] = float(v)
        return arr

    @staticmethod
    def _echo_user_input(
        value, in_size: int, feature_dim_names: Optional[List[str]]
    ):
        """Return a plain-Python echo of the user's per-dim input (for repr/save)."""
        if isinstance(value, Mapping):
            return {str(k): (None if v is None else float(v)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [None if v is None else float(v) for v in value]
        return value

    @property
    def device(self):
        return self.winsorized_mean.device

    @property
    def mean(self):
        """Alias for compatibility with code expecting a ``mean`` attribute."""
        return self.winsorized_mean

    @property
    def std(self):
        """Alias for compatibility with code expecting a ``std`` attribute."""
        return self.winsorized_std

    def update_stats(self, data: mbrl.types.TensorType) -> None:
        """Compute winsorized statistics and adaptive clip thresholds from *data*.

        Statistics (quantiles, IQR, mean, std, clip thresholds) are computed on
        CPU via numpy regardless of the input device.  This avoids two issues:
        (1) materialising 5-6 full-dataset copies on the GPU (OOM risk), and
        (2) ``torch.quantile`` hard element-count limit (~16 M elements) which
        raises "quantile() input tensor is too large" for large datasets
        (e.g. 451 k timesteps × history_len 80 → 36 M rows).  numpy's
        ``np.quantile`` has no such restriction.
        The resulting statistics are transferred back to the target device via the
        ``self.xxx.copy_()`` calls below, which handle cross-device copies
        transparently.

        Args:
            data (np.ndarray or torch.Tensor): shape ``(N, in_size)``.
        """
        # (CRITICAL) ToDo: assess support for model ensemble
        assert data.ndim == 2 and data.shape[1] == self.winsorized_mean.shape[1]

        # Force CPU then numpy: torch.quantile is limited to ~16 M elements and
        # requires multiple full-size sorted copies on GPU simultaneously (OOM).
        # NOTE: Do NOT use self._to_tensor(data).cpu() here — _to_tensor moves data
        # to self.device (GPU) first, then .cpu() pulls it back, unnecessarily
        # allocating the entire dataset on GPU and causing GPU memory pressure / OOM
        # for large datasets (e.g. 451k timesteps × history_len 80 → ~36M elements).
        # Instead convert directly to a CPU tensor to avoid the GPU round-trip.
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).cpu()
        elif isinstance(data, torch.Tensor):
            data = data.cpu()
        else:
            data = torch.tensor(data).cpu()

        if data.shape[0] < 10:
            warnings.warn(
                f"SoftWinsorizedNormalizer.update_stats called with only {data.shape[0]} samples. "
                "Statistics may be unreliable.",
                RuntimeWarning,
            )

        if torch.isnan(data).any() or torch.isinf(data).any():
            if self.strict_finite:
                raise ValueError(
                    "SoftWinsorizedNormalizer.update_stats received data "
                    "containing NaN or Inf (strict_finite=True). Fix the upstream "
                    "data pipeline, or pass strict_finite=False to tolerate it "
                    "(entries are then replaced with zeros)."
                )
            warnings.warn(
                "SoftWinsorizedNormalizer.update_stats received data containing NaN or Inf. "
                "These entries will be replaced with zeros.",
                RuntimeWarning,
            )
            data = torch.where(torch.isfinite(data), data, torch.zeros_like(data))

        # Per-`feature_dim` alpha — use numpy for quantile computation:
        # no element-count limit (avoids torch.quantile's ~16 M-element
        # hard ceiling), no GPU memory pressure.  When every dim shares
        # the same alpha (uniform fast path) this collapses to a single
        # np.quantile call per tail, matching the pre-per-dim numerical
        # behaviour exactly.
        data_np = data.numpy()
        alpha_per_dim = self._winsor_percentile_per_dim.detach().cpu().numpy().reshape(-1)
        in_size = data_np.shape[1]

        q_low_np = np.empty((1, in_size), dtype=np.float64)
        q_high_np = np.empty((1, in_size), dtype=np.float64)
        unique_alphas, inv = np.unique(alpha_per_dim, return_inverse=True)
        for u_idx, a in enumerate(unique_alphas):
            cols = np.where(inv == u_idx)[0]
            sub = data_np[:, cols]
            q_low_np[0, cols] = np.quantile(sub, float(a), axis=0)
            q_high_np[0, cols] = np.quantile(sub, 1.0 - float(a), axis=0)
        # ``q25`` and ``q75`` are independent of ``winsor_percentile``;
        # one ``np.quantile`` call suffices for each, regardless of per-dim
        # configuration.
        q25_np = np.quantile(data_np, 0.25, axis=0, keepdims=True).astype(np.float64)
        q75_np = np.quantile(data_np, 0.75, axis=0, keepdims=True).astype(np.float64)

        q_low = torch.from_numpy(q_low_np)
        q_high = torch.from_numpy(q_high_np)
        q25 = torch.from_numpy(q25_np)
        q75 = torch.from_numpy(q75_np)

        self.q_low.copy_(q_low)
        self.q_high.copy_(q_high)
        iqr = q75 - q25
        self.iqr.copy_(iqr)

        # Winsorize: clamp to [q_low, q_high]
        clamped_data = data.clamp(min=q_low, max=q_high)

        # Winsorized mean
        w_mean = clamped_data.mean(0, keepdim=True)
        self.winsorized_mean.copy_(w_mean)

        # Winsorized std with Bessel's correction + epsilon
        if data.shape[0] > 1:
            w_std = torch.sqrt(
                ((clamped_data - w_mean) ** 2).sum(0, keepdim=True)
                / (data.shape[0] - 1)
                + self.eps.item()
            )
        else:
            w_std = torch.ones_like(w_mean)

        w_std.clamp_(min=self.eps.item())

        # (CRITICAL) ToDo: validate setting NaN to arbitrary value i.e., 1.0
        # ToDo: assess raising a warning when NaN is encountered and replaced
        w_std[torch.isnan(w_std)] = 1.0
        self.winsorized_std.copy_(w_std)

        # Per-`feature_dim` adaptive soft-clip threshold:
        #   c_i = soft_clip_iqr_mult_i * IQR_i / sigma_i
        # When ``soft_clip_iqr_mult_i is None`` (encoded as NaN in the buffer)
        # the soft-clip stage is disabled on that dim and the corresponding
        # ``clip_threshold`` entry is left at 0 (consumed only inside the
        # masked branch; see ``normalize`` for the branchless hot path).
        # NOTE (device safety): all upstream quantile/mean/std tensors
        # (`iqr`, `w_std`) live on CPU because the quantile path uses
        # ``np.quantile`` (see top-of-method rationale).  We must therefore
        # build ``clip_t`` on CPU as well — pulling ``_soft_clip_iqr_mult_per_dim``
        # and ``_soft_clip_active_mask`` to CPU here — otherwise the
        # multiplication mixes CUDA and CPU tensors and raises
        # ``RuntimeError: Expected all tensors to be on the same device``
        # on GPU runs (observed on JetsonAGX-Orin and Valeria HPC nodes).
        # The final ``self.clip_threshold.copy_(...)`` handles the
        # cross-device transfer back to the registered-buffer device.
        mask_cpu = self._soft_clip_active_mask.detach().cpu()
        # (1, in_size) — broadcast safe.
        soft_k = (
            self._soft_clip_iqr_mult_per_dim.detach()
            .cpu()
            .to(self.clip_threshold.dtype)
            .view(1, -1)
        )
        clip_t = soft_k * iqr.to(soft_k.dtype) / w_std.to(soft_k.dtype)
        # Floor at 1.0 so the identity region always covers at least ±1
        # sigma — unchanged semantics from the pre-per-dim API.
        clip_t = clip_t.clamp(min=1.0)
        # Replace NaN (disabled dims) with 0 so denormalize/normalize are
        # well-defined even when the masked branch reads the buffer.
        clip_t = torch.where(mask_cpu.view(1, -1), clip_t, torch.zeros_like(clip_t))
        self.clip_threshold.copy_(clip_t.to(self.clip_threshold.dtype))

        return None

    @staticmethod
    def _soft_clip(z: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        r"""Per-feature adaptive **asinh** soft-clip (RLRP-684 WS-D).

        Forward map ``S_τ`` with per-feature threshold ``τ = threshold``:

            S_τ(z) = z                                    if |z| <= τ
                   = sign(z) · ( τ + asinh(|z| − τ) )      if |z| >  τ

        with ``asinh(u) = ln(u + sqrt(u² + 1))``.  This replaces the previous
        **bounded** ``tanh`` tail, whose image was the band ``|y| ∈ [τ, τ+1)``
        so any model output with ``|y| >= τ+1`` had no pre-image and was
        silently saturated by the inverse ``clamp`` — i.e. the winsorized path
        was NOT invertible on unbounded model outputs.

        Properties of the ``asinh`` tail (see the module maths note):
          * **Global bijection ℝ→ℝ**: ``asinh`` maps ``[0,∞)→[0,∞)`` so the tail
            image is ``|y| ∈ [τ,∞)`` (surjective) and is strictly increasing
            (injective) ⇒ exactly invertible for ANY real output (no clamp).
          * **C¹ at the knee** ``|z|=τ``: ``asinh(0)=0`` and ``d/du asinh(u)→1``
            as ``u→0⁺``, matching the identity-region slope 1 (same smoothness
            class as the old ``tanh`` version).
          * **Logarithmic outlier compression** (``asinh(u)=ln(2u)+O(1/u²)``):
            heavy-tail compression is preserved, but WITHOUT a ceiling.
        """
        abs_z = z.abs()
        within = abs_z <= threshold
        excess = abs_z - threshold
        soft_clipped = z.sign() * (threshold + torch.asinh(excess))
        return torch.where(within, z, soft_clipped)

    @staticmethod
    def _soft_clip_inverse(y: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        r"""Exact analytic inverse of :meth:`_soft_clip` (RLRP-684 WS-D).

            S_τ⁻¹(y) = y                                   if |y| <= τ
                     = sign(y) · ( τ + sinh(|y| − τ) )      if |y| >  τ

        with ``sinh(v) = (eᵛ − e⁻ᵛ)/2``.  Because ``sinh(asinh(u)) = u`` for all
        ``u``, ``S_τ⁻¹(S_τ(z)) = z`` for every real ``z`` and ``S_τ(S_τ⁻¹(y)) = y``
        for every real ``y``.  Unlike the previous ``atanh`` inverse there is
        **no** ``clamp`` (``sinh`` is defined on all of ℝ), so denormalization is
        exact for any model output.  Extreme ``|y|−τ`` may legitimately overflow
        ``sinh`` to ±inf; that non-finite signal is surfaced by the WS-C
        telemetry rather than silently saturated.
        """
        abs_y = y.abs()
        within = abs_y <= threshold
        excess = abs_y - threshold
        soft_unclipped = y.sign() * (threshold + torch.sinh(excess))
        return torch.where(within, y, soft_unclipped)

    @torch.compiler.disable
    def normalize(
        self,
        val: Union[float, mbrl.types.TensorType],
        strict_finite: Optional[bool] = None,
    ) -> torch.Tensor:
        """Winsorized z-score followed by per-feature adaptive asinh soft-clip.

        Args:
            val: The value to normalize.
            strict_finite: optional per-call override of :attr:`strict_finite`
                (see :meth:`Normalizer.normalize`). ``None`` uses the instance
                setting; pass ``False`` for tolerant clamping of model-output
                (e.g. test-time-rollout feedback) values.

        Returns:
            The normalized value (same dtype as input).
        """
        _strict_finite = self.strict_finite if strict_finite is None else strict_finite
        val = self._to_tensor(val)
        input_dtype = val.dtype
        compute_dtype = (
            torch.float64
            if val.dtype == torch.float64 or self.winsorized_mean.dtype == torch.float64
            else self.winsorized_mean.dtype
        )

        z = (
            val.to(compute_dtype) - self.winsorized_mean.to(compute_dtype)
        ) / self.winsorized_std.to(compute_dtype)

        if not torch.isfinite(z).all():
            non_finite_count = (~torch.isfinite(z)).sum().item()
            if _strict_finite:
                raise ValueError(
                    f"SoftWinsorizedNormalizer.normalize produced "
                    f"{non_finite_count} non-finite values (strict_finite=True). "
                    "This indicates corrupt input data or degenerate winsorized "
                    "statistics. Pass strict_finite=False to fall back to clamping."
                )
            warnings.warn(
                f"SoftWinsorizedNormalizer produced {non_finite_count} non-finite values. "
                "Clamping to finite data range.",
                RuntimeWarning,
                stacklevel=2,
            )
            finite_mask = torch.isfinite(z)
            if finite_mask.any():
                lo = z[finite_mask].min()
                hi = z[finite_mask].max()
                z = z.clamp(lo, hi)
            else:
                z = torch.zeros_like(z)

        if self._soft_clip_all_disabled:
            result = z
        elif self._soft_clip_all_enabled:
            result = self._soft_clip(z, self.clip_threshold.to(compute_dtype))
        else:
            # Mixed config: compute the soft-clipped tensor, then keep the
            # raw z-score on disabled dims via a single branchless
            # ``torch.where``.  The disabled-dim ``clip_threshold`` entries
            # are zero but their soft-clipped output is discarded by the
            # mask, so no NaN ever propagates.
            z_soft = self._soft_clip(z, self.clip_threshold.to(compute_dtype))
            mask = self._soft_clip_active_mask.to(z.device)
            result = torch.where(mask, z_soft, z)
        return self._apply_norm_mask(result.to(input_dtype), val)

    @torch.compiler.disable
    def denormalize(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        """Exact inverse: invert soft-clip then invert z-score.

        Args:
            val: The normalized value to de-normalize.

        Returns:
            The de-normalized value (same dtype as input).
        """
        val = self._to_tensor(val)
        input_dtype = val.dtype
        compute_dtype = (
            torch.float64
            if val.dtype == torch.float64 or self.winsorized_mean.dtype == torch.float64
            else self.winsorized_mean.dtype
        )

        if self._soft_clip_all_disabled:
            z = val.to(compute_dtype)
        elif self._soft_clip_all_enabled:
            z = self._soft_clip_inverse(
                val.to(compute_dtype), self.clip_threshold.to(compute_dtype)
            )
        else:
            # Mixed config — branchless mask-select per dim.
            v_c = val.to(compute_dtype)
            z_inv = self._soft_clip_inverse(
                v_c, self.clip_threshold.to(compute_dtype)
            )
            mask = self._soft_clip_active_mask.to(v_c.device)
            z = torch.where(mask, z_inv, v_c)
        result = z * self.winsorized_std.to(compute_dtype) + self.winsorized_mean.to(
            compute_dtype
        )

        if not torch.isfinite(result).all():
            non_finite_count = (~torch.isfinite(result)).sum().item()
            warnings.warn(
                f"SoftWinsorizedNormalizer produced {non_finite_count} non-finite values. "
                "Clamping to finite data range.",
                RuntimeWarning,
                stacklevel=2,
            )
            finite_mask = torch.isfinite(result)
            if finite_mask.any():
                lo = result[finite_mask].min()
                hi = result[finite_mask].max()
                result = result.clamp(lo, hi)
            else:
                result = torch.zeros_like(result)

        return self._apply_norm_mask(result.to(input_dtype), val)

    def save(self, save_dir: Union[str, pathlib.Path]) -> None:
        save_dir = pathlib.Path(save_dir)
        torch.save(
            {
                "winsorized_mean": self.winsorized_mean.cpu(),
                "winsorized_std": self.winsorized_std.cpu(),
                "q_low": self.q_low.cpu(),
                "q_high": self.q_high.cpu(),
                "iqr": self.iqr.cpu(),
                "clip_threshold": self.clip_threshold.cpu(),
                "eps": self.eps.cpu(),
                # Per-`feature_dim` config (format version 2).
                "_winsor_percentile_per_dim": self._winsor_percentile_per_dim.cpu(),
                "_soft_clip_iqr_mult_per_dim": self._soft_clip_iqr_mult_per_dim.cpu(),
                "_soft_clip_active_mask": self._soft_clip_active_mask.cpu(),
                "_normalizer_format_version": int(self._NORMALIZER_FORMAT_VERSION),
                "feature_dim_names": self._feature_dim_names,
                # User-facing echoes (legacy + v2).
                "winsor_percentile": self.winsor_percentile,
                "soft_clip_iqr_mult": self.soft_clip_iqr_mult,
            },
            save_dir / self._STATS_FNAME,
        )
        self._save_feature_mask(save_dir)
        return None

    def load(self, load_dir: Union[str, pathlib.Path]) -> None:
        load_dir = pathlib.Path(load_dir)
        path = load_dir / self._STATS_FNAME
        if not path.exists():
            raise FileNotFoundError(f"No SoftWinsorizedNormalizer stats found at '{path}'.")
        from mbrl.util.common import resolve_load_map_location

        # ``weights_only=False`` is required because v2 checkpoints contain
        # plain Python objects (the user-facing ``winsor_percentile`` echo
        # may be a ``dict`` or ``list``; ``feature_dim_names`` is a list).
        # The file is produced by this codebase and is never user-supplied
        # at load time.
        stats = torch.load(
            path, weights_only=False, map_location=resolve_load_map_location()
        )
        self.winsorized_mean.copy_(stats["winsorized_mean"].to(self.device))
        self.winsorized_std.copy_(stats["winsorized_std"].to(self.device))
        self.q_low.copy_(stats["q_low"].to(self.device))
        self.q_high.copy_(stats["q_high"].to(self.device))
        self.iqr.copy_(stats["iqr"].to(self.device))
        self.clip_threshold.copy_(stats["clip_threshold"].to(self.device))
        if "eps" in stats:
            self.eps.copy_(stats["eps"].to(self.device))

        # ---- Per-`feature_dim` config migration ----
        in_size = self.winsorized_mean.shape[1]
        if "_winsor_percentile_per_dim" in stats:
            # v2 checkpoint — load buffers directly.
            self._winsor_percentile_per_dim.copy_(
                stats["_winsor_percentile_per_dim"].to(self.device)
            )
            self._soft_clip_iqr_mult_per_dim.copy_(
                stats["_soft_clip_iqr_mult_per_dim"].to(self.device)
            )
            self._soft_clip_active_mask.copy_(
                stats["_soft_clip_active_mask"].to(self.device)
            )
        else:
            # Legacy (v1, 0-D scalars) — broadcast to (in_size,) per-dim
            # buffers and emit a ``DeprecationWarning``.
            warnings.warn(
                "Loading legacy SoftWinsorizedNormalizer checkpoint without "
                "per-`feature_dim` config buffers; broadcasting scalar "
                "winsor_percentile / soft_clip_iqr_mult to all dims.",
                DeprecationWarning,
                stacklevel=2,
            )
            legacy_w = stats.get("winsor_percentile", 0.05)
            legacy_s = stats.get("soft_clip_iqr_mult", 3.0)
            w_dtype = self._winsor_percentile_per_dim.dtype
            self._winsor_percentile_per_dim.copy_(
                torch.full((in_size,), float(legacy_w), dtype=w_dtype).to(self.device)
            )
            if legacy_s is None:
                self._soft_clip_iqr_mult_per_dim.copy_(
                    torch.full((in_size,), float("nan"), dtype=w_dtype).to(self.device)
                )
                self._soft_clip_active_mask.copy_(
                    torch.zeros((in_size,), dtype=torch.bool).to(self.device)
                )
            else:
                self._soft_clip_iqr_mult_per_dim.copy_(
                    torch.full((in_size,), float(legacy_s), dtype=w_dtype).to(self.device)
                )
                self._soft_clip_active_mask.copy_(
                    torch.ones((in_size,), dtype=torch.bool).to(self.device)
                )

        if "winsor_percentile" in stats:
            self.winsor_percentile = stats["winsor_percentile"]
        if "soft_clip_iqr_mult" in stats:
            self.soft_clip_iqr_mult = stats["soft_clip_iqr_mult"]

        self._load_feature_mask(load_dir)

        # Recompute fast-path flags from the loaded buffers so the hot path
        # reflects the loaded configuration.
        active_np = self._soft_clip_active_mask.detach().cpu().numpy().reshape(-1)
        soft_np = self._soft_clip_iqr_mult_per_dim.detach().cpu().numpy().reshape(-1)
        winsor_np = self._winsor_percentile_per_dim.detach().cpu().numpy().reshape(-1)
        self._soft_clip_all_disabled = bool(not active_np.any())
        self._soft_clip_all_enabled = bool(active_np.all())
        self._winsor_is_uniform = bool(np.unique(winsor_np).size == 1)
        self._scalar_fast_path = (
            self._winsor_is_uniform
            and (self._soft_clip_all_disabled or self._soft_clip_all_enabled)
            and (
                self._soft_clip_all_disabled
                or bool(np.unique(soft_np[active_np]).size == 1)
            )
        )
        return None


class QuantileNormalizer(Normalizer):
    """Robust normalizer using empirical CDF mapping to standard normal (quantile normalization).

    **What it does:**

    A non-parametric normalization pipeline that maps each feature through
    its empirical cumulative distribution function (CDF) to produce a
    standard-normal output, regardless of the original distribution shape.

    1. **Empirical CDF construction** — During :meth:`update_stats`, the
       observed data for each feature is sorted and divided into ``n_bins``
       equally-spaced probability bins.  The resulting quantile boundaries
       define a piecewise-linear mapping from raw values to uniform ``[0, 1]``
       probabilities.
    2. **Probit (inverse-normal) transform** — The uniform CDF value is
       converted to a standard-normal score via the inverse error function
       (``erfinv``), yielding an output that is approximately ``N(0, 1)``
       for any input distribution.
    3. **Linear tail extrapolation** — Values beyond the observed range are
       extrapolated using the slope of the outermost bin, preserving
       ordering and avoiding hard saturation.

    Uses ``torch.searchsorted`` for efficient O(log n_bins) bin lookup and
    linear interpolation between adjacent quantile boundaries.
    The transform is monotonic, differentiable almost everywhere, and
    invertible via :meth:`denormalize` (reverse lookup + interpolation).

    .. note::
       Quantile normalization is the most outlier-robust normalizer in this
       module but requires storing per-feature quantile boundaries
       (``n_bins + 1`` values per feature).  For very small calibration
       datasets (< 500 samples), consider lowering ``n_bins`` to
       approximately ``N / 2`` to avoid under-populated bins that degrade
       the CDF estimate.

    Args:
        in_size (int): the size of the data that will be normalized.
        device (torch.device): the device in which the data will reside.
        dtype (torch.dtype): the data type to use for the normalizer.
        n_bins (int): number of quantile bins (default 1000).  Higher values
            improve resolution for large datasets; lower values prevent
            under-populated bins for small datasets.
        tail_policy (str): ``"linear"`` (default) — extrapolates beyond the
            observed range using the slope of the outermost bin.
    """

    _STATS_FNAME = "quantile_stats.pt"

    def __init__(
        self,
        in_size: int,
        device: torch.device,
        dtype=torch.float32,
        n_bins: int = 1000,
        tail_policy: str = "linear",
        strict_finite: bool = True,
        feature_dim_names: Optional[Sequence[str]] = None,
        normalize_dims: Union[bool, Sequence[bool], Mapping[str, bool]] = True,
    ):
        super().__init__()
        # RLRP-684 WS-C: fail-fast on non-finite *data* in ``update_stats``;
        # ``denormalize`` (the model-output inverse) stays tolerant.
        self.strict_finite: bool = strict_finite
        assert tail_policy in ("linear",), f"Unsupported tail_policy: {tail_policy}"
        self.n_bins = n_bins
        self.tail_policy = tail_policy

        # quantile_boundaries: (n_bins+1, in_size) — per-feature empirical quantile boundaries
        self.register_buffer(
            "quantile_boundaries",
            torch.zeros((n_bins + 1, in_size), dtype=dtype),
        )
        # target_quantiles: (n_bins+1,) — standard normal target values
        eps_clamp = 1e-7
        p = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
        p_clamped = p.clamp(eps_clamp, 1.0 - eps_clamp)
        targets = torch.erfinv(2.0 * p_clamped - 1.0) * (2.0**0.5)
        self.register_buffer("target_quantiles", targets.to(dtype))

        # For mean/std compatibility aliases, compute after update_stats
        self.register_buffer("_mean_cache", torch.zeros((1, in_size), dtype=dtype))
        self.register_buffer("_std_cache", torch.ones((1, in_size), dtype=dtype))

        _eps_value = 1e-14 if dtype == torch.double else 1e-5
        self.register_buffer("eps", torch.tensor(_eps_value, dtype=dtype))
        # RLRP-736 S1.2a: per-feature-dim naming + enable/disable mask.
        self._setup_feature_dim_mask(in_size, feature_dim_names, normalize_dims)
        self.to(device)

    @property
    def device(self):
        return self.quantile_boundaries.device

    @property
    def mean(self):
        """Alias: median of data (approximated as the middle quantile boundary)."""
        return self._mean_cache

    @property
    def std(self):
        """Alias: IQR-based scale estimate."""
        return self._std_cache

    def update_stats(self, data: mbrl.types.TensorType):
        """Compute empirical quantile boundaries per feature.

        Args:
            data (np.ndarray or torch.Tensor): shape ``(N, in_size)``.
        """
        # (CRITICAL) ToDo: assess support for model ensemble
        in_size = self.quantile_boundaries.shape[1]
        assert data.ndim == 2 and data.shape[1] == in_size
        data = self._to_tensor(data)

        if data.shape[0] < 10:
            warnings.warn(
                f"QuantileNormalizer.update_stats called with only {data.shape[0]} samples. "
                "Statistics may be unreliable.",
                RuntimeWarning,
            )

        if torch.isnan(data).any() or torch.isinf(data).any():
            if self.strict_finite:
                raise ValueError(
                    "QuantileNormalizer.update_stats received data containing NaN "
                    "or Inf (strict_finite=True). Fix the upstream data pipeline, "
                    "or pass strict_finite=False to tolerate it (entries are then "
                    "replaced with zeros)."
                )
            warnings.warn(
                "QuantileNormalizer.update_stats received data containing NaN or Inf. "
                "These entries will be replaced with zeros.",
                RuntimeWarning,
            )
            data = torch.where(torch.isfinite(data), data, torch.zeros_like(data))

        # Probability grid
        p = torch.linspace(
            0.0, 1.0, self.n_bins + 1, dtype=data.dtype, device=data.device
        )

        # Per-feature quantile boundaries
        boundaries = torch.quantile(data, p, dim=0)  # (n_bins+1, in_size)

        # RLRP-684 WS-B (B3): ``normalize`` / ``denormalize`` rely on
        # ``torch.searchsorted`` over these boundaries, which requires them to be
        # monotone non-decreasing along the bin axis (per feature).  Empirical
        # quantiles are sorted by construction, but assert it explicitly so any
        # future regression (e.g. NaN-poisoned data slipping through) fails fast
        # with a clear message instead of returning silently wrong indices.
        min_diff = (boundaries[1:] - boundaries[:-1]).min().item()
        if min_diff < 0.0:
            raise ValueError(
                "QuantileNormalizer.update_stats produced non-monotone quantile "
                f"boundaries (min adjacent diff = {min_diff}). searchsorted "
                "requires monotone non-decreasing boundaries; check the input "
                "data for NaN/Inf or degenerate features."
            )
        self.quantile_boundaries.copy_(boundaries)

        # Update compatibility caches
        mid_idx = self.n_bins // 2
        self._mean_cache.copy_(boundaries[mid_idx].unsqueeze(0))
        q25_idx = self.n_bins // 4
        q75_idx = (3 * self.n_bins) // 4
        iqr = boundaries[q75_idx] - boundaries[q25_idx]
        iqr = iqr.clamp(min=self.eps.item())
        self._std_cache.copy_(iqr.unsqueeze(0))

    @torch.compiler.disable
    def normalize(
        self,
        val: Union[float, mbrl.types.TensorType],
        strict_finite: Optional[bool] = None,
    ) -> torch.Tensor:
        """Map values through empirical CDF to standard normal via linear interpolation.

        Args:
            val: The value to normalize.
            strict_finite: accepted for interface parity with
                :meth:`Normalizer.normalize`. The quantile mapping is bounded by
                construction (searchsorted + clamped linear interp/extrapolation),
                so it neither overflows nor raises on non-finite inputs; the flag
                is therefore inert here and kept only for a uniform call contract.

        Returns:
            The normalized value (same dtype as input).
        """
        del strict_finite  # inert: quantile mapping is bounded, cannot overflow/raise
        val = self._to_tensor(val)
        input_dtype = val.dtype
        compute_dtype = (
            torch.float64
            if val.dtype == torch.float64
            or self.quantile_boundaries.dtype == torch.float64
            else self.quantile_boundaries.dtype
        )
        val_c = val.to(compute_dtype)
        original_shape = val_c.shape
        # Flatten to 2D: (batch, in_size)
        if val_c.ndim == 1:
            val_c = val_c.unsqueeze(0)
        if val_c.ndim > 2:
            val_c = val_c.reshape(-1, val_c.shape[-1])

        boundaries = self.quantile_boundaries.to(compute_dtype)  # (K+1, d)
        targets = self.target_quantiles.to(compute_dtype)  # (K+1,)

        # Transpose boundaries to (d, K+1) for searchsorted along last dim
        boundaries_t = boundaries.t().contiguous()  # (d, K+1)
        val_t = val_c.t().contiguous()  # (d, batch)

        # searchsorted: find bin index k such that boundaries[k-1] <= val < boundaries[k]
        idx = torch.searchsorted(boundaries_t, val_t, right=False)  # (d, batch)
        idx = idx.clamp(1, self.n_bins)  # ensure valid interpolation range

        # Gather lower and upper boundaries
        idx_low = (idx - 1).clamp(0, self.n_bins)
        b_low = torch.gather(boundaries_t, 1, idx_low)  # (d, batch)
        b_high = torch.gather(boundaries_t, 1, idx)  # (d, batch)

        # Target quantiles for interpolation
        t_low = targets[idx_low.clamp(0, self.n_bins)]  # broadcast per-feature
        t_high = targets[idx.clamp(0, self.n_bins)]

        # Linear interpolation
        denom = (b_high - b_low).clamp(min=1e-12)
        frac = (val_t - b_low) / denom
        result_t = t_low + frac * (t_high - t_low)

        # Tail extrapolation (linear)
        lower_mask = val_t < boundaries_t[:, :1]
        upper_mask = val_t > boundaries_t[:, -1:]
        if lower_mask.any():
            slope_low = (targets[1] - targets[0]) / (
                boundaries_t[:, 1:2] - boundaries_t[:, 0:1]
            ).clamp(min=1e-12)
            extrap_low = targets[0] + slope_low * (val_t - boundaries_t[:, 0:1])
            result_t = torch.where(lower_mask, extrap_low, result_t)
        if upper_mask.any():
            slope_high = (targets[-1] - targets[-2]) / (
                boundaries_t[:, -1:] - boundaries_t[:, -2:-1]
            ).clamp(min=1e-12)
            extrap_high = targets[-1] + slope_high * (val_t - boundaries_t[:, -1:])
            result_t = torch.where(upper_mask, extrap_high, result_t)

        result = result_t.t().reshape(original_shape)  # (batch, d) then reshape
        return self._apply_norm_mask(result.to(input_dtype), val)

    @torch.compiler.disable
    def denormalize(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        """Inverse: map standard normal values back to data space via linear interpolation.

        Args:
            val: The normalized value to de-normalize.

        Returns:
            The de-normalized value (same dtype as input).
        """
        val = self._to_tensor(val)
        input_dtype = val.dtype
        compute_dtype = (
            torch.float64
            if val.dtype == torch.float64
            or self.quantile_boundaries.dtype == torch.float64
            else self.quantile_boundaries.dtype
        )
        val_c = val.to(compute_dtype)
        original_shape = val_c.shape
        if val_c.ndim == 1:
            val_c = val_c.unsqueeze(0)
        if val_c.ndim > 2:
            val_c = val_c.reshape(-1, val_c.shape[-1])

        boundaries = self.quantile_boundaries.to(compute_dtype)  # (K+1, d)
        targets = self.target_quantiles.to(compute_dtype)  # (K+1,)

        batch_size = val_c.shape[0]
        in_size = val_c.shape[1]

        # Search in target_quantiles (sorted 1D) for each value
        targets_sorted = targets.contiguous()
        # Expand val_c for searchsorted: (batch * in_size,) searched against (K+1,)
        val_flat = val_c.reshape(-1)  # (batch * in_size,)
        idx_flat = torch.searchsorted(targets_sorted, val_flat, right=False)
        idx_flat = idx_flat.clamp(1, self.n_bins)
        idx = idx_flat.reshape(batch_size, in_size)  # (batch, in_size)

        idx_low = (idx - 1).clamp(0, self.n_bins)

        t_low = targets[idx_low]  # (batch, in_size)
        t_high = targets[idx]

        # Gather boundaries per feature
        # boundaries shape: (K+1, d), we need to gather along dim=0 for each feature
        b_low = torch.gather(boundaries, 0, idx_low)  # (batch, in_size)
        b_high = torch.gather(boundaries, 0, idx)

        denom = (t_high - t_low).clamp(min=1e-12)
        frac = (val_c - t_low) / denom
        result = b_low + frac * (b_high - b_low)

        # Tail extrapolation
        lower_mask = val_c < targets[0]
        upper_mask = val_c > targets[-1]
        if lower_mask.any():
            slope_low = (boundaries[1] - boundaries[0]) / (
                targets[1] - targets[0]
            ).clamp(min=1e-12)
            extrap_low = boundaries[0] + slope_low * (val_c - targets[0])
            result = torch.where(lower_mask, extrap_low, result)
        if upper_mask.any():
            slope_high = (boundaries[-1] - boundaries[-2]) / (
                targets[-1] - targets[-2]
            ).clamp(min=1e-12)
            extrap_high = boundaries[-1] + slope_high * (val_c - targets[-1])
            result = torch.where(upper_mask, extrap_high, result)

        result = result.reshape(original_shape)
        return self._apply_norm_mask(result.to(input_dtype), val)

    def save(self, save_dir: Union[str, pathlib.Path]):
        save_dir = pathlib.Path(save_dir)
        torch.save(
            {
                "quantile_boundaries": self.quantile_boundaries.cpu(),
                "target_quantiles": self.target_quantiles.cpu(),
                "_mean_cache": self._mean_cache.cpu(),
                "_std_cache": self._std_cache.cpu(),
                "eps": self.eps.cpu(),
                "n_bins": self.n_bins,
                "tail_policy": self.tail_policy,
            },
            save_dir / self._STATS_FNAME,
        )
        self._save_feature_mask(save_dir)

    def load(self, load_dir: Union[str, pathlib.Path]):
        load_dir = pathlib.Path(load_dir)
        path = load_dir / self._STATS_FNAME
        if not path.exists():
            raise FileNotFoundError(f"No QuantileNormalizer stats found at '{path}'.")
        from mbrl.util.common import resolve_load_map_location

        stats = torch.load(path, weights_only=True, map_location=resolve_load_map_location())
        self.quantile_boundaries.copy_(stats["quantile_boundaries"].to(self.device))
        self.target_quantiles.copy_(stats["target_quantiles"].to(self.device))
        if "_mean_cache" in stats:
            self._mean_cache.copy_(stats["_mean_cache"].to(self.device))
        if "_std_cache" in stats:
            self._std_cache.copy_(stats["_std_cache"].to(self.device))
        if "eps" in stats:
            self.eps.copy_(stats["eps"].to(self.device))
        if "n_bins" in stats:
            self.n_bins = stats["n_bins"]
        if "tail_policy" in stats:
            self.tail_policy = stats["tail_policy"]

        self._load_feature_mask(load_dir)


class StrategyAwareNormalizer(Normalizer):
    """Thin wrapper adding per-dimension *strategy* transforms over a base.

    Introduced by stage 1 (action S1.2b) of the Per-Environment Feature
    Handling ``.junie`` plan
    (``rlrp-736-per-environment-feature-handling-plan-20260711.md``,
    YouTrack RLRP-736).

    The base normalizer (any concrete :class:`Normalizer`) handles the
    ``standard`` / ``winsorized`` / ``quantile`` statistics and the S1.2a
    per-dimension enable/disable mask.  This wrapper layers **block strategies**
    that cannot be expressed as an independent per-dimension scalar transform —
    currently only ``unit_norm``: contiguous runs of ``unit_norm`` dimensions
    (e.g. the 4-D quaternion attitude block) are re-projected onto the unit
    L2-sphere *after* the base transform, on both :meth:`normalize` and
    :meth:`denormalize`.

    Ordering contract: the ``unit_norm`` block is expected to be **disabled** in
    the base normalizer's S1.2a mask (``normalize_dims=False`` for those dims),
    so the base passes the raw block through untouched and this wrapper performs
    the unit projection.  ``identity`` strategy dims are handled entirely by the
    base mask and need no wrapper — hence a ``StrategyAwareNormalizer`` is only
    constructed when at least one ``unit_norm`` dimension is present.

    Back-compat: with no ``unit_norm`` dimension the wrapper is never created, so
    every legacy path is byte-identical.
    """

    #: File name persisting the resolved strategy alongside the base stats.
    _STRATEGY_FNAME = "norm_strategy.pt"
    _UNIT_NORM = "unit_norm"

    def __init__(self, base: Normalizer, strategy: Sequence[str]):
        super().__init__()
        self.base = base
        self._strategy: List[str] = [str(s) for s in strategy]
        self._unit_norm_blocks: List[Tuple[int, int]] = self._resolve_unit_blocks(
            self._strategy
        )
        self.register_buffer(
            "eps", torch.tensor(1e-12, dtype=torch.float32), persistent=False
        )

    @staticmethod
    def _resolve_unit_blocks(strategy: Sequence[str]) -> List[Tuple[int, int]]:
        """Return contiguous ``[start, stop)`` index runs of ``unit_norm`` dims."""
        blocks: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for idx, strat in enumerate(strategy):
            if strat == StrategyAwareNormalizer._UNIT_NORM:
                if start is None:
                    start = idx
            elif start is not None:
                blocks.append((start, idx))
                start = None
        if start is not None:
            blocks.append((start, len(strategy)))
        return blocks

    def _apply_unit_norm(self, val: torch.Tensor) -> torch.Tensor:
        if not self._unit_norm_blocks:
            return val
        if not torch.is_tensor(val):
            val = torch.as_tensor(val)
        eps = self.eps.to(val.dtype)
        # Reassemble the last dimension out of contiguous segments, projecting
        # only the ``unit_norm`` blocks onto the L2 unit sphere. We build the
        # result via ``torch.cat`` of freshly-computed slices instead of an
        # in-place write into a cloned tensor: an in-place assignment such as
        # ``out[..., start:stop] = block / norm`` mutates a strided view that is
        # also read on the RHS, which breaks autograd during backprop
        # ("one of the variables needed for gradient computation has been
        # modified by an inplace operation ... AsStridedBackward0"). RLRP-736.
        dim = val.shape[-1]
        segments: List[torch.Tensor] = []
        cursor = 0
        for start, stop in self._unit_norm_blocks:
            if start > cursor:
                segments.append(val[..., cursor:start])
            block = val[..., start:stop]
            norm = torch.linalg.norm(block, dim=-1, keepdim=True).clamp_min(eps)
            segments.append(block / norm)
            cursor = stop
        if cursor < dim:
            segments.append(val[..., cursor:dim])
        return torch.cat(segments, dim=-1)

    # --- delegated statistics ------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def mean(self) -> torch.Tensor:
        return self.base.mean

    @property
    def std(self) -> torch.Tensor:
        return self.base.std

    @property
    def strict_finite(self) -> bool:
        return getattr(self.base, "strict_finite", True)

    def update_stats(self, data: mbrl.types.TensorType) -> None:
        self.base.update_stats(data)

    def normalize(
        self,
        val: Union[float, mbrl.types.TensorType],
        strict_finite: Optional[bool] = None,
    ) -> torch.Tensor:
        out = self.base.normalize(val, strict_finite=strict_finite)
        return self._apply_unit_norm(out)

    def denormalize(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        out = self.base.denormalize(val)
        return self._apply_unit_norm(out)

    def save(self, save_dir: Union[str, pathlib.Path]) -> None:
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.base.save(save_dir)
        torch.save({"strategy": self._strategy}, save_dir / self._STRATEGY_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]) -> None:
        load_dir = pathlib.Path(load_dir)
        self.base.load(load_dir)
        path = load_dir / self._STRATEGY_FNAME
        if path.exists():
            payload = torch.load(path, weights_only=False, map_location="cpu")
            self._strategy = [str(s) for s in payload["strategy"]]
            self._unit_norm_blocks = self._resolve_unit_blocks(self._strategy)


def create_normalizer(
    normalizer_type: str,
    in_size: int,
    device: torch.device,
    dtype=torch.float32,
    **kwargs,
) -> Normalizer:
    """Factory function to create a normalizer by type string.

    Args:
        normalizer_type: ``"standard"``, ``"standard_symmetric"``, ``"winsorized"``,
            or ``"quantile"``. ``"standard_symmetric"`` builds a plain
            :class:`ZScoreNormalizer` just like ``"standard"`` — the distinction is
            handled by :class:`~mbrl.models.OneDTransitionRewardModel`, which uses the
            block-shared (input *and* output) facade for ``"standard_symmetric"`` and the
            single input-only facade for ``"standard"``.
        in_size: feature dimension.
        device: torch device.
        dtype: torch dtype.
        **kwargs: forwarded to the chosen normalizer constructor
            (e.g. ``winsor_percentile``, ``soft_clip_iqr_mult``, ``n_bins``, ``tail_policy``).

    Returns:
        A normalizer instance (``ZScoreNormalizer``, ``SoftWinsorizedNormalizer``,
        or ``QuantileNormalizer``).
    """
    # RLRP-684 WS-C: forward the fail-fast-on-non-finite policy to every type.
    strict_finite = kwargs.get("strict_finite", True)
    if normalizer_type in ("standard", "standard_symmetric"):
        clip_range = kwargs.get("clip_range", None)
        return ZScoreNormalizer(
            in_size,
            device,
            dtype=dtype,
            clip_range=clip_range,
            strict_finite=strict_finite,
            feature_dim_names=kwargs.get("feature_dim_names", None),
            normalize_dims=kwargs.get("normalize_dims", True),
        )
    elif normalizer_type == "winsorized":
        return SoftWinsorizedNormalizer(
            in_size,
            device,
            dtype=dtype,
            winsor_percentile=kwargs.get("winsor_percentile", 0.05),
            soft_clip_iqr_mult=kwargs.get("soft_clip_iqr_mult", 3.0),
            feature_dim_names=kwargs.get("feature_dim_names", None),
            strict_finite=strict_finite,
            normalize_dims=kwargs.get("normalize_dims", True),
        )
    elif normalizer_type == "quantile":
        return QuantileNormalizer(
            in_size,
            device,
            dtype=dtype,
            n_bins=kwargs.get("n_bins", 1000),
            tail_policy=kwargs.get("tail_policy", "linear"),
            strict_finite=strict_finite,
            feature_dim_names=kwargs.get("feature_dim_names", None),
            normalize_dims=kwargs.get("normalize_dims", True),
        )
    else:
        raise ValueError(
            f"Unknown normalizer_type '{normalizer_type}'. "
            "Choose from 'standard', 'standard_symmetric', 'winsorized', 'quantile'."
        )
