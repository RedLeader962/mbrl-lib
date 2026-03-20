# coding=utf-8
"""Normalization utilities for model-based reinforcement learning.

This module provides three normalizer classes with increasing outlier
robustness.  All share the same public API (``update_stats``, ``normalize``,
``denormalize``, ``save``/``load``) and can be instantiated through the
:func:`create_normalizer` factory.

**Quick comparison of normalizers:**

- ``Normalizer`` — Standard running-mean z-score.  No outlier protection.
  Differentiable, linearly invertible.  Cost: O(N).
- ``WinsorizedNormalizer`` — Winsorized z-score + adaptive tanh soft-clip.
  Moderate outlier robustness.  Differentiable, closed-form invertible.
  Cost: O(N log N).
- ``QuantileNormalizer`` — Empirical-CDF mapping to standard normal.
  Strong outlier robustness.  Differentiable, invertible via lookup +
  interpolation.  Cost: O(N log N).

**Rule of thumb — choosing a normalizer:**

- Use ``Normalizer`` when data is well-behaved (roughly Gaussian, no extreme
  outliers) and you want the fastest, simplest option.
- Use ``WinsorizedNormalizer`` when data has moderate outliers or heavy tails
  but the core distribution is roughly symmetric.  Good default choice for
  robotic state/action spaces.
- Use ``QuantileNormalizer`` when the distribution is highly skewed,
  multi-modal, or when outlier robustness is critical and you can afford
  storing per-feature quantile boundaries.

**Rule of thumb — parameter configuration:**

*Normalizer*

- ``clip_range`` (default ``None``): Leave at ``None`` for well-behaved data.
  Set to ``5.0``–``10.0`` when occasional spikes can produce z-scores that
  destabilize downstream layers.  A tighter range (e.g. ``3.0``) aggressively
  truncates tails and may discard useful signal.

*WinsorizedNormalizer*

- ``winsor_percentile`` (default ``0.05``): Fraction of each tail clamped
  before computing mean/std.  ``0.05`` (5 %) suits most robotic data.
  Lower it to ``0.01``–``0.02`` if outliers are rare; raise to ``0.10`` for
  heavily contaminated streams.
- ``soft_clip_iqr_mult`` (default ``3.0``): IQR multiplier for the per-feature
  tanh clip threshold (``c_i ≈ mult × IQR/σ ≈ mult × 1.35`` for Gaussian
  data).  ``3.0`` keeps ≈ ±4 σ in the identity region — a safe default.
  Reduce to ``1.5``–``2.0`` for aggressive tail compression; values below
  ``0.75`` hit the internal floor of 1.0 and trigger a warning.

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
import pathlib
import warnings
from typing import Optional, Union

import mbrl.types
import numpy as np
import torch
import torch.compiler
import torch.nn


def create_normalizer(
    normalizer_type: str,
    in_size: int,
    device: torch.device,
    dtype=torch.float32,
    **kwargs,
) -> torch.nn.Module:
    """Factory function to create a normalizer by type string.

    Args:
        normalizer_type: ``"standard"``, ``"winsorized"``, or ``"quantile"``.
        in_size: feature dimension.
        device: torch device.
        dtype: torch dtype.
        **kwargs: forwarded to the chosen normalizer constructor
            (e.g. ``winsor_percentile``, ``soft_clip_iqr_mult``, ``n_bins``, ``tail_policy``).

    Returns:
        A normalizer instance (``Normalizer``, ``WinsorizedNormalizer``, or ``QuantileNormalizer``).
    """
    if normalizer_type == "standard":
        clip_range = kwargs.get("clip_range", None)
        return Normalizer(in_size, device, dtype=dtype, clip_range=clip_range)
    elif normalizer_type == "winsorized":
        return WinsorizedNormalizer(
            in_size,
            device,
            dtype=dtype,
            winsor_percentile=kwargs.get("winsor_percentile", 0.05),
            soft_clip_iqr_mult=kwargs.get("soft_clip_iqr_mult", 3.0),
        )
    elif normalizer_type == "quantile":
        return QuantileNormalizer(
            in_size,
            device,
            dtype=dtype,
            n_bins=kwargs.get("n_bins", 1000),
            tail_policy=kwargs.get("tail_policy", "linear"),
        )
    else:
        raise ValueError(
            f"Unknown normalizer_type '{normalizer_type}'. "
            "Choose from 'standard', 'winsorized', 'quantile'."
        )


class Normalizer(torch.nn.Module):
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
       for near-constant features.  Unlike :class:`WinsorizedNormalizer`,
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
    ):
        super().__init__()
        self.register_buffer("mean", torch.zeros((1, in_size), dtype=dtype))
        self.register_buffer("std", torch.ones((1, in_size), dtype=dtype))
        # Minimum std floor chosen relative to each dtype's machine epsilon:
        #   float32  machine eps ≈ 1.19e-7  →  eps = 1e-5  (~84× machine eps)
        #   float64  machine eps ≈ 2.22e-16 →  eps = 1e-14 (~45× machine eps)
        # The buffer is stored in the normalizer's own dtype for consistency.
        _eps_value = 1e-14 if dtype == torch.double else 1e-5
        self.register_buffer("eps", torch.tensor(_eps_value, dtype=dtype))
        self.clip_range: Optional[float] = clip_range
        self.to(device)

    @property
    def device(self):
        return self.mean.device

    def _to_tensor(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        """Convert input to a tensor on the correct device, handling MPS dtype."""
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        if self.device.type == "mps" and val.dtype == torch.float64:
            val = val.float()
        return val.to(self.device)

    def update_stats(self, data: mbrl.types.TensorType):
        """Updates the stored statistics using the given data.

        Equivalent to ``self.mean = data.mean(0)`` and ``self.std = data.std(0)``.

        Args:
            data (np.ndarray or torch.Tensor): The data used to compute the statistics.
        """
        assert data.ndim == 2 and data.shape[1] == self.mean.shape[1]
        data = self._to_tensor(data)

        if data.shape[0] < 10:
            warnings.warn(
                f"Normalizer.update_stats called with only {data.shape[0]} samples. "
                "Statistics may be unreliable.",
                RuntimeWarning,
            )

        if torch.isnan(data).any() or torch.isinf(data).any():
            warnings.warn(
                "Normalizer.update_stats received data containing NaN or Inf. "
                "These entries will be replaced with zeros.",
                RuntimeWarning,
            )
            data = torch.where(torch.isfinite(data), data, torch.zeros_like(data))

        self.mean.copy_(data.mean(0, keepdim=True))
        if data.shape[0] > 1:
            self.std.copy_(data.std(0, keepdim=True))
        else:
            self.std.fill_(1.0)

        self.std.clamp_(min=self.eps.item())
        self.std[torch.isnan(self.std)] = 1.0

    @torch.compiler.disable
    def normalize(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
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

        Returns:
            (torch.Tensor): The normalized value.
        """
        val = self._to_tensor(val)
        input_dtype = val.dtype
        compute_dtype = (
            torch.float64
            if val.dtype == torch.float64 or self.mean.dtype == torch.float64
            else self.mean.dtype
        )
        result = (val.to(compute_dtype) - self.mean.to(compute_dtype)) / self.std.to(
            compute_dtype
        )
        if not torch.isfinite(result).all():
            non_finite_count = (~torch.isfinite(result)).sum().item()
            warnings.warn(
                f"Normalizer produced {non_finite_count} non-finite values. "
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
        return result.to(input_dtype)

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
            if val.dtype == torch.float64 or self.mean.dtype == torch.float64
            else self.mean.dtype
        )
        result = self.std.to(compute_dtype) * val.to(compute_dtype) + self.mean.to(
            compute_dtype
        )
        if not torch.isfinite(result).all():
            non_finite_count = (~torch.isfinite(result)).sum().item()
            warnings.warn(
                f"Normalizer produced {non_finite_count} non-finite values. "
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
        return result.to(input_dtype)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves statistics to a torch file."""
        save_dir = pathlib.Path(save_dir)
        torch.save(
            {"mean": self.mean.cpu(), "std": self.std.cpu(), "eps": self.eps.cpu()},
            save_dir / self._STATS_FNAME,
        )

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads statistics from a torch file, with legacy pickle fallback."""
        load_dir = pathlib.Path(load_dir)
        pt_path = load_dir / self._STATS_FNAME
        pickle_path = load_dir / self._LEGACY_STATS_FNAME

        if pt_path.exists():
            stats = torch.load(pt_path, weights_only=True)
            self.mean.copy_(stats["mean"].to(self.device))
            self.std.copy_(stats["std"].to(self.device))
            if "eps" in stats:
                self.eps.copy_(stats["eps"].to(self.device))
        elif pickle_path.exists():
            warnings.warn(
                f"Loading normalizer from legacy pickle format "
                f"'{self._LEGACY_STATS_FNAME}'. Please re-save to migrate "
                f"to the new '{self._STATS_FNAME}' format.",
                FutureWarning,
            )
            import pickle

            with open(pickle_path, "rb") as f:
                stats = pickle.load(f)
                self.mean.copy_(torch.from_numpy(stats["mean"]).to(self.device))
                self.std.copy_(torch.from_numpy(stats["std"]).to(self.device))
        else:
            raise FileNotFoundError(
                f"No normalizer stats found at '{pt_path}' or '{pickle_path}'."
            )


class WinsorizedNormalizer(torch.nn.Module):
    """Robust normalizer using winsorized z-score with per-feature adaptive tanh soft-clipping.

    **What it does:**

    A two-stage normalization pipeline designed to handle heavy-tailed and
    outlier-contaminated feature distributions commonly encountered in robotic
    sensory data (e.g. angular-velocity spikes, contact-force transients).

    1. **Winsorized z-score** — Clamps each feature to its ``[q_alpha, q_{1-alpha}]``
       quantile range before computing mean and standard deviation, then standardizes.
       This prevents extreme outliers from inflating the statistics.
    2. **Per-feature adaptive tanh soft-clip** — Derives an automatic clip threshold
       per feature from the interquartile range (IQR) as
       ``c_i = soft_clip_iqr_mult * IQR_i / sigma_i``, then smoothly compresses
       z-scores that exceed ``c_i`` via ``tanh``.  Values inside ``[-c_i, c_i]``
       pass through untouched (identity region).

    The transform is differentiable everywhere, monotonic, and has an exact
    closed-form inverse (see :meth:`denormalize`).

    .. note::
       The per-feature clip threshold ``c_i`` is floored at ``1.0`` during
       :meth:`update_stats` to guarantee that at least ±1 standard deviation
       of the z-score passes through the identity region.  If your
       ``soft_clip_iqr_mult`` is low enough that the computed ``c_i`` would
       fall below 1.0 for Gaussian-like features (roughly when
       ``soft_clip_iqr_mult < 0.75``), a warning is emitted at construction
       time because the floor will silently override the requested
       aggressiveness.

    Args:
        in_size (int): the size of the data that will be normalized.
        device (torch.device): the device in which the data will reside.
        dtype (torch.dtype): the data type to use for the normalizer.
        winsor_percentile (float): the percentile for winsorization (default 0.05).
        soft_clip_iqr_mult (float): IQR multiplier for the adaptive clip threshold
            (default 3.0).  For Gaussian data the resulting threshold is approximately
            ``soft_clip_iqr_mult * 1.35`` standard deviations.  Values below ~0.75 will
            cause the internal floor of 1.0 to dominate, effectively ignoring the
            requested multiplier.
    """

    _STATS_FNAME = "winsorized_stats.pt"

    def __init__(
        self,
        in_size: int,
        device: torch.device,
        dtype=torch.float32,
        winsor_percentile: float = 0.05,
        soft_clip_iqr_mult: float = 3.0,
    ):
        super().__init__()
        self.register_buffer("winsorized_mean", torch.zeros((1, in_size), dtype=dtype))
        self.register_buffer("winsorized_std", torch.ones((1, in_size), dtype=dtype))
        self.register_buffer("q_low", torch.zeros((1, in_size), dtype=dtype))
        self.register_buffer("q_high", torch.zeros((1, in_size), dtype=dtype))
        self.register_buffer("iqr", torch.ones((1, in_size), dtype=dtype))
        self.register_buffer("clip_threshold", torch.full((1, in_size), soft_clip_iqr_mult, dtype=dtype))
        _eps_value = 1e-14 if dtype == torch.double else 1e-5
        self.register_buffer("eps", torch.tensor(_eps_value, dtype=dtype))
        self.winsor_percentile = winsor_percentile
        self.soft_clip_iqr_mult = soft_clip_iqr_mult

        # Warn when the multiplier is so low that the internal floor (1.0)
        # will override the user's setting for Gaussian-like features.
        # For a Gaussian, IQR / sigma ≈ 1.35, so clip_t ≈ mult * 1.35.
        # The floor kicks in when mult * 1.35 < 1.0, i.e. mult < ~0.74.
        _FLOOR_WARNING_THRESHOLD = 0.75
        if soft_clip_iqr_mult < _FLOOR_WARNING_THRESHOLD:
            warnings.warn(
                f"soft_clip_iqr_mult={soft_clip_iqr_mult} is very low. "
                f"For Gaussian-like features the per-feature clip threshold "
                f"(≈ {soft_clip_iqr_mult} × IQR/σ ≈ {soft_clip_iqr_mult * 1.35:.2f}) "
                f"falls below the internal floor of 1.0, so the floor will "
                f"silently dominate. Consider using a value ≥ 0.75.",
                UserWarning,
                stacklevel=2,
            )

        self.to(device)

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

    def _to_tensor(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        if self.device.type == "mps" and val.dtype == torch.float64:
            val = val.float()
        return val.to(self.device)

    def update_stats(self, data: mbrl.types.TensorType):
        """Compute winsorized statistics and adaptive clip thresholds from *data*.

        Args:
            data (np.ndarray or torch.Tensor): shape ``(N, in_size)``.
        """
        assert data.ndim == 2 and data.shape[1] == self.winsorized_mean.shape[1]
        data = self._to_tensor(data)

        if data.shape[0] < 10:
            warnings.warn(
                f"WinsorizedNormalizer.update_stats called with only {data.shape[0]} samples. "
                "Statistics may be unreliable.",
                RuntimeWarning,
            )

        if torch.isnan(data).any() or torch.isinf(data).any():
            warnings.warn(
                "WinsorizedNormalizer.update_stats received data containing NaN or Inf. "
                "These entries will be replaced with zeros.",
                RuntimeWarning,
            )
            data = torch.where(torch.isfinite(data), data, torch.zeros_like(data))

        alpha = self.winsor_percentile
        # Compute quantiles per feature
        q_low = torch.quantile(data, alpha, dim=0, keepdim=True)
        q_high = torch.quantile(data, 1.0 - alpha, dim=0, keepdim=True)
        q25 = torch.quantile(data, 0.25, dim=0, keepdim=True)
        q75 = torch.quantile(data, 0.75, dim=0, keepdim=True)

        self.q_low.copy_(q_low)
        self.q_high.copy_(q_high)
        iqr = q75 - q25
        self.iqr.copy_(iqr)

        # Winsorize: clamp to [q_low, q_high]
        clamped = data.clamp(min=q_low, max=q_high)

        # Winsorized mean
        w_mean = clamped.mean(0, keepdim=True)
        self.winsorized_mean.copy_(w_mean)

        # Winsorized std with Bessel's correction + epsilon
        if data.shape[0] > 1:
            w_std = torch.sqrt(
                ((clamped - w_mean) ** 2).sum(0, keepdim=True) / (data.shape[0] - 1)
                + self.eps
            )
        else:
            w_std = torch.ones_like(w_mean)
        w_std.clamp_(min=self.eps.item())
        w_std[torch.isnan(w_std)] = 1.0
        self.winsorized_std.copy_(w_std)

        # Per-feature adaptive soft-clip threshold: c_i = gamma * IQR_i / sigma_i
        clip_t = self.soft_clip_iqr_mult * iqr / w_std
        # Floor at 1.0 so the identity region always covers at least ±1 sigma.
        # Without this floor, near-constant features (IQR ≈ 0) or very low
        # soft_clip_iqr_mult values would cause the tanh soft-clip to compress
        # even the core of the distribution, distorting well-behaved data.
        clip_t = clip_t.clamp(min=1.0)
        self.clip_threshold.copy_(clip_t)

    @staticmethod
    def _soft_clip(z: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """Per-feature adaptive tanh soft-clip."""
        abs_z = z.abs()
        within = abs_z <= threshold
        excess = abs_z - threshold
        clipped = z.sign() * (threshold + torch.tanh(excess))
        return torch.where(within, z, clipped)

    @staticmethod
    def _soft_clip_inverse(y: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """Exact inverse of per-feature adaptive tanh soft-clip."""
        abs_y = y.abs()
        within = abs_y <= threshold
        excess = (abs_y - threshold).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        unclipped = y.sign() * (threshold + torch.atanh(excess))
        return torch.where(within, y, unclipped)

    @torch.compiler.disable
    def normalize(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        """Winsorized z-score followed by per-feature adaptive tanh soft-clip.

        Args:
            val: The value to normalize.

        Returns:
            The normalized value (same dtype as input).
        """
        val = self._to_tensor(val)
        input_dtype = val.dtype
        compute_dtype = (
            torch.float64
            if val.dtype == torch.float64 or self.winsorized_mean.dtype == torch.float64
            else self.winsorized_mean.dtype
        )
        z = (val.to(compute_dtype) - self.winsorized_mean.to(compute_dtype)) / self.winsorized_std.to(compute_dtype)

        if not torch.isfinite(z).all():
            non_finite_count = (~torch.isfinite(z)).sum().item()
            warnings.warn(
                f"WinsorizedNormalizer produced {non_finite_count} non-finite values. "
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

        result = self._soft_clip(z, self.clip_threshold.to(compute_dtype))
        return result.to(input_dtype)

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
        z = self._soft_clip_inverse(val.to(compute_dtype), self.clip_threshold.to(compute_dtype))
        result = z * self.winsorized_std.to(compute_dtype) + self.winsorized_mean.to(compute_dtype)

        if not torch.isfinite(result).all():
            non_finite_count = (~torch.isfinite(result)).sum().item()
            warnings.warn(
                f"WinsorizedNormalizer produced {non_finite_count} non-finite values. "
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
        return result.to(input_dtype)

    def save(self, save_dir: Union[str, pathlib.Path]):
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
                "winsor_percentile": self.winsor_percentile,
                "soft_clip_iqr_mult": self.soft_clip_iqr_mult,
            },
            save_dir / self._STATS_FNAME,
        )

    def load(self, load_dir: Union[str, pathlib.Path]):
        load_dir = pathlib.Path(load_dir)
        path = load_dir / self._STATS_FNAME
        if not path.exists():
            raise FileNotFoundError(f"No WinsorizedNormalizer stats found at '{path}'.")
        stats = torch.load(path, weights_only=True)
        self.winsorized_mean.copy_(stats["winsorized_mean"].to(self.device))
        self.winsorized_std.copy_(stats["winsorized_std"].to(self.device))
        self.q_low.copy_(stats["q_low"].to(self.device))
        self.q_high.copy_(stats["q_high"].to(self.device))
        self.iqr.copy_(stats["iqr"].to(self.device))
        self.clip_threshold.copy_(stats["clip_threshold"].to(self.device))
        if "eps" in stats:
            self.eps.copy_(stats["eps"].to(self.device))
        if "winsor_percentile" in stats:
            self.winsor_percentile = stats["winsor_percentile"]
        if "soft_clip_iqr_mult" in stats:
            self.soft_clip_iqr_mult = stats["soft_clip_iqr_mult"]


class QuantileNormalizer(torch.nn.Module):
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
    ):
        super().__init__()
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
        targets = torch.erfinv(2.0 * p_clamped - 1.0) * (2.0 ** 0.5)
        self.register_buffer("target_quantiles", targets.to(dtype))

        # For mean/std compatibility aliases, compute after update_stats
        self.register_buffer("_mean_cache", torch.zeros((1, in_size), dtype=dtype))
        self.register_buffer("_std_cache", torch.ones((1, in_size), dtype=dtype))

        _eps_value = 1e-14 if dtype == torch.double else 1e-5
        self.register_buffer("eps", torch.tensor(_eps_value, dtype=dtype))
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

    def _to_tensor(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        if self.device.type == "mps" and val.dtype == torch.float64:
            val = val.float()
        return val.to(self.device)

    def update_stats(self, data: mbrl.types.TensorType):
        """Compute empirical quantile boundaries per feature.

        Args:
            data (np.ndarray or torch.Tensor): shape ``(N, in_size)``.
        """
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
            warnings.warn(
                "QuantileNormalizer.update_stats received data containing NaN or Inf. "
                "These entries will be replaced with zeros.",
                RuntimeWarning,
            )
            data = torch.where(torch.isfinite(data), data, torch.zeros_like(data))

        # Probability grid
        p = torch.linspace(0.0, 1.0, self.n_bins + 1, dtype=data.dtype, device=data.device)

        # Per-feature quantile boundaries
        boundaries = torch.quantile(data, p, dim=0)  # (n_bins+1, in_size)
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
    def normalize(self, val: Union[float, mbrl.types.TensorType]) -> torch.Tensor:
        """Map values through empirical CDF to standard normal via linear interpolation.

        Args:
            val: The value to normalize.

        Returns:
            The normalized value (same dtype as input).
        """
        val = self._to_tensor(val)
        input_dtype = val.dtype
        compute_dtype = (
            torch.float64
            if val.dtype == torch.float64 or self.quantile_boundaries.dtype == torch.float64
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
            slope_low = (targets[1] - targets[0]) / (boundaries_t[:, 1:2] - boundaries_t[:, 0:1]).clamp(min=1e-12)
            extrap_low = targets[0] + slope_low * (val_t - boundaries_t[:, 0:1])
            result_t = torch.where(lower_mask, extrap_low, result_t)
        if upper_mask.any():
            slope_high = (targets[-1] - targets[-2]) / (boundaries_t[:, -1:] - boundaries_t[:, -2:-1]).clamp(min=1e-12)
            extrap_high = targets[-1] + slope_high * (val_t - boundaries_t[:, -1:])
            result_t = torch.where(upper_mask, extrap_high, result_t)

        result = result_t.t().reshape(original_shape)  # (batch, d) then reshape
        return result.to(input_dtype)

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
            if val.dtype == torch.float64 or self.quantile_boundaries.dtype == torch.float64
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
            slope_low = (boundaries[1] - boundaries[0]) / (targets[1] - targets[0]).clamp(min=1e-12)
            extrap_low = boundaries[0] + slope_low * (val_c - targets[0])
            result = torch.where(lower_mask, extrap_low, result)
        if upper_mask.any():
            slope_high = (boundaries[-1] - boundaries[-2]) / (targets[-1] - targets[-2]).clamp(min=1e-12)
            extrap_high = boundaries[-1] + slope_high * (val_c - targets[-1])
            result = torch.where(upper_mask, extrap_high, result)

        result = result.reshape(original_shape)
        return result.to(input_dtype)

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

    def load(self, load_dir: Union[str, pathlib.Path]):
        load_dir = pathlib.Path(load_dir)
        path = load_dir / self._STATS_FNAME
        if not path.exists():
            raise FileNotFoundError(f"No QuantileNormalizer stats found at '{path}'.")
        stats = torch.load(path, weights_only=True)
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
