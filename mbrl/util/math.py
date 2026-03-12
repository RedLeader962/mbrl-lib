# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import warnings
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions
import torch.fft
import torch.nn.functional as F
from packaging import version

import mbrl.types


def truncated_linear(
    min_x: float, max_x: float, min_y: float, max_y: float, x: float
) -> float:
    """Truncated linear function.

    Implements the following function:
        f1(x) = min_y + (x - min_x) / (max_x - min_x) * (max_y - min_y)
        f(x) = min(max_y, max(min_y, f1(x)))

    If max_x - min_x < 1e-10, then it behaves as the constant f(x) = max_y
    """
    if max_x - min_x < 1e-10:
        return max_y
    if x <= min_x:
        y: float = min_y
    else:
        dx = (x - min_x) / (max_x - min_x)
        dx = min(dx, 1.0)
        y = dx * (max_y - min_y) + min_y
    return y


def gaussian_nll(
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """Negative log-likelihood for Gaussian distribution

    Args:
        pred_mean (tensor): the predicted mean.
        pred_logvar (tensor): the predicted log variance.
        target (tensor): the target value.
        reduce (bool): if ``False`` the loss is returned w/o reducing.
            Defaults to ``True``.

    Returns:
        (tensor): the negative log-likelihood.
    """
    l2 = F.mse_loss(pred_mean, target, reduction="none")
    inv_var = (-pred_logvar).exp()
    losses = l2 * inv_var + pred_logvar
    if reduce:
        return losses.sum(dim=1).mean()
    return losses


# inplace truncated normal function for pytorch.
# credit to https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/model.py#L64
def truncated_normal_(
    tensor: torch.Tensor, mean: float = 0, std: float = 1
) -> torch.Tensor:
    """Samples from a truncated normal distribution in-place.

    Args:
        tensor (tensor): the tensor in which sampled values will be stored.
        mean (float): the desired mean (default = 0).
        std (float): the desired standard deviation (default = 1).

    Returns:
        (tensor): the tensor with the stored values. Note that this modifies the input tensor
            in place, so this is just a pointer to the same object.
    """
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
        bound_violations = torch.sum(cond).item()
        if bound_violations == 0:
            break
        tensor[cond] = torch.normal(
            mean, std, size=(bound_violations,), device=tensor.device
        )
    return tensor


class Normalizer(torch.nn.Module):
    """Class that keeps a running mean and variance and normalizes data accordingly.

    The statistics are stored as registered buffers, ensuring they are
    included in ``state_dict()`` and correctly moved between devices.

    Args:
        in_size (int): the size of the data that will be normalized.
        device (torch.device): the device in which the data will reside.
        dtype (torch.dtype): the data type to use for the normalizer.
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

    Two-stage pipeline:
    1. **Winsorized z-score**: clamp data to [q_alpha, q_{1-alpha}] per feature before computing
       mean/std, then standardize.
    2. **Per-feature adaptive tanh soft-clip**: derive an automatic clip threshold from the IQR
       and apply a smooth tanh compression beyond it.

    The transform is differentiable everywhere, monotonic, and has an exact closed-form inverse.

    Args:
        in_size (int): the size of the data that will be normalized.
        device (torch.device): the device in which the data will reside.
        dtype (torch.dtype): the data type to use for the normalizer.
        winsor_percentile (float): the percentile for winsorization (default 0.05).
        soft_clip_iqr_mult (float): IQR multiplier for the adaptive clip threshold (default 3.0).
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
        # Ensure minimum threshold of 1.0 to avoid degenerate clipping
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

    Maps each feature's values through its empirical CDF to produce a standard-normal output.
    Uses ``torch.searchsorted`` for efficient bin lookup and linear interpolation between
    adjacent quantile boundaries.

    Args:
        in_size (int): the size of the data that will be normalized.
        device (torch.device): the device in which the data will reside.
        dtype (torch.dtype): the data type to use for the normalizer.
        n_bins (int): number of quantile bins (default 1000).
        tail_policy (str): ``"linear"`` (default) — extend slope of outermost bin.
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


# ------------------------------------------------------------------------ #
# Uncertainty propagation functions
# ------------------------------------------------------------------------ #
def propagate_from_indices(
    predicted_tensor: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Propagates ensemble outputs using the given indices.

    Args:
        predicted_tensor (tensor): the prediction to propagate. Shape must
            be ``E x B x Od``, where ``E``, ``B``, and ``Od`` represent the
            number of models, batch size, and output dimension, respectively.
        indices (tensor): the model indices to choose.

    Returns:
        (tensor): the chosen prediction, so that
            `output[i, :] = predicted_tensor[indices[i], i, :]`.
    """
    return predicted_tensor[indices, torch.arange(predicted_tensor.shape[1]), :]


def propagate_random_model(
    predictions: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, ...]:
    """Propagates ensemble outputs by choosing a random model.

    Args:
        predictions (tuple of tensors): the predictions to propagate. Each tensor's
            shape must be ``E x B x Od``, where ``E``, ``B``, and ``Od`` represent the
            number of models, batch size, and output dimension, respectively.

    Returns:
        (tuple of tensors): the chosen predictions, so that
            `output[k][i, :] = predictions[k][random_choice, i, :]`.
    """
    output: List[torch.Tensor] = []
    for i, predicted_tensor in enumerate(predictions):
        assert predicted_tensor.ndim == 3
        num_models, batch_size, pred_dim = predicted_tensor.shape
        model_indices = torch.randint(
            num_models, size=(batch_size,), device=predicted_tensor.device
        )
        output.append(propagate_from_indices(predicted_tensor, model_indices))
    return tuple(output)


def propagate_expectation(
    predictions: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, ...]:
    """Propagates ensemble outputs by taking expectation over model predictions.

    Args:
        predictions (tuple of tensors): the predictions to propagate. Each tensor's
            shape must be ``E x B x Od``, where ``E``, ``B``, and ``Od`` represent the
            number of models, batch size, and output dimension, respectively.

    Returns:
        (tuple of tensors): the chosen predictions, so that
            `output[k][i, :] = predictions[k].mean(dim=0)`
    """
    output: List[torch.Tensor] = []
    for i, predicted_tensor in enumerate(predictions):
        assert predicted_tensor.ndim == 3
        output.append(predicted_tensor.mean(dim=0))
    return tuple(output)


def propagate_fixed_model(
    predictions: Tuple[torch.Tensor, ...], propagation_indices: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    """Propagates ensemble outputs by taking expectation over model predictions.

    Args:
        predictions (tuple of tensors): the predictions to propagate. Each tensor's
            shape must be ``E x B x Od``, where ``E``, ``B``, and ``Od`` represent the
            number of models, batch size, and output dimension, respectively.
        propagation_indices (tensor): the model indices to choose (will use the same for all
            predictions).

    Returns:
        (tuple of tensors): the chosen predictions, so that
            `output[k][i, :] = predictions[k].mean(dim=0)`
    """
    output: List[torch.Tensor] = []
    for i, predicted_tensor in enumerate(predictions):
        assert predicted_tensor.ndim == 3
        output.append(propagate_from_indices(predicted_tensor, propagation_indices))
    return tuple(output)


def propagate(
    predictions: Tuple[torch.Tensor, ...],
    propagation_method: str = "expectation",
    propagation_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """Propagates ensemble outputs according to desired method.

    Implements propagations options as described in  Chua et al., NeurIPS 2018 paper (PETS)
    https://arxiv.org/pdf/1805.12114.pdf

    Valid propagation options are:

        - "random_model": equivalent to :meth:`propagate_random_model`.
          This corresponds to TS1 propagation in the PETS paper.
        - "fixed_model": equivalent to :meth:`propagate_fixed_model`.
          This can be used to implement TSinf propagation, described in the PETS paper.
        - "expectation": equivalent to :meth:`propagate_expectation`.

    Args:
        predictions (tuple of tensors): the predictions to propagate. Each tensor's
            shape must be ``E x B x Od``, where ``E``, ``B``, and ``Od`` represent the
            number of models, batch size, and output dimension, respectively.
        propagation_method (str): the propagation method to use.
        propagation_indices (tensor, optional): the model indices to choose
            (will use the same for all predictions).
            Only needed if ``propagation == "fixed_model"``.

    Returns:
        (tuple of tensors): the propagated predictions.
    """
    if propagation_method == "random_model":
        return propagate_random_model(predictions)
    if propagation_method == "fixed_model":
        return propagate_fixed_model(predictions, propagation_indices)
    if propagation_method == "expectation":
        return propagate_expectation(predictions)
    raise ValueError(f"Invalid propagation method {propagation_method}.")


def rfftfreq(samples: int, device: torch.device) -> torch.Tensor:
    if version.parse(torch.__version__) >= version.parse("1.8.0"):
        return torch.fft.rfftfreq(samples, device=device)
    freqs = np.fft.rfftfreq(samples)  # type: ignore
    return torch.from_numpy(freqs).to(device)


# ------------------------------------------------------------------------ #
# Colored noise generator for iCEM
# ------------------------------------------------------------------------ #
# Generate colored noise (Gaussian distributed noise with a power law spectrum)
# Adapted from colorednoise package, credit: https://github.com/felixpatzelt/colorednoise
def powerlaw_psd_gaussian(
    exponent: float,
    size: Union[int, Iterable[int]],
    device: torch.device,
    fmin: float = 0,
):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in: Timmer, J. and Koenig, M.:On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Args:
        exponent (float): the power-spectrum of the generated noise is proportional to
            S(f) = (1 / f)**exponent.
        size (int or iterable): the output shape and the desired power spectrum is in the last
            coordinate.
        device (torch.device): device where computations will be performed.
        fmin (float): low-frequency cutoff. Default: 0 corresponds to original paper.

    Returns
        (torch.Tensor): The samples.
    """

    # Make sure size is a list so we can iterate it and assign to it.
    if isinstance(size, int):
        size = [size]
    else:
        size = list(size)

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we assume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples, device=device)

    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1.0 / samples)  # Low frequency cutoff
    ix = torch.sum(s_scale < fmin)  # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-exponent / 2.0)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].detach().clone()
    w[-1] *= (1 + (samples % 2)) / 2.0  # correct f = +-0.5
    sigma = 2 * torch.sqrt(torch.sum(w**2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(None,) * dims_to_add + (Ellipsis,)]

    # Generate scaled random power + phase
    m = torch.distributions.Normal(loc=0.0, scale=s_scale.flatten())
    sr = m.sample(tuple(size[:-1]))
    si = m.sample(tuple(size[:-1]))

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si[..., -1] = 0

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0

    # Combine power + corrected phase to Fourier components
    s = sr + 1j * si

    # Transform to real time series & scale to unit variance
    y = torch.fft.irfft(s, n=samples, axis=-1) / sigma

    return y


# ------------------------------------------------------------------------ #
# Pixel manipulation
# ------------------------------------------------------------------------ #
def quantize_obs(
    obs: np.ndarray,
    bit_depth: int,
    original_bit_depth: int = 8,
    add_noise: bool = False,
):
    """Quantizes an array of pixel observations to the desired bit depth.

    Args:
        obs (np.ndarray): the array to quantize.
        bit_depth (int): the desired bit depth.
        original_bit_depth (int, optional): the original bit depth, defaults to 8.
        add_noise (bool, optional): if ``True``, uniform noise in the range
            (0, 2 ** (8 - bit_depth)) will be added. Defaults to ``False``.`

    Returns:
        (np.ndarray): the quantized version of the array.
    """
    ratio = 2 ** (original_bit_depth - bit_depth)
    quantized_obs = (obs // ratio) * ratio
    if add_noise:
        quantized_obs = quantized_obs.astype(np.double) + ratio * np.random.rand(
            *obs.shape
        )
    return quantized_obs
