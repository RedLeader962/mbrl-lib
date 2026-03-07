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

    def __init__(self, in_size: int, device: torch.device, dtype=torch.float32):
        super().__init__()
        self.register_buffer("mean", torch.zeros((1, in_size), dtype=dtype))
        self.register_buffer("std", torch.ones((1, in_size), dtype=dtype))
        # Minimum std floor chosen relative to each dtype's machine epsilon:
        #   float32  machine eps ≈ 1.19e-7  →  eps = 1e-5  (~84× machine eps)
        #   float64  machine eps ≈ 2.22e-16 →  eps = 1e-14 (~45× machine eps)
        # The buffer is stored in the normalizer's own dtype for consistency.
        _eps_value = 1e-14 if dtype == torch.double else 1e-5
        self.register_buffer("eps", torch.tensor(_eps_value, dtype=dtype))
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
        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
        return result.to(input_dtype)

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
        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
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
