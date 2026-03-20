# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import pickle
import tempfile
import warnings

import numpy as np
import pytest
import torch

import mbrl.models
import mbrl.util.math
import mbrl.util.normalization

_DEVICE = "cpu"


# ------------------------------------------------------------------ #
#  6.1 — Basic Functionality
# ------------------------------------------------------------------ #
class TestBasicFunctionality:
    def test_init_default(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(5, torch.device(_DEVICE))
        assert norm.mean.shape == (1, 5)
        assert norm.std.shape == (1, 5)
        assert torch.allclose(norm.mean, torch.zeros(1, 5))
        assert torch.allclose(norm.std, torch.ones(1, 5))
        assert norm.eps.item() == pytest.approx(1e-5)

    def test_init_double_precision(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(
            3, torch.device(_DEVICE), dtype=torch.double
        )
        assert norm.mean.dtype == torch.double
        assert norm.std.dtype == torch.double
        assert norm.eps.item() == pytest.approx(1e-14)
        assert norm.eps.dtype == torch.double

    def test_update_stats_basic(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        data = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] * 10, dtype=torch.float32
        )
        norm.update_stats(data)
        assert torch.allclose(norm.mean, data.mean(0, keepdim=True), atol=1e-5)
        assert torch.allclose(norm.std, data.std(0, keepdim=True), atol=1e-5)

    def test_normalize_basic(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        data = torch.randn(100, 2)
        norm.update_stats(data)
        normalized = norm.normalize(data)
        assert torch.allclose(normalized.mean(0), torch.zeros(2), atol=0.1)

    def test_denormalize_basic(self):
        """Round-trip: denormalize(normalize(x)) ≈ x."""
        norm = mbrl.util.normalization.ZScoreNormalizer(3, torch.device(_DEVICE))
        data = torch.randn(100, 3)
        norm.update_stats(data)
        reconstructed = norm.denormalize(norm.normalize(data))
        assert torch.allclose(reconstructed, data, atol=1e-5)

    def test_normalize_numpy_input(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        data_np = np.random.randn(50, 2).astype(np.float32)
        norm.update_stats(data_np)
        result = norm.normalize(data_np)
        assert isinstance(result, torch.Tensor)

    def test_normalize_float_input(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(1, torch.device(_DEVICE))
        result = norm.normalize(3.14)
        assert isinstance(result, torch.Tensor)


# ------------------------------------------------------------------ #
#  6.2 — Registered Buffer Integrity (Issue 3)
# ------------------------------------------------------------------ #
class TestBufferIntegrity:
    def test_buffers_in_state_dict(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(4, torch.device(_DEVICE))
        data = torch.randn(20, 4)
        norm.update_stats(data)
        sd = norm.state_dict()
        assert "mean" in sd
        assert "std" in sd
        assert "eps" in sd

    def test_buffers_survive_update_stats(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(3, torch.device(_DEVICE))
        data1 = torch.randn(20, 3)
        data2 = torch.randn(30, 3)
        norm.update_stats(data1)
        norm.update_stats(data2)
        sd = norm.state_dict()
        assert "mean" in sd
        assert "std" in sd
        assert torch.allclose(sd["mean"], data2.mean(0, keepdim=True), atol=1e-5)

    def test_state_dict_round_trip(self):
        norm1 = mbrl.util.normalization.ZScoreNormalizer(3, torch.device(_DEVICE))
        data = torch.randn(50, 3)
        norm1.update_stats(data)

        norm2 = mbrl.util.normalization.ZScoreNormalizer(3, torch.device(_DEVICE))
        norm2.load_state_dict(norm1.state_dict())
        assert torch.allclose(norm1.mean, norm2.mean)
        assert torch.allclose(norm1.std, norm2.std)
        assert torch.allclose(norm1.eps, norm2.eps)


# ------------------------------------------------------------------ #
#  6.3 — Save/Load (Issue 2)
# ------------------------------------------------------------------ #
class TestSaveLoad:
    def test_save_load_torch_format(self):
        norm1 = mbrl.util.normalization.ZScoreNormalizer(4, torch.device(_DEVICE))
        data = torch.randn(50, 4)
        norm1.update_stats(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            norm1.save(tmpdir)
            norm2 = mbrl.util.normalization.ZScoreNormalizer(4, torch.device(_DEVICE))
            norm2.load(tmpdir)

        assert torch.allclose(norm1.mean, norm2.mean)
        assert torch.allclose(norm1.std, norm2.std)
        assert torch.allclose(norm1.eps, norm2.eps)

    def test_save_creates_pt_file(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        with tempfile.TemporaryDirectory() as tmpdir:
            norm.save(tmpdir)
            assert (pathlib.Path(tmpdir) / "env_stats.pt").exists()
            assert not (pathlib.Path(tmpdir) / "env_stats.pickle").exists()

    def test_load_legacy_pickle_fallback(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        mean_np = np.array([[1.0, 2.0]], dtype=np.float32)
        std_np = np.array([[0.5, 1.5]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_path = pathlib.Path(tmpdir) / "env_stats.pickle"
            with open(pickle_path, "wb") as f:
                pickle.dump({"mean": mean_np, "std": std_np}, f)

            with pytest.warns(FutureWarning, match="legacy pickle format"):
                norm.load(tmpdir)

        assert torch.allclose(norm.mean, torch.tensor(mean_np))
        assert torch.allclose(norm.std, torch.tensor(std_np))

    def test_load_missing_file_raises(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                norm.load(tmpdir)


# ------------------------------------------------------------------ #
#  6.4 — Numerical Robustness
# ------------------------------------------------------------------ #
class TestNumericalRobustness:
    def test_update_stats_with_nan(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        data = torch.randn(20, 2)
        data[5, 0] = float("nan")
        with pytest.warns(RuntimeWarning, match="NaN or Inf"):
            norm.update_stats(data)
        assert torch.isfinite(norm.mean).all()
        assert torch.isfinite(norm.std).all()

    def test_update_stats_with_inf(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        data = torch.randn(20, 2)
        data[3, 1] = float("inf")
        with pytest.warns(RuntimeWarning, match="NaN or Inf"):
            norm.update_stats(data)
        assert torch.isfinite(norm.mean).all()
        assert torch.isfinite(norm.std).all()

    def test_update_stats_constant_dimension(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        data = torch.zeros(20, 2)
        data[:, 1] = torch.randn(20)
        norm.update_stats(data)
        # Constant column std should be clamped to eps, not zero
        assert norm.std[0, 0].item() >= norm.eps.item()

    def test_update_stats_single_sample(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(3, torch.device(_DEVICE))
        data = torch.tensor([[1.0, 2.0, 3.0]])
        with pytest.warns(RuntimeWarning, match="only 1 samples"):
            norm.update_stats(data)
        assert torch.allclose(norm.std, torch.ones(1, 3))

    def test_update_stats_small_sample_warning(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        data = torch.randn(5, 2)
        with pytest.warns(RuntimeWarning, match="only 5 samples"):
            norm.update_stats(data)

    def test_normalize_degenerate_stats(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        norm.mean.fill_(float("inf"))
        with pytest.warns(RuntimeWarning, match="non-finite values"):
            result = norm.normalize(torch.ones(1, 2))
        assert torch.isfinite(result).all()
        assert torch.allclose(result, torch.zeros(1, 2))

    def test_normalize_large_dynamic_range(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        data = torch.zeros(100, 2)
        data[:, 0] = torch.randn(100) * 1e-6
        data[:, 1] = torch.randn(100) * 1e6
        norm.update_stats(data)
        normalized = norm.normalize(data)
        assert torch.isfinite(normalized).all()

    def test_robotic_data_scenario(self):
        """Simulate realistic robotic data with different scales and NaN spikes."""
        norm = mbrl.util.normalization.ZScoreNormalizer(4, torch.device(_DEVICE))
        n = 200
        data = torch.zeros(n, 4)
        data[:, 0] = torch.randn(n) * 0.01  # joint position (small)
        data[:, 1] = torch.randn(n) * 10.0  # angular velocity (large)
        data[:, 2] = torch.randn(n) * 100.0  # torque (very large)
        data[:, 3] = torch.randn(n) * 0.001  # IMU bias (tiny)
        # Inject NaN spikes
        data[50, 0] = float("nan")
        data[100, 2] = float("inf")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            norm.update_stats(data)

        assert torch.isfinite(norm.mean).all()
        assert torch.isfinite(norm.std).all()

        # Clean data for round-trip
        clean_data = data.clone()
        clean_data[50, 0] = 0.0
        clean_data[100, 2] = 0.0
        normalized = norm.normalize(clean_data)
        assert torch.isfinite(normalized).all()
        reconstructed = norm.denormalize(normalized)
        assert torch.allclose(reconstructed, clean_data, atol=1e-4)


# ------------------------------------------------------------------ #
#  6.5 — Device and Dtype Handling
# ------------------------------------------------------------------ #
class TestDeviceDtype:
    def test_to_tensor_numpy(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(3, torch.device(_DEVICE))
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = norm._to_tensor(arr)
        assert isinstance(t, torch.Tensor)
        assert t.device.type == _DEVICE

    def test_to_tensor_float(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(1, torch.device(_DEVICE))
        t = norm._to_tensor(3.14)
        assert isinstance(t, torch.Tensor)

    def test_device_transfer(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        data = torch.randn(20, 2)
        norm.update_stats(data)
        norm = norm.to("cpu")
        assert norm.mean.device.type == "cpu"
        assert norm.std.device.type == "cpu"
        assert norm.eps.device.type == "cpu"

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_mps_float64_downcast(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device("mps"))
        data = torch.randn(20, 2, dtype=torch.float64)
        t = norm._to_tensor(data)
        assert t.dtype == torch.float32


# ------------------------------------------------------------------ #
#  6.5b — Input Dtype Preservation
# ------------------------------------------------------------------ #
class TestDtypePreservation:
    """Verify that normalize/denormalize return tensors in the caller's dtype."""

    def test_normalize_float32_input_float32_normalizer(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE), dtype=torch.float32)
        data = torch.randn(50, 2, dtype=torch.float32)
        norm.update_stats(data)
        result = norm.normalize(data)
        assert result.dtype == torch.float32

    def test_normalize_float64_input_float64_normalizer(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE), dtype=torch.float64)
        data = torch.randn(50, 2, dtype=torch.float64)
        norm.update_stats(data)
        result = norm.normalize(data)
        assert result.dtype == torch.float64

    def test_normalize_float32_input_float64_normalizer_preserves_float32(self):
        """float32 input to a double-precision normalizer must return float32."""
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE), dtype=torch.float64)
        data_f64 = torch.randn(50, 2, dtype=torch.float64)
        norm.update_stats(data_f64)
        query = torch.randn(10, 2, dtype=torch.float32)
        result = norm.normalize(query)
        assert result.dtype == torch.float32

    def test_normalize_float64_input_float32_normalizer_preserves_float64(self):
        """float64 input to a single-precision normalizer must return float64."""
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE), dtype=torch.float32)
        data = torch.randn(50, 2, dtype=torch.float32)
        norm.update_stats(data)
        query = torch.randn(10, 2, dtype=torch.float64)
        result = norm.normalize(query)
        assert result.dtype == torch.float64

    def test_denormalize_float32_input_float64_normalizer_preserves_float32(self):
        """float32 input to a double-precision denormalizer must return float32."""
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE), dtype=torch.float64)
        data_f64 = torch.randn(50, 2, dtype=torch.float64)
        norm.update_stats(data_f64)
        query = torch.randn(10, 2, dtype=torch.float32)
        result = norm.denormalize(query)
        assert result.dtype == torch.float32

    def test_denormalize_float64_input_float32_normalizer_preserves_float64(self):
        """float64 input to a single-precision denormalizer must return float64."""
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE), dtype=torch.float32)
        data = torch.randn(50, 2, dtype=torch.float32)
        norm.update_stats(data)
        query = torch.randn(10, 2, dtype=torch.float64)
        result = norm.denormalize(query)
        assert result.dtype == torch.float64

    def test_round_trip_cross_dtype_precision(self):
        """Round-trip through double normalizer with float32 input stays accurate."""
        norm = mbrl.util.normalization.ZScoreNormalizer(3, torch.device(_DEVICE), dtype=torch.float64)
        data_f64 = torch.randn(100, 3, dtype=torch.float64)
        norm.update_stats(data_f64)
        query = torch.randn(20, 3, dtype=torch.float32)
        reconstructed = norm.denormalize(norm.normalize(query))
        assert reconstructed.dtype == torch.float32
        assert torch.allclose(reconstructed, query, atol=1e-5)

    def test_normalize_uses_high_precision_arithmetic(self):
        """Even when output is float32, internal arithmetic should use float64
        when the normalizer stores double stats — reducing catastrophic cancellation."""
        norm = mbrl.util.normalization.ZScoreNormalizer(1, torch.device(_DEVICE), dtype=torch.float64)
        # A large mean makes (val - mean) prone to cancellation in float32
        norm.mean.fill_(1e7)
        norm.std.fill_(1.0)
        val = torch.tensor([[1e7 + 1.0]], dtype=torch.float32)
        result = norm.normalize(val)
        # If arithmetic were float32, the result would be 0.0 due to cancellation
        assert result.item() == pytest.approx(1.0, abs=1e-4)


# ------------------------------------------------------------------ #
#  6.6 — Integration with OneDTransitionRewardModel
# ------------------------------------------------------------------ #
class TestIntegration:
    def test_one_dim_tr_model_save_load_normalizer(self):
        from mbrl.models.gaussian_mlp import GaussianMLP

        model = GaussianMLP(
            in_size=4,
            out_size=3,
            device=_DEVICE,
            num_layers=2,
            hid_size=32,
            ensemble_size=1,
        )
        wrapper = mbrl.models.OneDTransitionRewardModel(
            model, target_is_delta=True, normalize=True,
            normalizer_type="standard",
        )
        assert wrapper.input_normalizer is not None

        # Simulate updating normalizer
        data = torch.randn(50, 4)
        wrapper.input_normalizer.update_stats(data)
        original_mean = wrapper.input_normalizer.mean.clone()
        original_std = wrapper.input_normalizer.std.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper.save(tmpdir)

            # Create a new model and load
            model2 = GaussianMLP(
                in_size=4,
                out_size=3,
                device=_DEVICE,
                num_layers=2,
                hid_size=32,
                ensemble_size=1,
            )
            wrapper2 = mbrl.models.OneDTransitionRewardModel(
                model2, target_is_delta=True, normalize=True,
                normalizer_type="standard",
            )
            wrapper2.load(tmpdir)

        assert torch.allclose(wrapper2.input_normalizer.mean, original_mean)
        assert torch.allclose(wrapper2.input_normalizer.std, original_std)


# ------------------------------------------------------------------ #
#  6.7 — Eps dtype and value consistency
# ------------------------------------------------------------------ #
class TestEpsDtypeConsistency:
    def test_eps_dtype_matches_normalizer_float32(self):
        """eps buffer dtype must match the normalizer's dtype."""
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE), dtype=torch.float32)
        assert norm.eps.dtype == torch.float32

    def test_eps_dtype_matches_normalizer_float64(self):
        """eps buffer dtype must match the normalizer's dtype."""
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE), dtype=torch.float64)
        assert norm.eps.dtype == torch.float64

    def test_eps_value_float32(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(1, torch.device(_DEVICE), dtype=torch.float32)
        assert norm.eps.item() == pytest.approx(1e-5)

    def test_eps_value_float64(self):
        norm = mbrl.util.normalization.ZScoreNormalizer(1, torch.device(_DEVICE), dtype=torch.float64)
        assert norm.eps.item() == pytest.approx(1e-14)

    def test_eps_clamps_std_float32(self):
        """For float32 normalizer, std must never fall below eps (~1e-5)."""
        norm = mbrl.util.normalization.ZScoreNormalizer(1, torch.device(_DEVICE), dtype=torch.float32)
        # Near-constant data: true std ≈ 1e-7, well below eps=1e-5
        data = torch.ones(100, 1, dtype=torch.float32) + torch.randn(100, 1) * 1e-7
        norm.update_stats(data)
        # Use approx to account for float32 representation of 1e-5
        assert norm.std.item() == pytest.approx(1e-5, rel=1e-6)

    def test_eps_clamps_std_float64(self):
        """For float64 normalizer, std must never fall below 1e-14."""
        norm = mbrl.util.normalization.ZScoreNormalizer(1, torch.device(_DEVICE), dtype=torch.float64)
        # Near-constant data: true std ≈ 1e-15, below eps=1e-14
        data = torch.ones(100, 1, dtype=torch.float64) + torch.randn(100, 1).double() * 1e-15
        norm.update_stats(data)
        assert norm.std.item() >= 1e-14

    def test_float64_eps_preserves_fine_variance(self):
        """float64 normalizer with eps=1e-14 must NOT clamp variance at 1e-10 scale."""
        norm = mbrl.util.normalization.ZScoreNormalizer(1, torch.device(_DEVICE), dtype=torch.float64)
        data = torch.ones(100, 1, dtype=torch.float64) + torch.randn(100, 1).double() * 1e-10
        true_std = data.std(0).item()
        norm.update_stats(data)
        # With the old eps=1e-12 this would pass too, but with eps=1e-14 the
        # normalizer can preserve even finer variance distinctions.
        assert norm.std.item() == pytest.approx(true_std, rel=1e-6)


# ------------------------------------------------------------------ #
#  6.8 — Non-finite Warn-and-Clamp Behavior
# ------------------------------------------------------------------ #
class TestNonFiniteWarnAndClamp:
    """Verify that normalize/denormalize warn and clamp (not zero) on non-finite values."""

    def test_normalize_warns_on_non_finite(self):
        """RuntimeWarning must be raised when normalize produces non-finite values."""
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        norm.mean.fill_(float("inf"))
        with pytest.warns(RuntimeWarning, match="non-finite values"):
            norm.normalize(torch.ones(1, 2))

    def test_denormalize_warns_on_non_finite(self):
        """RuntimeWarning must be raised when denormalize produces non-finite values."""
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        norm.std.fill_(float("inf"))
        with pytest.warns(RuntimeWarning, match="non-finite values"):
            norm.denormalize(torch.tensor([[1e30, 1e30]]))

    def test_normalize_clamps_to_finite_range(self):
        """Non-finite values should be clamped to [min, max] of the finite values."""
        norm = mbrl.util.normalization.ZScoreNormalizer(3, torch.device(_DEVICE))
        data = torch.randn(50, 3)
        norm.update_stats(data)
        # Force one std entry to zero to trigger inf in division
        norm.std[0, 0] = 0.0
        with pytest.warns(RuntimeWarning, match="non-finite values"):
            result = norm.normalize(torch.ones(1, 3))
        assert torch.isfinite(result).all()
        # The non-finite dimension should be clamped to the range of
        # the finite dimensions, not zeroed
        finite_vals = result[torch.isfinite(result)]
        assert finite_vals.numel() == result.numel()

    def test_normalize_all_non_finite_fallback_to_zeros(self):
        """When all values are non-finite, fallback must produce zeros."""
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        norm.mean.fill_(float("inf"))
        norm.std.fill_(1.0)
        with pytest.warns(RuntimeWarning, match="non-finite values"):
            result = norm.normalize(torch.ones(1, 2))
        assert torch.allclose(result, torch.zeros(1, 2))

    def test_denormalize_all_non_finite_fallback_to_zeros(self):
        """When all denormalized values are non-finite, fallback must produce zeros."""
        norm = mbrl.util.normalization.ZScoreNormalizer(2, torch.device(_DEVICE))
        norm.mean.fill_(float("inf"))
        norm.std.fill_(float("inf"))
        with pytest.warns(RuntimeWarning, match="non-finite values"):
            result = norm.denormalize(torch.ones(1, 2))
        assert torch.allclose(result, torch.zeros(1, 2))

    def test_no_warning_on_finite_values(self):
        """No warning should be raised when all values are finite."""
        norm = mbrl.util.normalization.ZScoreNormalizer(3, torch.device(_DEVICE))
        data = torch.randn(50, 3)
        norm.update_stats(data)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # This should not raise any RuntimeWarning about non-finite values
            norm.normalize(data)
            norm.denormalize(data)


# ------------------------------------------------------------------ #
#  6.9 — clip_range clamping of z-scored output
# ------------------------------------------------------------------ #
class TestClipRange:
    """Verify clip_range clamps z-scored output inside normalize().

    Motivation: heavy-tailed features (e.g. NeuroBem angular-velocity)
    produce outliers reaching ±10–15 σ after z-scoring. Without clamping
    these cause large gradient magnitudes / numerical instability even with
    non-saturating activations (GELU, LeakyReLU). clip_range=5.0 is the
    recommended default for NeuroBem.
    """

    def _make_fitted_normalizer(
        self, in_size: int, clip_range: float = None
    ) -> mbrl.util.normalization.ZScoreNormalizer:
        """Return a normalizer fitted to N(0,1) data with given clip_range."""
        norm = mbrl.util.normalization.ZScoreNormalizer(
            in_size, torch.device(_DEVICE), clip_range=clip_range
        )
        data = torch.randn(200, in_size)
        norm.update_stats(data)
        return norm

    def test_clip_range_none_by_default(self):
        """clip_range must default to None (no clamping)."""
        norm = mbrl.util.normalization.ZScoreNormalizer(3, torch.device(_DEVICE))
        assert norm.clip_range is None

    def test_clip_range_stored_on_init(self):
        """clip_range passed at construction must be stored on the instance."""
        norm = mbrl.util.normalization.ZScoreNormalizer(3, torch.device(_DEVICE), clip_range=5.0)
        assert norm.clip_range == 5.0

    def test_clip_range_none_does_not_clamp(self):
        """When clip_range is None extreme outliers must pass through unchanged."""
        norm = self._make_fitted_normalizer(3, clip_range=None)
        # Construct inputs that normalize to ~±15 σ
        extreme = norm.mean + 15.0 * norm.std
        result = norm.normalize(extreme)
        assert result.abs().max().item() == pytest.approx(15.0, rel=0.05), (
            "Without clip_range, extreme normalized values must not be clamped"
        )

    def test_clip_range_clamps_extreme_values(self):
        """Normalized values exceeding clip_range must be clamped to ±clip_range."""
        clip = 5.0
        norm = self._make_fitted_normalizer(3, clip_range=clip)
        # Construct inputs that would normalize to ±15 σ
        extreme_pos = norm.mean + 15.0 * norm.std
        extreme_neg = norm.mean - 15.0 * norm.std
        result_pos = norm.normalize(extreme_pos)
        result_neg = norm.normalize(extreme_neg)
        assert result_pos.max().item() == pytest.approx(clip, rel=1e-5), (
            f"Positive outlier should be clamped to +{clip}"
        )
        assert result_neg.min().item() == pytest.approx(-clip, rel=1e-5), (
            f"Negative outlier should be clamped to -{clip}"
        )

    def test_clip_range_preserves_in_distribution_values(self):
        """Values within [-clip_range, clip_range] must not be affected by clamping."""
        clip = 5.0
        norm = self._make_fitted_normalizer(3, clip_range=clip)
        # Inputs normalizing to ±2 σ (well inside clip window)
        in_dist_pos = norm.mean + 2.0 * norm.std
        in_dist_neg = norm.mean - 2.0 * norm.std
        result_pos = norm.normalize(in_dist_pos)
        result_neg = norm.normalize(in_dist_neg)
        assert result_pos.max().item() == pytest.approx(2.0, rel=0.05), (
            "In-distribution positive value must not be clipped"
        )
        assert result_neg.min().item() == pytest.approx(-2.0, rel=0.05), (
            "In-distribution negative value must not be clipped"
        )

    def test_clip_range_dtype_preserved_after_clamping(self):
        """clip_range clamping must preserve the input tensor dtype."""
        for dtype in (torch.float32, torch.float64):
            norm = mbrl.util.normalization.ZScoreNormalizer(
                2, torch.device(_DEVICE), dtype=dtype, clip_range=3.0
            )
            data = torch.randn(100, 2, dtype=dtype)
            norm.update_stats(data)
            extreme = norm.mean + 10.0 * norm.std
            result = norm.normalize(extreme.to(dtype))
            assert result.dtype == dtype, (
                f"dtype must be preserved after clamping (expected {dtype}, got {result.dtype})"
            )

    @pytest.mark.parametrize("clip", [1.0, 3.0, 5.0, 10.0])
    def test_clip_range_boundary_parametrized(self, clip: float):
        """Normalized values must be bounded by ±clip for any clip value."""
        norm = self._make_fitted_normalizer(4, clip_range=clip)
        # Use a batch with both extreme positive and negative outliers
        extreme = torch.cat(
            [norm.mean + 20.0 * norm.std, norm.mean - 20.0 * norm.std], dim=0
        )
        result = norm.normalize(extreme)
        assert result.max().item() <= clip + 1e-5, (
            f"All normalized values must be ≤ {clip}"
        )
        assert result.min().item() >= -clip - 1e-5, (
            f"All normalized values must be ≥ -{clip}"
        )
