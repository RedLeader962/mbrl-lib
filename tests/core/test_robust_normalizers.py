# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for WinsorizedNormalizer, QuantileNormalizer, and create_normalizer factory."""
import pathlib
import tempfile
import warnings

import numpy as np
import pytest
import torch

import mbrl.models
import mbrl.types
import mbrl.util.math
import mbrl.util.normalization

_DEVICE = "cpu"


# ------------------------------------------------------------------ #
#  WinsorizedNormalizer
# ------------------------------------------------------------------ #
class TestWinsorizedBasic:
    def test_init_default(self):
        norm = mbrl.util.normalization.WinsorizedNormalizer(5, torch.device(_DEVICE))
        assert norm.winsorized_mean.shape == (1, 5)
        assert norm.winsorized_std.shape == (1, 5)
        assert torch.allclose(norm.winsorized_mean, torch.zeros(1, 5))
        assert torch.allclose(norm.winsorized_std, torch.ones(1, 5))
        assert norm.winsor_percentile == 0.05
        assert norm.soft_clip_iqr_mult == 3.0

    def test_init_double_precision(self):
        norm = mbrl.util.normalization.WinsorizedNormalizer(
            3, torch.device(_DEVICE), dtype=torch.double
        )
        assert norm.winsorized_mean.dtype == torch.double
        assert norm.winsorized_std.dtype == torch.double
        assert norm.eps.item() == pytest.approx(1e-14)

    def test_mean_std_aliases(self):
        norm = mbrl.util.normalization.WinsorizedNormalizer(3, torch.device(_DEVICE))
        assert norm.mean is norm.winsorized_mean
        assert norm.std is norm.winsorized_std

    def test_update_stats_basic(self):
        norm = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE))
        data = torch.randn(200, 2)
        norm.update_stats(data)
        assert torch.isfinite(norm.winsorized_mean).all()
        assert torch.isfinite(norm.winsorized_std).all()
        assert (norm.winsorized_std > 0).all()

    def test_normalize_basic(self):
        norm = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE))
        data = torch.randn(200, 2)
        norm.update_stats(data)
        normalized = norm.normalize(data)
        assert torch.isfinite(normalized).all()
        # Bulk of data should be near zero mean
        assert normalized.mean(0).abs().max().item() < 0.5


class TestWinsorizedRoundTrip:
    def test_round_trip_in_distribution(self):
        """denormalize(normalize(x)) ≈ x for values within the soft-clip linear region."""
        norm = mbrl.util.normalization.WinsorizedNormalizer(3, torch.device(_DEVICE))
        data = torch.randn(500, 3)
        norm.update_stats(data)
        # Use values within ~2 sigma (well within clip threshold)
        query = data[:50]
        reconstructed = norm.denormalize(norm.normalize(query))
        assert torch.allclose(reconstructed, query, atol=1e-5)

    def test_round_trip_with_outliers(self):
        """Round-trip for extreme values (in the soft-clip region)."""
        norm = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE))
        data = torch.randn(500, 2)
        norm.update_stats(data)
        # Values at ~5 sigma — may be in soft-clip region
        extreme = norm.winsorized_mean + 5.0 * norm.winsorized_std
        reconstructed = norm.denormalize(norm.normalize(extreme))
        assert torch.allclose(reconstructed, extreme, atol=1e-4)


class TestWinsorizedRobustness:
    def test_outlier_robustness(self):
        """Statistics should be stable when outliers are injected."""
        norm_clean = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE))
        norm_dirty = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE))

        data = torch.randn(500, 2)
        norm_clean.update_stats(data)

        dirty = data.clone()
        dirty[0, 0] = 1e6  # extreme outlier
        dirty[1, 1] = -1e6
        norm_dirty.update_stats(dirty)

        # Winsorized statistics should be close despite outliers
        assert torch.allclose(
            norm_clean.winsorized_mean, norm_dirty.winsorized_mean, atol=0.5
        )
        assert torch.allclose(
            norm_clean.winsorized_std, norm_dirty.winsorized_std, rtol=0.5
        )

    def test_nan_handling(self):
        norm = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE))
        data = torch.randn(50, 2)
        data[5, 0] = float("nan")
        with pytest.warns(RuntimeWarning, match="NaN or Inf"):
            norm.update_stats(data)
        assert torch.isfinite(norm.winsorized_mean).all()
        assert torch.isfinite(norm.winsorized_std).all()

    def test_constant_feature(self):
        norm = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE))
        data = torch.zeros(100, 2)
        data[:, 1] = torch.randn(100)
        norm.update_stats(data)
        assert norm.winsorized_std[0, 0].item() >= norm.eps.item()

    def test_small_sample_warning(self):
        norm = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE))
        data = torch.randn(5, 2)
        with pytest.warns(RuntimeWarning, match="only 5 samples"):
            norm.update_stats(data)

    def test_per_feature_independence(self):
        """Modifying one feature's data should not affect another feature's statistics."""
        norm1 = mbrl.util.normalization.WinsorizedNormalizer(3, torch.device(_DEVICE))
        norm2 = mbrl.util.normalization.WinsorizedNormalizer(3, torch.device(_DEVICE))
        data = torch.randn(200, 3)
        norm1.update_stats(data)
        data2 = data.clone()
        data2[:, 2] = torch.randn(200) * 100  # change only feature 2
        norm2.update_stats(data2)
        # Features 0 and 1 should have identical statistics
        assert torch.allclose(norm1.winsorized_mean[0, :2], norm2.winsorized_mean[0, :2], atol=1e-5)
        assert torch.allclose(norm1.winsorized_std[0, :2], norm2.winsorized_std[0, :2], atol=1e-5)


class TestWinsorizedSoftClip:
    def test_soft_clip_bounded(self):
        """Output should be bounded to ±(c_i + 1)."""
        norm = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE))
        data = torch.randn(500, 2)
        norm.update_stats(data)
        extreme = norm.winsorized_mean + 20.0 * norm.winsorized_std
        result = norm.normalize(extreme)
        max_bound = norm.clip_threshold + 1.0
        assert (result.abs() <= max_bound + 1e-5).all()

    def test_soft_clip_monotonic(self):
        """The soft-clip function must be monotonic."""
        threshold = torch.tensor([[3.0, 4.0]])
        z = torch.linspace(-10, 10, 100).unsqueeze(1).expand(-1, 2)
        clipped = mbrl.util.normalization.WinsorizedNormalizer._soft_clip(z, threshold)
        # Check that differences are all non-negative (monotonically increasing)
        diffs = clipped[1:] - clipped[:-1]
        assert (diffs >= -1e-6).all()

    def test_soft_clip_inverse_exact(self):
        """_soft_clip_inverse should exactly invert _soft_clip for values in valid range."""
        threshold = torch.tensor([[3.0]])
        z = torch.linspace(-5, 5, 200).unsqueeze(1)
        clipped = mbrl.util.normalization.WinsorizedNormalizer._soft_clip(z, threshold)
        recovered = mbrl.util.normalization.WinsorizedNormalizer._soft_clip_inverse(clipped, threshold)
        assert torch.allclose(recovered, z, atol=1e-5)


class TestWinsorizedDtype:
    def test_normalize_preserves_float32(self):
        norm = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE), dtype=torch.float32)
        data = torch.randn(100, 2)
        norm.update_stats(data)
        result = norm.normalize(data[:10])
        assert result.dtype == torch.float32

    def test_normalize_preserves_float64(self):
        norm = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE), dtype=torch.float64)
        data = torch.randn(100, 2, dtype=torch.float64)
        norm.update_stats(data)
        result = norm.normalize(data[:10])
        assert result.dtype == torch.float64

    def test_cross_dtype_float32_input_float64_normalizer(self):
        norm = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE), dtype=torch.float64)
        data = torch.randn(100, 2, dtype=torch.float64)
        norm.update_stats(data)
        query = torch.randn(10, 2, dtype=torch.float32)
        result = norm.normalize(query)
        assert result.dtype == torch.float32

    def test_denormalize_preserves_dtype(self):
        norm = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE), dtype=torch.float64)
        data = torch.randn(100, 2, dtype=torch.float64)
        norm.update_stats(data)
        query = torch.randn(10, 2, dtype=torch.float32)
        result = norm.denormalize(query)
        assert result.dtype == torch.float32


class TestWinsorizedSaveLoad:
    def test_save_load_round_trip(self):
        norm1 = mbrl.util.normalization.WinsorizedNormalizer(4, torch.device(_DEVICE))
        data = torch.randn(200, 4)
        norm1.update_stats(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            norm1.save(tmpdir)
            norm2 = mbrl.util.normalization.WinsorizedNormalizer(4, torch.device(_DEVICE))
            norm2.load(tmpdir)

        assert torch.allclose(norm1.winsorized_mean, norm2.winsorized_mean)
        assert torch.allclose(norm1.winsorized_std, norm2.winsorized_std)
        assert torch.allclose(norm1.clip_threshold, norm2.clip_threshold)
        assert norm1.winsor_percentile == norm2.winsor_percentile
        assert norm1.soft_clip_iqr_mult == norm2.soft_clip_iqr_mult

    def test_load_missing_raises(self):
        norm = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE))
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                norm.load(tmpdir)

    def test_state_dict_round_trip(self):
        norm1 = mbrl.util.normalization.WinsorizedNormalizer(3, torch.device(_DEVICE))
        data = torch.randn(100, 3)
        norm1.update_stats(data)
        norm2 = mbrl.util.normalization.WinsorizedNormalizer(3, torch.device(_DEVICE))
        norm2.load_state_dict(norm1.state_dict())
        assert torch.allclose(norm1.winsorized_mean, norm2.winsorized_mean)
        assert torch.allclose(norm1.winsorized_std, norm2.winsorized_std)


class TestWinsorizedGradient:
    def test_gradient_flow(self):
        """Verify autograd propagates through normalize."""
        norm = mbrl.util.normalization.WinsorizedNormalizer(2, torch.device(_DEVICE))
        data = torch.randn(200, 2)
        norm.update_stats(data)
        x = torch.randn(5, 2, requires_grad=True)
        y = norm.normalize(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ------------------------------------------------------------------ #
#  QuantileNormalizer
# ------------------------------------------------------------------ #
class TestQuantileBasic:
    def test_init_default(self):
        norm = mbrl.util.normalization.QuantileNormalizer(5, torch.device(_DEVICE))
        assert norm.quantile_boundaries.shape == (1001, 5)
        assert norm.target_quantiles.shape == (1001,)
        assert norm.n_bins == 1000
        assert norm.tail_policy == "linear"

    def test_init_custom_bins(self):
        norm = mbrl.util.normalization.QuantileNormalizer(3, torch.device(_DEVICE), n_bins=100)
        assert norm.quantile_boundaries.shape == (101, 3)
        assert norm.n_bins == 100

    def test_target_quantiles_sorted(self):
        norm = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE))
        diffs = norm.target_quantiles[1:] - norm.target_quantiles[:-1]
        assert (diffs >= 0).all()

    def test_update_stats_basic(self):
        norm = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE), n_bins=100)
        data = torch.randn(500, 2)
        norm.update_stats(data)
        # Boundaries should be sorted per feature
        diffs = norm.quantile_boundaries[1:] - norm.quantile_boundaries[:-1]
        assert (diffs >= -1e-6).all()

    def test_normalize_basic(self):
        norm = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE), n_bins=100)
        data = torch.randn(500, 2)
        norm.update_stats(data)
        normalized = norm.normalize(data)
        assert torch.isfinite(normalized).all()
        # Output should be approximately standard normal
        assert normalized.mean(0).abs().max().item() < 0.3
        assert (normalized.std(0) - 1.0).abs().max().item() < 0.3


class TestQuantileRoundTrip:
    def test_round_trip_in_distribution(self):
        """denormalize(normalize(x)) ≈ x for in-distribution values."""
        norm = mbrl.util.normalization.QuantileNormalizer(3, torch.device(_DEVICE), n_bins=500)
        data = torch.randn(1000, 3)
        norm.update_stats(data)
        query = data[:50]
        reconstructed = norm.denormalize(norm.normalize(query))
        assert torch.allclose(reconstructed, query, atol=0.05)

    def test_round_trip_1d_input(self):
        """Round-trip should work for 1D input."""
        norm = mbrl.util.normalization.QuantileNormalizer(3, torch.device(_DEVICE), n_bins=200)
        data = torch.randn(500, 3)
        norm.update_stats(data)
        query = data[0]  # 1D
        reconstructed = norm.denormalize(norm.normalize(query))
        assert torch.allclose(reconstructed, query, atol=0.05)


class TestQuantileRobustness:
    def test_outlier_robustness(self):
        """Statistics should be stable despite extreme outliers."""
        norm_clean = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE), n_bins=100)
        norm_dirty = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE), n_bins=100)

        data = torch.randn(500, 2)
        norm_clean.update_stats(data)

        dirty = data.clone()
        dirty[0, 0] = 1e6
        dirty[1, 1] = -1e6
        norm_dirty.update_stats(dirty)

        # Mid-range in-distribution values should normalize similarly
        # (exclude tail values where a single outlier can shift the extreme quantile bins)
        query = data[10:20]
        result_clean = norm_clean.normalize(query)
        result_dirty = norm_dirty.normalize(query)
        # Compare only values that are within the central 95% of the clean normalizer output
        mask = result_clean.abs() < 2.0
        assert mask.sum() > 0, "Need at least some mid-range values"
        assert torch.allclose(result_clean[mask], result_dirty[mask], atol=0.1)

    def test_nan_handling(self):
        norm = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE), n_bins=50)
        data = torch.randn(50, 2)
        data[5, 0] = float("nan")
        with pytest.warns(RuntimeWarning, match="NaN or Inf"):
            norm.update_stats(data)

    def test_constant_feature(self):
        """Constant feature should not cause division by zero."""
        norm = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE), n_bins=50)
        data = torch.zeros(100, 2)
        data[:, 1] = torch.randn(100)
        norm.update_stats(data)
        result = norm.normalize(data[:10])
        assert torch.isfinite(result).all()

    def test_small_sample_warning(self):
        norm = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE), n_bins=10)
        data = torch.randn(5, 2)
        with pytest.warns(RuntimeWarning, match="only 5 samples"):
            norm.update_stats(data)

    def test_tail_extrapolation(self):
        """Values beyond the training range should be extrapolated."""
        norm = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE), n_bins=100)
        data = torch.randn(500, 2)
        norm.update_stats(data)
        # Values far outside the training range
        extreme_high = torch.full((1, 2), 10.0)
        extreme_low = torch.full((1, 2), -10.0)
        result_high = norm.normalize(extreme_high)
        result_low = norm.normalize(extreme_low)
        assert torch.isfinite(result_high).all()
        assert torch.isfinite(result_low).all()
        assert (result_high > 0).all()
        assert (result_low < 0).all()


class TestQuantileDtype:
    def test_normalize_preserves_float32(self):
        norm = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE), n_bins=50)
        data = torch.randn(100, 2)
        norm.update_stats(data)
        result = norm.normalize(data[:10])
        assert result.dtype == torch.float32

    def test_normalize_preserves_float64(self):
        norm = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE), dtype=torch.float64, n_bins=50)
        data = torch.randn(100, 2, dtype=torch.float64)
        norm.update_stats(data)
        result = norm.normalize(data[:10])
        assert result.dtype == torch.float64

    def test_cross_dtype(self):
        norm = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE), dtype=torch.float64, n_bins=50)
        data = torch.randn(100, 2, dtype=torch.float64)
        norm.update_stats(data)
        query = torch.randn(10, 2, dtype=torch.float32)
        result = norm.normalize(query)
        assert result.dtype == torch.float32


class TestQuantileSaveLoad:
    def test_save_load_round_trip(self):
        norm1 = mbrl.util.normalization.QuantileNormalizer(4, torch.device(_DEVICE), n_bins=50)
        data = torch.randn(200, 4)
        norm1.update_stats(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            norm1.save(tmpdir)
            norm2 = mbrl.util.normalization.QuantileNormalizer(4, torch.device(_DEVICE), n_bins=50)
            norm2.load(tmpdir)

        assert torch.allclose(norm1.quantile_boundaries, norm2.quantile_boundaries)
        assert torch.allclose(norm1.target_quantiles, norm2.target_quantiles)
        assert norm1.n_bins == norm2.n_bins

    def test_load_missing_raises(self):
        norm = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE), n_bins=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                norm.load(tmpdir)


class TestQuantileGradient:
    def test_gradient_flow(self):
        """Verify autograd propagates through normalize."""
        norm = mbrl.util.normalization.QuantileNormalizer(2, torch.device(_DEVICE), n_bins=50)
        data = torch.randn(200, 2)
        norm.update_stats(data)
        x = torch.randn(5, 2, requires_grad=True)
        y = norm.normalize(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ------------------------------------------------------------------ #
#  create_normalizer factory
# ------------------------------------------------------------------ #
class TestCreateNormalizer:
    def test_standard(self):
        norm = mbrl.util.normalization.create_normalizer("standard", 5, torch.device(_DEVICE))
        assert isinstance(norm, mbrl.util.normalization.ZScoreNormalizer)

    def test_winsorized(self):
        norm = mbrl.util.normalization.create_normalizer("winsorized", 5, torch.device(_DEVICE))
        assert isinstance(norm, mbrl.util.normalization.WinsorizedNormalizer)

    def test_quantile(self):
        norm = mbrl.util.normalization.create_normalizer("quantile", 5, torch.device(_DEVICE))
        assert isinstance(norm, mbrl.util.normalization.QuantileNormalizer)

    def test_quantile_with_kwargs(self):
        norm = mbrl.util.normalization.create_normalizer(
            "quantile", 5, torch.device(_DEVICE), n_bins=200, tail_policy="linear"
        )
        assert isinstance(norm, mbrl.util.normalization.QuantileNormalizer)
        assert norm.n_bins == 200

    def test_winsorized_with_kwargs(self):
        norm = mbrl.util.normalization.create_normalizer(
            "winsorized", 5, torch.device(_DEVICE),
            winsor_percentile=0.1, soft_clip_iqr_mult=2.0,
        )
        assert isinstance(norm, mbrl.util.normalization.WinsorizedNormalizer)
        assert norm.winsor_percentile == 0.1
        assert norm.soft_clip_iqr_mult == 2.0

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown normalizer_type"):
            mbrl.util.normalization.create_normalizer("unknown", 5, torch.device(_DEVICE))

    def test_all_types_share_interface(self):
        """All normalizer types should have update_stats, normalize, denormalize."""
        for ntype in ("standard", "winsorized", "quantile"):
            kwargs = {"n_bins": 50} if ntype == "quantile" else {}
            norm = mbrl.util.normalization.create_normalizer(
                ntype, 3, torch.device(_DEVICE), **kwargs
            )
            data = torch.randn(100, 3)
            norm.update_stats(data)
            result = norm.normalize(data[:10])
            assert torch.isfinite(result).all()
            reconstructed = norm.denormalize(result)
            assert torch.isfinite(reconstructed).all()

    def test_all_types_have_mean_std(self):
        """All normalizer types should expose mean and std attributes."""
        for ntype in ("standard", "winsorized", "quantile"):
            kwargs = {"n_bins": 50} if ntype == "quantile" else {}
            norm = mbrl.util.normalization.create_normalizer(
                ntype, 3, torch.device(_DEVICE), **kwargs
            )
            data = torch.randn(100, 3)
            norm.update_stats(data)
            assert hasattr(norm, "mean")
            assert hasattr(norm, "std")
            assert norm.mean.shape == (1, 3)
            assert norm.std.shape == (1, 3)


# ------------------------------------------------------------------ #
#  Robotic data scenario (shared across normalizer types)
# ------------------------------------------------------------------ #
class TestRoboticScenario:
    @pytest.mark.parametrize("ntype", ["standard", "winsorized", "quantile"])
    def test_heterogeneous_scales(self, ntype):
        """Simulate realistic robotic data with different scales per feature."""
        kwargs = {"n_bins": 100} if ntype == "quantile" else {}
        norm = mbrl.util.normalization.create_normalizer(
            ntype, 4, torch.device(_DEVICE), **kwargs
        )
        n = 500
        data = torch.zeros(n, 4)
        data[:, 0] = torch.randn(n) * 0.01  # joint position (small)
        data[:, 1] = torch.randn(n) * 10.0  # angular velocity (large)
        data[:, 2] = torch.randn(n) * 100.0  # torque (very large)
        data[:, 3] = torch.randn(n) * 0.001  # IMU bias (tiny)
        norm.update_stats(data)

        normalized = norm.normalize(data)
        assert torch.isfinite(normalized).all()
        # All features should be on comparable scale after normalization
        per_feat_std = normalized.std(0)
        assert per_feat_std.max() / per_feat_std.min() < 10.0


# ------------------------------------------------------------------ #
#  Composed-observation regression tests (multi-step model)
# ------------------------------------------------------------------ #
class _MockMultiStepModel:
    """Lightweight mock that exposes the attributes OneDTransitionRewardModel
    needs when wrapping a multi-step model (singlestep_obs_len, singlestep_act_len,
    history_len) without requiring the full MultiStepMLP stack."""

    def __init__(self, Do: int, Da: int, H: int):
        self.singlestep_obs_len = Do
        self.singlestep_act_len = Da
        self.history_len = H
        # model.in_size = Do*H + Da*H  (composed_obs + current_act)
        self.in_size = Do * H + Da * H
        self.device = torch.device(_DEVICE)


class _MockSingleStepModel:
    """Mock for a plain single-step model (no history_len attribute)."""

    def __init__(self, Do: int, Da: int):
        self.in_size = Do + Da
        self.device = torch.device(_DEVICE)


def _make_composed_obs(N, Do, Da, H, seed=42):
    """Build a composed observation batch: (N, Do*H + Da*(H-1)).
    Obs block has H timesteps, act block has H-1 timesteps."""
    rng = torch.Generator().manual_seed(seed)
    obs_block = torch.randn(N, Do * H, generator=rng)
    act_block = torch.randn(N, Da * (H - 1), generator=rng)
    return torch.cat([obs_block, act_block], dim=-1)


def _make_batch(N, Do, Da, H, seed=42):
    """Build a TransitionBatch with composed observations."""
    rng = torch.Generator().manual_seed(seed)
    obs = _make_composed_obs(N, Do, Da, H, seed=seed)
    act = torch.randn(N, Da, generator=rng)
    next_obs = _make_composed_obs(N, Do, Da, H, seed=seed + 1)
    rewards = torch.randn(N, generator=rng)
    terms = torch.zeros(N)
    truncs = torch.zeros(N)
    return mbrl.types.TransitionBatch(obs, act, next_obs, rewards, terms, truncs)


class TestComposedObsNormalization:
    """Regression tests: robust normalizers must correctly decompose composed
    observations (Do*H + Da*(H-1)) instead of passing them whole to a
    normalizer of size Do."""

    Do, Da, H = 3, 1, 13  # Lorenz-like: obs=3D, act=1D, history=13
    composed_dim = Do * H + Da * (H - 1)  # 39 + 12 = 51
    N = 100

    @pytest.mark.parametrize("ntype", ["winsorized", "quantile"])
    def test_normalize_composed_obs_no_crash(self, ntype):
        """_normalize_composed_obs must not crash on multi-step composed obs."""
        kwargs = {"n_bins": 50} if ntype == "quantile" else {}
        model = _MockMultiStepModel(self.Do, self.Da, self.H)
        one_d = mbrl.models.OneDTransitionRewardModel(
            model=model,
            normalize=True,
            normalizer_type=ntype,
            obs_dim=self.Do,
            act_dim=self.Da,
            target_is_delta=False,
            learned_rewards=False,
            normalizer_kwargs=kwargs,
        )
        # Feed training data to normalizer
        batch = _make_batch(self.N, self.Do, self.Da, self.H)
        one_d.update_normalizer(batch)

        # This used to crash with dimension mismatch
        composed_obs = batch.obs
        result = one_d._normalize_composed_obs(composed_obs)
        assert result.shape == composed_obs.shape
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("ntype", ["winsorized", "quantile"])
    def test_denormalize_composed_obs_round_trip(self, ntype):
        """denormalize(normalize(composed_obs)) ≈ composed_obs."""
        kwargs = {"n_bins": 50} if ntype == "quantile" else {}
        model = _MockMultiStepModel(self.Do, self.Da, self.H)
        one_d = mbrl.models.OneDTransitionRewardModel(
            model=model,
            normalize=True,
            normalizer_type=ntype,
            obs_dim=self.Do,
            act_dim=self.Da,
            target_is_delta=False,
            learned_rewards=False,
            normalizer_kwargs=kwargs,
        )
        batch = _make_batch(self.N, self.Do, self.Da, self.H)
        one_d.update_normalizer(batch)

        composed_obs = batch.obs
        normed = one_d._normalize_composed_obs(composed_obs)
        recovered = one_d._denormalize_composed_obs(normed)
        assert torch.allclose(recovered, composed_obs, atol=1e-4)

    @pytest.mark.parametrize("ntype", ["winsorized", "quantile"])
    def test_get_model_input_shape(self, ntype):
        """_get_model_input must produce (N, Do*H + Da*H) from composed obs + act."""
        kwargs = {"n_bins": 50} if ntype == "quantile" else {}
        model = _MockMultiStepModel(self.Do, self.Da, self.H)
        one_d = mbrl.models.OneDTransitionRewardModel(
            model=model,
            normalize=True,
            normalizer_type=ntype,
            obs_dim=self.Do,
            act_dim=self.Da,
            target_is_delta=False,
            learned_rewards=False,
            normalizer_kwargs=kwargs,
        )
        batch = _make_batch(self.N, self.Do, self.Da, self.H)
        one_d.update_normalizer(batch)

        model_in = one_d._get_model_input(batch.obs, batch.act)
        expected_dim = self.Do * self.H + self.Da * self.H  # composed + current act
        assert model_in.shape == (self.N, expected_dim)
        assert torch.isfinite(model_in).all()

    @pytest.mark.parametrize("ntype", ["winsorized", "quantile"])
    def test_process_batch_shapes(self, ntype):
        """_process_batch must return correct shapes for model_in and target."""
        kwargs = {"n_bins": 50} if ntype == "quantile" else {}
        model = _MockMultiStepModel(self.Do, self.Da, self.H)
        one_d = mbrl.models.OneDTransitionRewardModel(
            model=model,
            normalize=True,
            normalizer_type=ntype,
            obs_dim=self.Do,
            act_dim=self.Da,
            target_is_delta=False,
            learned_rewards=False,
            normalizer_kwargs=kwargs,
        )
        batch = _make_batch(self.N, self.Do, self.Da, self.H)
        one_d.update_normalizer(batch)

        model_in, target = one_d._process_batch(batch)
        assert model_in.shape == (self.N, self.Do * self.H + self.Da * self.H)
        assert target.shape == (self.N, self.composed_dim)
        assert torch.isfinite(model_in).all()
        assert torch.isfinite(target).all()

    @pytest.mark.parametrize("ntype", ["winsorized", "quantile"])
    def test_obs_block_uses_obs_normalizer_stats(self, ntype):
        """Obs block positions in composed obs must use obs_normalizer statistics,
        not act_normalizer statistics (and vice versa for act block)."""
        kwargs = {"n_bins": 50} if ntype == "quantile" else {}
        model = _MockMultiStepModel(self.Do, self.Da, self.H)
        one_d = mbrl.models.OneDTransitionRewardModel(
            model=model,
            normalize=True,
            normalizer_type=ntype,
            obs_dim=self.Do,
            act_dim=self.Da,
            target_is_delta=False,
            learned_rewards=False,
            normalizer_kwargs=kwargs,
        )
        # Create data with very different scales for obs vs act
        N = 500
        obs_data = torch.randn(N, self.Do) * 100.0  # large scale
        act_data = torch.randn(N, self.Da) * 0.01  # small scale
        # Build composed obs manually
        obs_block = obs_data.repeat(1, self.H)  # (N, Do*H)
        act_block = act_data.repeat(1, self.H - 1)  # (N, Da*(H-1))
        composed = torch.cat([obs_block, act_block], dim=-1)
        act_current = act_data  # (N, Da)

        batch = mbrl.types.TransitionBatch(
            composed, act_current, composed, torch.zeros(N), torch.zeros(N), torch.zeros(N)
        )
        one_d.update_normalizer(batch)

        normed = one_d._normalize_composed_obs(composed)
        # Obs block should be normalized with obs stats (large std)
        obs_normed = normed[..., : self.Do * self.H].reshape(-1, self.Do)
        # Act block should be normalized with act stats (small std)
        act_normed = normed[..., self.Do * self.H :].reshape(-1, self.Da)

        # Both blocks should have comparable scale after normalization
        assert obs_normed.std(0).max().item() < 5.0
        assert act_normed.std(0).max().item() < 5.0

    @pytest.mark.parametrize("ntype", ["winsorized", "quantile"])
    def test_update_normalizer_pools_act_from_composed_and_batch(self, ntype):
        """update_normalizer must pool actions from both composed obs (H-1 steps)
        and batch.act (1 step)."""
        kwargs = {"n_bins": 50} if ntype == "quantile" else {}
        model = _MockMultiStepModel(self.Do, self.Da, self.H)
        one_d = mbrl.models.OneDTransitionRewardModel(
            model=model,
            normalize=True,
            normalizer_type=ntype,
            obs_dim=self.Do,
            act_dim=self.Da,
            target_is_delta=False,
            learned_rewards=False,
            normalizer_kwargs=kwargs,
        )
        batch = _make_batch(self.N, self.Do, self.Da, self.H)
        one_d.update_normalizer(batch)

        # act_normalizer should have been updated (non-default stats)
        assert not torch.allclose(
            one_d.act_normalizer.mean, torch.zeros(1, self.Da), atol=1e-6
        ) or not torch.allclose(
            one_d.act_normalizer.std, torch.ones(1, self.Da), atol=1e-6
        )

    @pytest.mark.parametrize("ntype", ["winsorized", "quantile"])
    def test_single_step_fallback(self, ntype):
        """For single-step models, _normalize_composed_obs should just call
        obs_normalizer.normalize directly (no decomposition)."""
        kwargs = {"n_bins": 50} if ntype == "quantile" else {}
        model = _MockSingleStepModel(self.Do, self.Da)
        one_d = mbrl.models.OneDTransitionRewardModel(
            model=model,
            normalize=True,
            normalizer_type=ntype,
            obs_dim=self.Do,
            act_dim=self.Da,
            target_is_delta=False,
            learned_rewards=False,
            normalizer_kwargs=kwargs,
        )
        raw_obs = torch.randn(50, self.Do)
        raw_act = torch.randn(50, self.Da)
        batch = mbrl.types.TransitionBatch(
            raw_obs, raw_act, raw_obs, torch.zeros(50), torch.zeros(50), torch.zeros(50)
        )
        one_d.update_normalizer(batch)

        result = one_d._normalize_composed_obs(raw_obs)
        assert result.shape == raw_obs.shape
        assert torch.isfinite(result).all()

        # Round-trip
        recovered = one_d._denormalize_composed_obs(result)
        assert torch.allclose(recovered, raw_obs, atol=1e-4)

    @pytest.mark.parametrize("ntype", ["winsorized", "quantile"])
    def test_standard_normalizer_unaffected(self, ntype):
        """standard normalizer_type should not use _normalize_composed_obs at all."""
        model = _MockMultiStepModel(self.Do, self.Da, self.H)
        one_d = mbrl.models.OneDTransitionRewardModel(
            model=model,
            normalize=True,
            normalizer_type="standard",
            target_is_delta=False,
            learned_rewards=False,
        )
        batch = _make_batch(self.N, self.Do, self.Da, self.H)
        one_d.update_normalizer(batch)
        assert not one_d._uses_robust_normalizer
        model_in = one_d._get_model_input(batch.obs, batch.act)
        expected_dim = self.Do * self.H + self.Da * self.H
        assert model_in.shape == (self.N, expected_dim)
        assert torch.isfinite(model_in).all()
