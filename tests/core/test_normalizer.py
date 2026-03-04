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

_DEVICE = "cpu"


# ------------------------------------------------------------------ #
#  6.1 — Basic Functionality
# ------------------------------------------------------------------ #
class TestBasicFunctionality:
    def test_init_default(self):
        norm = mbrl.util.math.Normalizer(5, torch.device(_DEVICE))
        assert norm.mean.shape == (1, 5)
        assert norm.std.shape == (1, 5)
        assert torch.allclose(norm.mean, torch.zeros(1, 5))
        assert torch.allclose(norm.std, torch.ones(1, 5))
        assert norm.eps.item() == pytest.approx(1e-5)

    def test_init_double_precision(self):
        norm = mbrl.util.math.Normalizer(
            3, torch.device(_DEVICE), dtype=torch.double
        )
        assert norm.mean.dtype == torch.double
        assert norm.std.dtype == torch.double
        assert norm.eps.item() == pytest.approx(1e-12)

    def test_update_stats_basic(self):
        norm = mbrl.util.math.Normalizer(2, torch.device(_DEVICE))
        data = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] * 10, dtype=torch.float32
        )
        norm.update_stats(data)
        assert torch.allclose(norm.mean, data.mean(0, keepdim=True), atol=1e-5)
        assert torch.allclose(norm.std, data.std(0, keepdim=True), atol=1e-5)

    def test_normalize_basic(self):
        norm = mbrl.util.math.Normalizer(2, torch.device(_DEVICE))
        data = torch.randn(100, 2)
        norm.update_stats(data)
        normalized = norm.normalize(data)
        assert torch.allclose(normalized.mean(0), torch.zeros(2), atol=0.1)

    def test_denormalize_basic(self):
        """Round-trip: denormalize(normalize(x)) ≈ x."""
        norm = mbrl.util.math.Normalizer(3, torch.device(_DEVICE))
        data = torch.randn(100, 3)
        norm.update_stats(data)
        reconstructed = norm.denormalize(norm.normalize(data))
        assert torch.allclose(reconstructed, data, atol=1e-5)

    def test_normalize_numpy_input(self):
        norm = mbrl.util.math.Normalizer(2, torch.device(_DEVICE))
        data_np = np.random.randn(50, 2).astype(np.float32)
        norm.update_stats(data_np)
        result = norm.normalize(data_np)
        assert isinstance(result, torch.Tensor)

    def test_normalize_float_input(self):
        norm = mbrl.util.math.Normalizer(1, torch.device(_DEVICE))
        result = norm.normalize(3.14)
        assert isinstance(result, torch.Tensor)


# ------------------------------------------------------------------ #
#  6.2 — Registered Buffer Integrity (Issue 3)
# ------------------------------------------------------------------ #
class TestBufferIntegrity:
    def test_buffers_in_state_dict(self):
        norm = mbrl.util.math.Normalizer(4, torch.device(_DEVICE))
        data = torch.randn(20, 4)
        norm.update_stats(data)
        sd = norm.state_dict()
        assert "mean" in sd
        assert "std" in sd
        assert "eps" in sd

    def test_buffers_survive_update_stats(self):
        norm = mbrl.util.math.Normalizer(3, torch.device(_DEVICE))
        data1 = torch.randn(20, 3)
        data2 = torch.randn(30, 3)
        norm.update_stats(data1)
        norm.update_stats(data2)
        sd = norm.state_dict()
        assert "mean" in sd
        assert "std" in sd
        assert torch.allclose(sd["mean"], data2.mean(0, keepdim=True), atol=1e-5)

    def test_state_dict_round_trip(self):
        norm1 = mbrl.util.math.Normalizer(3, torch.device(_DEVICE))
        data = torch.randn(50, 3)
        norm1.update_stats(data)

        norm2 = mbrl.util.math.Normalizer(3, torch.device(_DEVICE))
        norm2.load_state_dict(norm1.state_dict())
        assert torch.allclose(norm1.mean, norm2.mean)
        assert torch.allclose(norm1.std, norm2.std)
        assert torch.allclose(norm1.eps, norm2.eps)


# ------------------------------------------------------------------ #
#  6.3 — Save/Load (Issue 2)
# ------------------------------------------------------------------ #
class TestSaveLoad:
    def test_save_load_torch_format(self):
        norm1 = mbrl.util.math.Normalizer(4, torch.device(_DEVICE))
        data = torch.randn(50, 4)
        norm1.update_stats(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            norm1.save(tmpdir)
            norm2 = mbrl.util.math.Normalizer(4, torch.device(_DEVICE))
            norm2.load(tmpdir)

        assert torch.allclose(norm1.mean, norm2.mean)
        assert torch.allclose(norm1.std, norm2.std)
        assert torch.allclose(norm1.eps, norm2.eps)

    def test_save_creates_pt_file(self):
        norm = mbrl.util.math.Normalizer(2, torch.device(_DEVICE))
        with tempfile.TemporaryDirectory() as tmpdir:
            norm.save(tmpdir)
            assert (pathlib.Path(tmpdir) / "env_stats.pt").exists()
            assert not (pathlib.Path(tmpdir) / "env_stats.pickle").exists()

    def test_load_legacy_pickle_fallback(self):
        norm = mbrl.util.math.Normalizer(2, torch.device(_DEVICE))
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
        norm = mbrl.util.math.Normalizer(2, torch.device(_DEVICE))
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                norm.load(tmpdir)


# ------------------------------------------------------------------ #
#  6.4 — Numerical Robustness
# ------------------------------------------------------------------ #
class TestNumericalRobustness:
    def test_update_stats_with_nan(self):
        norm = mbrl.util.math.Normalizer(2, torch.device(_DEVICE))
        data = torch.randn(20, 2)
        data[5, 0] = float("nan")
        with pytest.warns(RuntimeWarning, match="NaN or Inf"):
            norm.update_stats(data)
        assert torch.isfinite(norm.mean).all()
        assert torch.isfinite(norm.std).all()

    def test_update_stats_with_inf(self):
        norm = mbrl.util.math.Normalizer(2, torch.device(_DEVICE))
        data = torch.randn(20, 2)
        data[3, 1] = float("inf")
        with pytest.warns(RuntimeWarning, match="NaN or Inf"):
            norm.update_stats(data)
        assert torch.isfinite(norm.mean).all()
        assert torch.isfinite(norm.std).all()

    def test_update_stats_constant_dimension(self):
        norm = mbrl.util.math.Normalizer(2, torch.device(_DEVICE))
        data = torch.zeros(20, 2)
        data[:, 1] = torch.randn(20)
        norm.update_stats(data)
        # Constant column std should be clamped to eps, not zero
        assert norm.std[0, 0].item() >= norm.eps.item()

    def test_update_stats_single_sample(self):
        norm = mbrl.util.math.Normalizer(3, torch.device(_DEVICE))
        data = torch.tensor([[1.0, 2.0, 3.0]])
        with pytest.warns(RuntimeWarning, match="only 1 samples"):
            norm.update_stats(data)
        assert torch.allclose(norm.std, torch.ones(1, 3))

    def test_update_stats_small_sample_warning(self):
        norm = mbrl.util.math.Normalizer(2, torch.device(_DEVICE))
        data = torch.randn(5, 2)
        with pytest.warns(RuntimeWarning, match="only 5 samples"):
            norm.update_stats(data)

    def test_normalize_degenerate_stats(self):
        norm = mbrl.util.math.Normalizer(2, torch.device(_DEVICE))
        norm.mean.fill_(float("inf"))
        result = norm.normalize(torch.ones(1, 2))
        assert torch.isfinite(result).all()
        assert torch.allclose(result, torch.zeros(1, 2))

    def test_normalize_large_dynamic_range(self):
        norm = mbrl.util.math.Normalizer(2, torch.device(_DEVICE))
        data = torch.zeros(100, 2)
        data[:, 0] = torch.randn(100) * 1e-6
        data[:, 1] = torch.randn(100) * 1e6
        norm.update_stats(data)
        normalized = norm.normalize(data)
        assert torch.isfinite(normalized).all()

    def test_robotic_data_scenario(self):
        """Simulate realistic robotic data with different scales and NaN spikes."""
        norm = mbrl.util.math.Normalizer(4, torch.device(_DEVICE))
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
        norm = mbrl.util.math.Normalizer(3, torch.device(_DEVICE))
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = norm._to_tensor(arr)
        assert isinstance(t, torch.Tensor)
        assert t.device.type == _DEVICE

    def test_to_tensor_float(self):
        norm = mbrl.util.math.Normalizer(1, torch.device(_DEVICE))
        t = norm._to_tensor(3.14)
        assert isinstance(t, torch.Tensor)

    def test_device_transfer(self):
        norm = mbrl.util.math.Normalizer(2, torch.device(_DEVICE))
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
        norm = mbrl.util.math.Normalizer(2, torch.device("mps"))
        data = torch.randn(20, 2, dtype=torch.float64)
        t = norm._to_tensor(data)
        assert t.dtype == torch.float32


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
            model, target_is_delta=True, normalize=True
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
                model2, target_is_delta=True, normalize=True
            )
            wrapper2.load(tmpdir)

        assert torch.allclose(wrapper2.input_normalizer.mean, original_mean)
        assert torch.allclose(wrapper2.input_normalizer.std, original_std)
