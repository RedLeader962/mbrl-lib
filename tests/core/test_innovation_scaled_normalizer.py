# coding=utf-8
"""Tests for the RLRP-761 ``S4`` innovation-scaled normalization path.

Covers :class:`mbrl.util.normalization.InnovationScaledNormalizer` (``S4.1``),
its registration in ``create_normalizer`` (``S4.2``), the decoupled block-facade
build and the diagonal AR bridge (``S4.3`` / ``S4.4``), the sequence-id
derivation from the composed history window (``S4.5``), the versioned payload
(``S4.6``) and the ``target_is_delta`` affine identity (``S4.8``).
"""
import math
import pathlib

import numpy as np
import pytest
import torch

from mbrl.util.normalization import (
    InnovationScaledNormalizer,
    ZScoreNormalizer,
    create_normalizer,
)

_DEVICE = torch.device("cpu")


def _ar1_sequences(n_seq, length, phi, innovation_std, seed=0):
    """AR(1) ``x[t+1] = phi x[t] + e``, whose one-step innovation std is KNOWN."""
    rng = np.random.default_rng(seed)
    n_features = len(phi)
    sequences = []
    for _ in range(n_seq):
        x = np.zeros((length, n_features))
        for t in range(1, length):
            x[t] = np.asarray(phi) * x[t - 1] + rng.normal(
                0.0, innovation_std, size=n_features
            )
        sequences.append(x)
    return sequences


def _pool(sequences):
    data = torch.as_tensor(np.concatenate(sequences, axis=0), dtype=torch.float32)
    ids = torch.repeat_interleave(
        torch.arange(len(sequences)),
        torch.as_tensor([s.shape[0] for s in sequences]),
    )
    return data, ids


class TestInnovationScaleEstimation:
    def test_recovers_the_ar1_innovation_std(self):
        # phi ~ 0 => the one-step delta std is sqrt(2) * innovation std;
        # the estimator must track the DELTA, not the state std.
        phi = [0.9, 0.2]
        innovation_std = [0.05, 0.3]
        sequences = _ar1_sequences(12, 400, phi, innovation_std, seed=1)
        data, ids = _pool(sequences)

        norm = InnovationScaledNormalizer(2, _DEVICE)
        norm.update_stats(data, sequence_ids=ids)

        expected = np.concatenate(
            [np.diff(s, axis=0) for s in sequences], axis=0
        ).std(axis=0)
        assert norm.std.reshape(-1).numpy() == pytest.approx(expected, rel=0.02)
        # ... and it is NOT the state std, which is what standard_symmetric uses.
        assert not np.allclose(
            norm.std.reshape(-1).numpy(), norm.state_std.reshape(-1).numpy(), rtol=0.1
        )

    def test_a_boundary_difference_is_never_taken(self):
        # Two sequences with a huge offset between them: pooling them naively
        # would fabricate one enormous "innovation" at the seam.
        a = np.zeros((50, 1))
        b = np.full((50, 1), 1000.0)
        a[:, 0] = np.linspace(0.0, 0.49, 50)
        b[:, 0] = 1000.0 + np.linspace(0.0, 0.49, 50)
        data, ids = _pool([a, b])

        norm = InnovationScaledNormalizer(1, _DEVICE)
        norm.update_stats(data, sequence_ids=ids)
        # Every within-sequence step is exactly 0.01 -> zero-variance delta,
        # floored. A seam-crossing difference of 1000 would blow this up.
        assert float(norm.std.reshape(-1)[0]) < 1e-3

    def test_noise_floor_mode_ignores_a_rare_large_excursion(self):
        # The adverse-event property the research program depends on: a rare
        # large excursion must NOT raise the floor it is measured against.
        rng = np.random.default_rng(7)
        clean = rng.normal(0.0, 0.01, size=(2000, 1))
        spiked = clean.copy()
        spiked[1000] += 5.0
        data_clean, ids = _pool([clean])
        data_spiked, _ = _pool([spiked])

        floor_clean = InnovationScaledNormalizer(
            1, _DEVICE, innovation_scale_mode="noise_floor"
        )
        floor_clean.update_stats(data_clean, sequence_ids=ids)
        floor_spiked = InnovationScaledNormalizer(
            1, _DEVICE, innovation_scale_mode="noise_floor"
        )
        floor_spiked.update_stats(data_spiked, sequence_ids=ids)

        assert float(floor_spiked.std.reshape(-1)[0]) == pytest.approx(
            float(floor_clean.std.reshape(-1)[0]), rel=0.05
        )
        # The one_step_delta estimator, by contrast, DOES react to the spike.
        delta_spiked = InnovationScaledNormalizer(1, _DEVICE)
        delta_spiked.update_stats(data_spiked, sequence_ids=ids)
        delta_clean = InnovationScaledNormalizer(1, _DEVICE)
        delta_clean.update_stats(data_clean, sequence_ids=ids)
        assert float(delta_spiked.std.reshape(-1)[0]) > 2.0 * float(
            delta_clean.std.reshape(-1)[0]
        )

    def test_without_sequence_ids_it_degrades_to_standard_symmetric_and_warns(self):
        data, _ = _pool(_ar1_sequences(4, 100, [0.9], [0.1], seed=2))
        norm = InnovationScaledNormalizer(1, _DEVICE)
        with pytest.warns(RuntimeWarning, match="STATE std"):
            norm.update_stats(data)
        assert torch.allclose(norm.std, norm.state_std)
        assert float(norm.bridge_gain.reshape(-1)[0]) == pytest.approx(1.0)

    def test_explicit_mode(self):
        norm = InnovationScaledNormalizer(
            2, _DEVICE, innovation_scale_mode="explicit", innovation_scale=[0.5, 2.0]
        )
        data, ids = _pool(_ar1_sequences(3, 60, [0.5, 0.5], [0.2, 0.2], seed=3))
        norm.update_stats(data, sequence_ids=ids)
        assert norm.std.reshape(-1).numpy() == pytest.approx([0.5, 2.0])

    def test_explicit_mode_requires_the_vector(self):
        with pytest.raises(ValueError, match="requires"):
            InnovationScaledNormalizer(2, _DEVICE, innovation_scale_mode="explicit")

    def test_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="innovation_scale_mode"):
            InnovationScaledNormalizer(2, _DEVICE, innovation_scale_mode="bogus")

    def test_normalize_denormalize_round_trip(self):
        data, ids = _pool(_ar1_sequences(5, 80, [0.8, 0.3], [0.1, 0.4], seed=4))
        norm = InnovationScaledNormalizer(2, _DEVICE)
        norm.update_stats(data, sequence_ids=ids)
        assert torch.allclose(norm.denormalize(norm.normalize(data)), data, atol=1e-5)


class TestFactoryRegistration:
    def test_create_normalizer_builds_the_innovation_type(self):
        norm = create_normalizer(
            "standard_symmetric_innovation", 3, _DEVICE, innovation_scale_floor=1e-4
        )
        assert isinstance(norm, InnovationScaledNormalizer)
        assert norm.innovation_scale_floor == pytest.approx(1e-4)

    def test_legacy_types_are_untouched(self):
        assert type(create_normalizer("standard_symmetric", 3, _DEVICE)) is (
            ZScoreNormalizer
        )


class TestPersistence:
    def test_save_load_round_trip(self, tmp_path: pathlib.Path):
        data, ids = _pool(_ar1_sequences(5, 80, [0.8, 0.3], [0.1, 0.4], seed=5))
        norm = InnovationScaledNormalizer(2, _DEVICE)
        norm.update_stats(data, sequence_ids=ids)
        norm.save(tmp_path)

        restored = InnovationScaledNormalizer(2, _DEVICE)
        restored.load(tmp_path)
        assert torch.allclose(restored.std, norm.std)
        assert torch.allclose(restored.state_std, norm.state_std)
        assert torch.allclose(restored.bridge_gain, norm.bridge_gain)

    def test_v1_payload_falls_back_to_the_state_std(self, tmp_path: pathlib.Path):
        # A checkpoint written by a plain ZScoreNormalizer (no innovation file).
        legacy = ZScoreNormalizer(2, _DEVICE)
        data, _ = _pool(_ar1_sequences(5, 80, [0.8, 0.3], [0.1, 0.4], seed=6))
        legacy.update_stats(data)
        legacy.save(tmp_path)

        restored = InnovationScaledNormalizer(2, _DEVICE)
        with pytest.warns(FutureWarning, match="payload v1"):
            restored.load(tmp_path)
        # Bridge gain == 1 => the checkpoint reproduces standard_symmetric
        # instead of silently mixing two spaces.
        assert torch.allclose(
            restored.bridge_gain, torch.ones_like(restored.bridge_gain)
        )


# ======================================================================================
# Block facade: decoupled scales + AR bridge (S4.3 / S4.4 / S4.5 / S4.8 / M5)
# ======================================================================================
class _FakeMultistepModel(torch.nn.Module):
    def __init__(self, obs_len=2, act_len=1, history_len=4, horizon_len=1):
        super().__init__()
        self.singlestep_obs_len = obs_len
        self.singlestep_act_len = act_len
        self.history_len = history_len
        self.horizon_len = horizon_len
        self.in_size = (obs_len + act_len) * history_len
        self.out_size = obs_len * history_len
        self.device = _DEVICE

    def forward(self, x, *args, **kwargs):  # pragma: no cover - unused
        return x


def _build_wrapper(normalizer_type, double_precision=False, **normalizer_kwargs):
    from mbrl.models.one_dim_tr_model import OneDTransitionRewardModel

    return OneDTransitionRewardModel(
        _FakeMultistepModel(),
        target_is_delta=False,
        normalize=True,
        normalize_double_precision=double_precision,
        learned_rewards=False,
        normalizer_type=normalizer_type,
        normalizer_kwargs=normalizer_kwargs or None,
    )


def _composed_batch(n_rows=64, obs_len=2, act_len=1, history_len=4, seed=11):
    """A composed multistep batch whose obs windows are genuine AR(1) walks."""
    rng = np.random.default_rng(seed)
    obs_rows, act_rows = [], []
    for _ in range(n_rows):
        window = np.zeros((history_len, obs_len))
        for t in range(1, history_len):
            window[t] = 0.9 * window[t - 1] + rng.normal(0.0, [0.05, 0.4][:obs_len])
        obs_rows.append(
            np.concatenate(
                [window.reshape(-1), rng.normal(0.0, 1.0, act_len * (history_len - 1))]
            )
        )
        act_rows.append(rng.normal(0.0, 1.0, act_len))
    return (
        torch.as_tensor(np.stack(obs_rows), dtype=torch.float32),
        torch.as_tensor(np.stack(act_rows), dtype=torch.float32),
    )


class _Batch:
    def __init__(self, obs, act):
        self.obs = obs
        self.act = act

    def astuple(self):
        return self.obs, self.act, self.obs, None, None, None


class TestDecoupledBlockFacade:
    def test_input_and_target_obs_subs_are_distinct_and_differently_scaled(self):
        wrapper = _build_wrapper("standard_symmetric_innovation")
        obs, act = _composed_batch()
        wrapper.update_normalizer(_Batch(obs, act))

        assert wrapper.uses_decoupled_obs_scales
        input_sub = wrapper.input_normalizer.obs_sub
        target_sub = wrapper.output_normalizer.obs_sub
        assert input_sub is not target_sub
        # The input keeps the well-conditioned state scale ...
        assert isinstance(input_sub, ZScoreNormalizer)
        # ... the target is innovation-scaled, hence a STRICTLY smaller scale
        # for an autocorrelated signal.
        assert torch.all(target_sub.std < input_sub.std)

    def test_the_act_sub_is_decoupled_between_the_two_facades(self):
        """RLRP-761 ``S12.1`` re-baseline (was
        ``test_the_act_sub_is_shared_between_the_two_facades``).

        The original assertion encoded the ``S4.3`` contract, where only the OBS
        block was decoupled and the act block stayed a single shared
        ``standard_symmetric`` sub. ``S12.1`` deliberately decoupled the act
        block too: the MS forecast self-feed predicts and re-injects the
        commands, so the act TARGET is a genuine forecast target and is
        innovation-scaled, while the act INPUT keeps the well-conditioned state
        z-score. The two are reconciled by :attr:`ar_bridge_gain_act`.

        Keeping the old assertion would have pinned a contract the production
        code abandoned, so it is replaced (not deleted) by the current one.
        """
        wrapper = _build_wrapper("standard_symmetric_innovation")
        input_act = wrapper.input_normalizer.act_sub
        target_act = wrapper.output_normalizer.act_sub

        assert input_act is not target_act, (
            "S12.1: the act block must be decoupled under the innovation type."
        )
        assert isinstance(input_act, ZScoreNormalizer)
        assert isinstance(target_act, InnovationScaledNormalizer)
        assert wrapper.uses_decoupled_act_scales

    def test_the_act_facades_share_one_location(self):
        """RLRP-761 ``F-1`` precondition, pinned.

        ``ar_bridge_gain_act`` is a PURE diagonal multiply, which is only the
        exact ``target -> input`` conversion because the same ``mu`` appears in
        both facades' affine maps. ``F-1`` was precisely the violation of that
        premise (the input act sub was fitted on a pooled tensor while the
        target act sub was fitted on the tail only), silently dropping
        ``(mu_target - mu_input) / sigma_input`` from every bridged act slot.

        The fix lives at the FIT (``update_stats(location_data=...)``), so this
        test guards the invariant at its source rather than at the bridge.
        """
        wrapper = _build_wrapper("standard_symmetric_innovation")
        obs, act = _composed_batch()
        wrapper.update_normalizer(_Batch(obs, act))

        input_mu = wrapper.input_normalizer.act_sub.mean.reshape(-1)
        target_mu = wrapper.output_normalizer.act_sub.mean.reshape(-1)
        assert torch.allclose(input_mu, target_mu, rtol=0.0, atol=0.0), (
            "F-1 regression: the act facades no longer share one location, so "
            "the pure-diagonal act bridge silently drops the offset. Fit the "
            "target act sub with ``location_data=`` (see _update_act_subs)."
        )

    def test_the_act_bridge_is_the_exact_target_to_input_conversion(self):
        """``S12.4`` — the act analogue of
        :meth:`test_bridge_is_the_exact_target_to_input_conversion`.

        This is the behavioural consequence of the two tests above: with the
        location shared, the diagonal gain reproduces the INPUT-space value
        exactly. Had ``F-1`` still been live, this identity would fail by the
        dropped offset.
        """
        wrapper = _build_wrapper(
            "standard_symmetric_innovation", double_precision=True
        )
        obs, act = _composed_batch()
        wrapper.update_normalizer(_Batch(obs, act))

        gain = wrapper.ar_bridge_gain_act
        assert gain is not None, "S12.4: the act bridge gain must be registered."

        raw = act[:2].to(torch.float64)
        target_space = wrapper.output_normalizer.act_sub.normalize(raw)
        via_bridge = target_space * gain.to(torch.float64)
        direct = wrapper.input_normalizer.act_sub.normalize(raw)
        ulp = 4 * torch.finfo(torch.float64).eps
        assert torch.allclose(via_bridge, direct, rtol=ulp, atol=1e-12)

    def test_bridge_gain_equals_the_scale_ratio(self):
        wrapper = _build_wrapper("standard_symmetric_innovation")
        obs, act = _composed_batch()
        wrapper.update_normalizer(_Batch(obs, act))

        expected = (
            wrapper.output_normalizer.obs_sub.std.reshape(-1)
            / wrapper.input_normalizer.obs_sub.std.reshape(-1)
        )
        assert torch.allclose(wrapper.ar_bridge_gain, expected, atol=1e-7)

    def test_bridge_is_the_exact_target_to_input_conversion(self):
        # The invariant S4.4 rests on: converting a TARGET-space value with the
        # diagonal gain must equal renormalizing its raw value in INPUT space.
        # Risk ``R-C`` is stated in float64, so the normalizer statistics are
        # fitted in double precision here; in the default float32 build the
        # same identity holds to float32 rounding instead.
        wrapper = _build_wrapper(
            "standard_symmetric_innovation", double_precision=True
        )
        obs, act = _composed_batch()
        wrapper.update_normalizer(_Batch(obs, act))

        raw = torch.tensor([[0.3, -1.2], [0.0, 0.7]], dtype=torch.float64)
        target_space = wrapper.output_normalizer.obs_sub.normalize(raw)
        via_bridge = target_space * wrapper.ar_bridge_gain.to(torch.float64)
        direct = wrapper.input_normalizer.obs_sub.normalize(raw)
        # 4 ULP relative to the magnitude of the compared values (R-C).
        ulp = 4 * torch.finfo(torch.float64).eps
        assert torch.allclose(via_bridge, direct, rtol=ulp, atol=0.0)

    @pytest.mark.parametrize(
        "normalizer_type", ["standard_symmetric", "winsorized", "quantile"]
    )
    def test_legacy_types_stay_coupled_and_gainless(self, normalizer_type):
        # Measure M5: the S4 machinery must be strictly inert elsewhere.
        wrapper = _build_wrapper(normalizer_type)
        assert not wrapper.uses_decoupled_obs_scales
        assert wrapper.ar_bridge_gain is None
        assert wrapper.input_normalizer.obs_sub is wrapper.output_normalizer.obs_sub

    def test_standard_type_has_no_gain(self):
        wrapper = _build_wrapper("standard")
        assert wrapper.ar_bridge_gain is None

    def test_sequence_ids_come_from_the_composed_history_window(self):
        wrapper = _build_wrapper("standard_symmetric_innovation")
        obs, _ = _composed_batch(n_rows=3)
        obs_block = obs[..., : 2 * 4].reshape(-1, 2)
        ids = wrapper._composed_history_sequence_ids(obs_block, 4)
        assert ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    def test_unit_norm_dims_keep_a_unit_gain(self):
        # A quaternion / gravity block is a pass-through in BOTH spaces, so
        # dividing it by an innovation would be a correctness bug.
        wrapper = _build_wrapper(
            "standard_symmetric_innovation",
            feature_dim_names=["a", "b", "u"],
            normalize_dims={"a": True, "b": True, "u": False},
            per_dim_strategy=["inherit", "inherit", "unit_norm"],
        )
        obs, act = _composed_batch(obs_len=2, act_len=1)
        wrapper.update_normalizer(_Batch(obs, act))
        # ``u`` is the single act dim here; the obs block carries the two
        # ``inherit`` dims, so the obs gain must be entirely non-unit.
        assert wrapper.ar_bridge_gain.numel() == 2

    def test_target_is_delta_is_the_exact_innovation_delta(self):
        # S4.8: the transform is affine, so normalize(y) - normalize(x) is
        # exactly (y - x)/s -- "the delta expressed in innovation units".
        wrapper = _build_wrapper("standard_symmetric_innovation")
        obs, act = _composed_batch()
        wrapper.update_normalizer(_Batch(obs, act))
        sub = wrapper.output_normalizer.obs_sub

        x = torch.tensor([[0.1, -0.4]], dtype=torch.float64)
        y = torch.tensor([[0.3, 0.2]], dtype=torch.float64)
        lhs = sub.normalize(y) - sub.normalize(x)
        rhs = (y - x) / sub.std.to(torch.float64)
        assert torch.allclose(lhs, rhs, atol=1e-12)

    def test_target_is_delta_emits_no_unsupported_warning(self):
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", RuntimeWarning)
            from mbrl.models.one_dim_tr_model import OneDTransitionRewardModel

            OneDTransitionRewardModel(
                _FakeMultistepModel(),
                target_is_delta=True,
                normalize=True,
                learned_rewards=False,
                normalizer_type="standard_symmetric_innovation",
            )


class TestStrategyWrapperKeywordForwarding:
    def test_sequence_ids_are_not_forwarded_to_a_base_that_rejects_them(self):
        # Regression: the wrapper is transparent, so the caller cannot know
        # which base it is talking to; forwarding blindly crashed every
        # strategy-wrapped LEGACY normalizer.
        from mbrl.util.normalization import StrategyAwareNormalizer

        base = ZScoreNormalizer(2, _DEVICE)
        wrapper = StrategyAwareNormalizer(base, ["zscore", "zscore"])
        data, ids = _pool(_ar1_sequences(3, 40, [0.5, 0.5], [0.2, 0.2], seed=8))
        wrapper.update_stats(data, sequence_ids=ids)  # must not raise
        assert torch.allclose(base.std, data.std(0, keepdim=True), rtol=1e-4)

    def test_sequence_ids_reach_an_innovation_base(self):
        from mbrl.util.normalization import StrategyAwareNormalizer

        base = InnovationScaledNormalizer(2, _DEVICE)
        wrapper = StrategyAwareNormalizer(base, ["zscore", "zscore"])
        data, ids = _pool(_ar1_sequences(6, 200, [0.9, 0.9], [0.1, 0.1], seed=9))
        wrapper.update_stats(data, sequence_ids=ids)
        assert not torch.allclose(base.std, base.state_std, rtol=0.1)


class TestNoSilentFallbackInProduction:
    def test_the_wrapper_fit_never_falls_back(self, recwarn):
        # The composed history window always supplies sequence structure, so
        # the "shuffled transitions" warning must NOT fire on the real path.
        wrapper = _build_wrapper("standard_symmetric_innovation")
        obs, act = _composed_batch()
        wrapper.update_normalizer(_Batch(obs, act))
        assert not [
            w for w in recwarn if "STATE std" in str(w.message)
        ]
