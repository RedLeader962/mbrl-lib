#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# Profiling script for ReplayBuffer modernization overhead analysis.
# Measures key operations: add(), add_batch(), get_all(), sample(),
# property accessors, and save/load roundtrip.
"""
Profile ReplayBuffer overhead to quantify the impact of the TorchRL migration.

Usage:
    python tests/profiling/profile_replay_buffer_overhead.py
"""
import sys
import time
import tempfile
import pathlib

import numpy as np
import torch

from mbrl.util.replay_buffer import ReplayBuffer


def _timer(fn, warmup=2, repeats=10):
    """Run *fn* with warmup, return (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def benchmark_add_single(capacity, obs_dim, act_dim, n_adds):
    """Benchmark single-item add() calls."""
    rb = ReplayBuffer(capacity, (obs_dim,), (act_dim,))
    obs = np.random.randn(obs_dim).astype(np.float32)
    act = np.random.randn(act_dim).astype(np.float32)
    next_obs = np.random.randn(obs_dim).astype(np.float32)

    def fn():
        for _ in range(n_adds):
            rb.add(obs, act, next_obs, 1.0, False, False)

    mean_ms, std_ms = _timer(fn, warmup=1, repeats=5)
    per_call_us = (mean_ms / n_adds) * 1e3
    return mean_ms, std_ms, per_call_us


def benchmark_add_batch(capacity, obs_dim, act_dim, batch_size):
    """Benchmark add_batch() with a given batch size."""
    rb = ReplayBuffer(capacity, (obs_dim,), (act_dim,))
    obs = np.random.randn(batch_size, obs_dim).astype(np.float32)
    act = np.random.randn(batch_size, act_dim).astype(np.float32)
    next_obs = np.random.randn(batch_size, obs_dim).astype(np.float32)
    rewards = np.random.randn(batch_size).astype(np.float32)
    terminated = np.zeros(batch_size, dtype=bool)
    truncated = np.zeros(batch_size, dtype=bool)

    def fn():
        rb.cur_idx = 0
        rb.num_stored = 0
        rb.add_batch(obs, act, next_obs, rewards, terminated, truncated)

    mean_ms, std_ms = _timer(fn, warmup=1, repeats=5)
    per_item_us = (mean_ms / batch_size) * 1e3
    return mean_ms, std_ms, per_item_us


def benchmark_add_batch_torch_input(capacity, obs_dim, act_dim, batch_size):
    """Benchmark add_batch() when inputs are already torch tensors."""
    rb = ReplayBuffer(capacity, (obs_dim,), (act_dim,))
    obs = torch.randn(batch_size, obs_dim)
    act = torch.randn(batch_size, act_dim)
    next_obs = torch.randn(batch_size, obs_dim)
    rewards = torch.randn(batch_size)
    terminated = torch.zeros(batch_size, dtype=torch.bool)
    truncated = torch.zeros(batch_size, dtype=torch.bool)

    def fn():
        rb.cur_idx = 0
        rb.num_stored = 0
        rb.add_batch(obs, act, next_obs, rewards, terminated, truncated)

    mean_ms, std_ms = _timer(fn, warmup=1, repeats=5)
    per_item_us = (mean_ms / batch_size) * 1e3
    return mean_ms, std_ms, per_item_us


def benchmark_get_all(capacity, obs_dim, act_dim):
    """Benchmark get_all() on a full buffer."""
    rb = ReplayBuffer(capacity, (obs_dim,), (act_dim,))
    # Fill the buffer
    obs = np.random.randn(capacity, obs_dim).astype(np.float32)
    act = np.random.randn(capacity, act_dim).astype(np.float32)
    next_obs = np.random.randn(capacity, obs_dim).astype(np.float32)
    rewards = np.random.randn(capacity).astype(np.float32)
    terminated = np.zeros(capacity, dtype=bool)
    truncated = np.zeros(capacity, dtype=bool)
    rb.add_batch(obs, act, next_obs, rewards, terminated, truncated)

    def fn_torch():
        rb.get_all(shuffle=False)

    def fn_torch_shuffle():
        rb.get_all(shuffle=True)

    mean_no_shuffle, std_no_shuffle = _timer(fn_torch, warmup=2, repeats=10)[:2]
    mean_shuffle, std_shuffle = _timer(fn_torch_shuffle, warmup=2, repeats=10)[:2]
    return mean_no_shuffle, std_no_shuffle, mean_shuffle, std_shuffle


def benchmark_sample(capacity, obs_dim, act_dim, sample_size):
    """Benchmark sample() on a full buffer."""
    rb = ReplayBuffer(capacity, (obs_dim,), (act_dim,))
    obs = np.random.randn(capacity, obs_dim).astype(np.float32)
    act = np.random.randn(capacity, act_dim).astype(np.float32)
    next_obs = np.random.randn(capacity, obs_dim).astype(np.float32)
    rewards = np.random.randn(capacity).astype(np.float32)
    terminated = np.zeros(capacity, dtype=bool)
    truncated = np.zeros(capacity, dtype=bool)
    rb.add_batch(obs, act, next_obs, rewards, terminated, truncated)

    def fn():
        rb.sample(sample_size)

    mean_ms, std_ms = _timer(fn, warmup=2, repeats=10)
    return mean_ms, std_ms


def benchmark_property_accessors(capacity, obs_dim, act_dim):
    """Benchmark property accessors (obs, action, reward, etc.)."""
    rb = ReplayBuffer(capacity, (obs_dim,), (act_dim,))
    obs = np.random.randn(capacity, obs_dim).astype(np.float32)
    act = np.random.randn(capacity, act_dim).astype(np.float32)
    next_obs = np.random.randn(capacity, obs_dim).astype(np.float32)
    rewards = np.random.randn(capacity).astype(np.float32)
    terminated = np.zeros(capacity, dtype=bool)
    truncated = np.zeros(capacity, dtype=bool)
    rb.add_batch(obs, act, next_obs, rewards, terminated, truncated)

    results = {}
    for prop_name in ["obs", "action", "reward", "next_obs", "terminated", "truncated"]:
        def fn(name=prop_name):
            getattr(rb, name)
        mean_ms, std_ms = _timer(fn, warmup=2, repeats=10)
        results[prop_name] = (mean_ms, std_ms)
    return results


def benchmark_save_load(capacity, obs_dim, act_dim):
    """Benchmark save/load roundtrip."""
    rb = ReplayBuffer(capacity, (obs_dim,), (act_dim,))
    obs = np.random.randn(capacity, obs_dim).astype(np.float32)
    act = np.random.randn(capacity, act_dim).astype(np.float32)
    next_obs = np.random.randn(capacity, obs_dim).astype(np.float32)
    rewards = np.random.randn(capacity).astype(np.float32)
    terminated = np.zeros(capacity, dtype=bool)
    truncated = np.zeros(capacity, dtype=bool)
    rb.add_batch(obs, act, next_obs, rewards, terminated, truncated)

    with tempfile.TemporaryDirectory() as tmpdir:
        def fn_save():
            rb.save(tmpdir)

        def fn_load():
            rb2 = ReplayBuffer(capacity, (obs_dim,), (act_dim,))
            rb2.load(tmpdir)

        # Save once for load benchmark
        rb.save(tmpdir)

        save_mean, save_std = _timer(fn_save, warmup=1, repeats=5)
        load_mean, load_std = _timer(fn_load, warmup=1, repeats=5)

    return save_mean, save_std, load_mean, load_std


def benchmark_dtype_conversion():
    """Test the dtype conversion pattern used in __init__."""
    dtypes_to_test = [np.float32, np.float64, np.int32]
    results = {}
    for dt in dtypes_to_test:
        try:
            torch_dt = getattr(torch, str(np.dtype(dt)))
            results[str(np.dtype(dt))] = (str(torch_dt), "OK")
        except AttributeError as e:
            results[str(np.dtype(dt))] = (None, f"FAIL: {e}")

    # Test what happens with torch dtypes (the problematic case)
    torch_dtypes = [torch.float32, torch.float64]
    for dt in torch_dtypes:
        try:
            torch_dt = getattr(torch, str(np.dtype(dt)))
            results[f"torch.{dt}"] = (str(torch_dt), "OK")
        except Exception as e:
            results[f"torch.{dt}"] = (None, f"FAIL: {e}")

    return results


def main():
    print("=" * 78)
    print("ReplayBuffer Profiling Benchmark")
    print(f"PyTorch {torch.__version__}")
    print(f"Device: CPU")
    print("=" * 78)

    # Configuration scenarios
    configs = [
        {"label": "Small (1K, obs=4, act=2)", "capacity": 1000, "obs_dim": 4, "act_dim": 2},
        {"label": "Medium (10K, obs=18, act=7)", "capacity": 10000, "obs_dim": 18, "act_dim": 7},
        {"label": "Large (100K, obs=18, act=7)", "capacity": 100000, "obs_dim": 18, "act_dim": 7},
    ]

    # ---- Benchmark 1: add() single item ----
    print("\n--- Benchmark 1: add() single item ---")
    print(f"{'Config':<35} {'Total (ms)':<15} {'Per-call (µs)':<15}")
    print("-" * 65)
    for cfg in configs:
        n_adds = min(cfg["capacity"], 1000)
        mean_ms, std_ms, per_call = benchmark_add_single(
            cfg["capacity"], cfg["obs_dim"], cfg["act_dim"], n_adds
        )
        print(f"{cfg['label']:<35} {mean_ms:>8.2f}±{std_ms:.2f}  {per_call:>10.1f}")

    # ---- Benchmark 2: add_batch() ----
    print("\n--- Benchmark 2: add_batch() (numpy input) ---")
    print(f"{'Config':<35} {'Total (ms)':<15} {'Per-item (µs)':<15}")
    print("-" * 65)
    for cfg in configs:
        batch_size = cfg["capacity"]
        mean_ms, std_ms, per_item = benchmark_add_batch(
            cfg["capacity"], cfg["obs_dim"], cfg["act_dim"], batch_size
        )
        print(f"{cfg['label']:<35} {mean_ms:>8.2f}±{std_ms:.2f}  {per_item:>10.1f}")

    # ---- Benchmark 2b: add_batch() with torch input ----
    print("\n--- Benchmark 2b: add_batch() (torch input) ---")
    print(f"{'Config':<35} {'Total (ms)':<15} {'Per-item (µs)':<15}")
    print("-" * 65)
    for cfg in configs:
        batch_size = cfg["capacity"]
        mean_ms, std_ms, per_item = benchmark_add_batch_torch_input(
            cfg["capacity"], cfg["obs_dim"], cfg["act_dim"], batch_size
        )
        print(f"{cfg['label']:<35} {mean_ms:>8.2f}±{std_ms:.2f}  {per_item:>10.1f}")

    # ---- Benchmark 3: get_all() ----
    print("\n--- Benchmark 3: get_all() ---")
    print(f"{'Config':<35} {'No-shuffle (ms)':<18} {'Shuffle (ms)':<18}")
    print("-" * 71)
    for cfg in configs:
        ns_mean, ns_std, s_mean, s_std = benchmark_get_all(
            cfg["capacity"], cfg["obs_dim"], cfg["act_dim"]
        )
        print(f"{cfg['label']:<35} {ns_mean:>8.2f}±{ns_std:.2f}      {s_mean:>8.2f}±{s_std:.2f}")

    # ---- Benchmark 4: sample() ----
    print("\n--- Benchmark 4: sample() ---")
    print(f"{'Config':<35} {'256-sample (ms)':<18} {'1024-sample (ms)':<18}")
    print("-" * 71)
    for cfg in configs:
        s256_mean, s256_std = benchmark_sample(
            cfg["capacity"], cfg["obs_dim"], cfg["act_dim"], 256
        )
        s1024_mean, s1024_std = benchmark_sample(
            cfg["capacity"], cfg["obs_dim"], cfg["act_dim"], 1024
        )
        print(f"{cfg['label']:<35} {s256_mean:>8.2f}±{s256_std:.2f}      {s1024_mean:>8.2f}±{s1024_std:.2f}")

    # ---- Benchmark 5: Property accessors ----
    print("\n--- Benchmark 5: Property accessors (10K buffer) ---")
    cfg = configs[1]  # Medium config
    results = benchmark_property_accessors(cfg["capacity"], cfg["obs_dim"], cfg["act_dim"])
    print(f"{'Property':<20} {'Time (ms)':<15}")
    print("-" * 35)
    for prop, (mean_ms, std_ms) in results.items():
        print(f"{prop:<20} {mean_ms:>8.2f}±{std_ms:.2f}")

    # ---- Benchmark 6: Save/Load roundtrip ----
    print("\n--- Benchmark 6: Save/Load roundtrip ---")
    print(f"{'Config':<35} {'Save (ms)':<15} {'Load (ms)':<15}")
    print("-" * 65)
    for cfg in configs:
        save_mean, save_std, load_mean, load_std = benchmark_save_load(
            cfg["capacity"], cfg["obs_dim"], cfg["act_dim"]
        )
        print(f"{cfg['label']:<35} {save_mean:>8.2f}±{save_std:.2f}  {load_mean:>8.2f}±{load_std:.2f}")

    # ---- Benchmark 7: Dtype conversion ----
    print("\n--- Benchmark 7: Dtype conversion pattern ---")
    results = benchmark_dtype_conversion()
    print(f"{'Input dtype':<25} {'Torch dtype':<20} {'Status':<15}")
    print("-" * 60)
    for dt, (torch_dt, status) in results.items():
        print(f"{dt:<25} {str(torch_dt):<20} {status:<15}")

    # ---- Summary estimates ----
    print("\n--- Cumulative overhead estimates ---")
    print("Scenario: 10K buffer, obs=18, act=7, typical training pipeline")
    # Add batch for aggregate (called during setup)
    ab_mean, _, ab_per = benchmark_add_batch(10000, 18, 7, 10000)
    # get_all for iterators (called each ERLL epoch)
    ga_mean, _, _, _ = benchmark_get_all(10000, 18, 7)
    print(f"  add_batch(10K items):       {ab_mean:>8.2f} ms  ({ab_per:.1f} µs/item)")
    print(f"  get_all(10K, no shuffle):   {ga_mean:>8.2f} ms")
    print(f"  Per ERLL epoch overhead:    ~{ga_mean:.1f} ms (get_all for train/val split)")
    print(f"  100 ERLL epochs total:      ~{ga_mean * 100 / 1000:.2f} s")

    print("\n" + "=" * 78)
    print("Profiling complete.")


if __name__ == "__main__":
    main()
