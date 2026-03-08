"""Profiling script for ModelTrainer overhead analysis.

Measures:
1. pl.Trainer construction overhead (per-call vs amortized)
2. Gradient cloning overhead per training step (various model sizes)
3. Normalizer non-finite value handling cost

Run inside DNA container:
    dna run develop -- bash -c \
        'cd $DN_PROJECT_PATH/utilities/mbrl-lib && python tests/profiling/profile_model_trainer_overhead.py'
"""
import gc
import sys
import time
import warnings
from contextlib import contextmanager
from typing import List, Tuple

import numpy as np
import torch
import pytorch_lightning as pl


# ── Helpers ──────────────────────────────────────────────────────────────────

@contextmanager
def suppress_pl_warnings():
    """Suppress noisy PL warnings during profiling."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*GPU available but not used.*")
        warnings.filterwarnings("ignore", ".*does not have many workers.*")
        warnings.filterwarnings("ignore", ".*IterableDataset.*__len__.*")
        yield


def time_fn(fn, n_repeats: int = 10, warmup: int = 2) -> Tuple[float, float, List[float]]:
    """Time a callable, returning (mean_ms, std_ms, raw_times_ms)."""
    times = []
    for i in range(warmup + n_repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000.0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std()), times


# ── Benchmark 1: pl.Trainer construction overhead ────────────────────────────

def bench_trainer_construction():
    """Measure cost of creating a new pl.Trainer instance."""
    print("=" * 72)
    print("BENCHMARK 1: pl.Trainer Construction Overhead")
    print("=" * 72)

    # Determine accelerator (same logic as ModelTrainer)
    if torch.cuda.is_available():
        accelerator = "gpu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"

    print(f"  Accelerator: {accelerator}")
    print(f"  PyTorch Lightning version: {pl.__version__}")

    import logging
    _pl_logger = logging.getLogger("pytorch_lightning")
    _prev_level = _pl_logger.level
    _pl_logger.setLevel(logging.ERROR)

    def create_trainer():
        with suppress_pl_warnings():
            t = pl.Trainer(
                max_epochs=50,
                callbacks=[],
                enable_progress_bar=False,
                enable_model_summary=False,
                devices="auto",
                accelerator=accelerator,
                logger=False,
                num_sanity_val_steps=0,
                enable_checkpointing=False,
            )
        return t

    mean_ms, std_ms, raw = time_fn(create_trainer, n_repeats=20, warmup=3)
    _pl_logger.setLevel(_prev_level)

    print(f"  Trainer construction: {mean_ms:.2f} ± {std_ms:.2f} ms  (n=20)")
    print(f"  Min: {min(raw):.2f} ms | Max: {max(raw):.2f} ms")
    print()

    return {"mean_ms": mean_ms, "std_ms": std_ms, "min_ms": min(raw), "max_ms": max(raw)}


# ── Benchmark 2: Gradient cloning overhead ───────────────────────────────────

def _make_model(num_params_approx: int) -> torch.nn.Module:
    """Create a simple model with approximately num_params parameters."""
    # Use a sequential model with linear layers to get close to target param count
    # Each Linear(in, out) has in*out + out parameters
    hidden = int(np.sqrt(num_params_approx / 4))
    hidden = max(hidden, 8)
    model = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
    )
    actual = sum(p.numel() for p in model.parameters())
    return model, actual


def bench_gradient_cloning():
    """Measure cost of cloning all gradients (current on_before_zero_grad behavior)."""
    print("=" * 72)
    print("BENCHMARK 2: Per-Step Gradient Cloning Overhead")
    print("=" * 72)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"  Device: {device}")

    configs = [
        ("Small  (~10K params)", 10_000),
        ("Medium (~100K params)", 100_000),
        ("Large  (~500K params)", 500_000),
        ("XLarge (~1M params)", 1_000_000),
    ]

    results = {}
    for label, target_params in configs:
        model, actual_params = _make_model(target_params)
        model = model.to(device)

        # Simulate a backward pass to create gradients
        x = torch.randn(32, int(np.sqrt(target_params / 4)), device=device)
        x = max(8, x.shape[1])
        x = torch.randn(32, x, device=device) if isinstance(x, int) else x
        out = model(x)
        out.sum().backward()

        # Method A: Current approach — clone every step
        def clone_all_grads():
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    param._last_grad = param.grad.clone()

        # Method B: In-place copy (reuse buffer)
        # First call allocates
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param._buffer_grad = param.grad.clone()

        def inplace_copy_grads():
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    if hasattr(param, "_buffer_grad") and param._buffer_grad is not None:
                        param._buffer_grad.copy_(param.grad)
                    else:
                        param._buffer_grad = param.grad.clone()

        # Method C: Skip entirely (no-op baseline)
        def noop():
            pass

        mean_clone, std_clone, _ = time_fn(clone_all_grads, n_repeats=100, warmup=10)
        mean_inplace, std_inplace, _ = time_fn(inplace_copy_grads, n_repeats=100, warmup=10)
        mean_noop, std_noop, _ = time_fn(noop, n_repeats=100, warmup=10)

        print(f"  {label} (actual: {actual_params:,} params):")
        print(f"    Clone (current):     {mean_clone:.4f} ± {std_clone:.4f} ms/step")
        print(f"    In-place copy:       {mean_inplace:.4f} ± {std_inplace:.4f} ms/step")
        print(f"    No-op (baseline):    {mean_noop:.4f} ± {std_noop:.4f} ms/step")
        print(f"    Clone overhead vs noop: {mean_clone - mean_noop:.4f} ms/step")

        results[label] = {
            "actual_params": actual_params,
            "clone_ms": mean_clone,
            "inplace_ms": mean_inplace,
            "noop_ms": mean_noop,
        }

        # Cleanup
        del model
        gc.collect()

    print()
    return results


# ── Benchmark 3: Normalizer non-finite handling ─────────────────────────────

def bench_normalizer_nonfinite():
    """Measure cost of the torch.where(isfinite(...)) guard in normalize/denormalize."""
    print("=" * 72)
    print("BENCHMARK 3: Normalizer Non-Finite Handling Overhead")
    print("=" * 72)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"  Device: {device}")

    # Simulate normalizer computation
    batch_sizes = [64, 256, 1024]
    obs_dims = [7, 23, 50]

    for batch_size in batch_sizes:
        for obs_dim in obs_dims:
            mean = torch.randn(obs_dim, device=device)
            std = torch.randn(obs_dim, device=device).abs() + 1e-5
            eps = torch.tensor(1e-5, device=device)
            data = torch.randn(batch_size, obs_dim, device=device)

            # Method A: Current — normalize + torch.where guard
            def normalize_with_guard():
                result = (data - mean) / (std + eps)
                result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
                return result

            # Method B: Without guard
            def normalize_without_guard():
                result = (data - mean) / (std + eps)
                return result

            # Method C: Warn-and-clamp (recommended approach)
            def normalize_with_clamp():
                result = (data - mean) / (std + eps)
                if not torch.isfinite(result).all():
                    finite_mask = torch.isfinite(result)
                    if finite_mask.any():
                        lo = result[finite_mask].min()
                        hi = result[finite_mask].max()
                        result = result.clamp(lo, hi)
                    else:
                        result = torch.zeros_like(result)
                return result

            mean_guard, std_guard, _ = time_fn(normalize_with_guard, n_repeats=200, warmup=20)
            mean_no, std_no, _ = time_fn(normalize_without_guard, n_repeats=200, warmup=20)
            mean_clamp, std_clamp, _ = time_fn(normalize_with_clamp, n_repeats=200, warmup=20)

            overhead_pct = ((mean_guard - mean_no) / mean_no * 100) if mean_no > 0 else 0

            print(f"  Batch={batch_size:4d}, Dim={obs_dim:2d}:")
            print(f"    With guard (current):   {mean_guard:.4f} ± {std_guard:.4f} ms")
            print(f"    Without guard:          {mean_no:.4f} ± {std_no:.4f} ms")
            print(f"    Warn-and-clamp (rec.):  {mean_clamp:.4f} ± {std_clamp:.4f} ms")
            print(f"    Guard overhead: {overhead_pct:.1f}%")

    print()


# ── Benchmark 4: Cumulative impact estimate ──────────────────────────────────

def bench_cumulative_impact(trainer_result, grad_result):
    """Estimate cumulative overhead per ERLL epoch."""
    print("=" * 72)
    print("BENCHMARK 4: Cumulative Overhead Estimate per ERLL Epoch")
    print("=" * 72)

    trainer_ms = trainer_result["mean_ms"]

    # Typical RLRC training configs
    configs = [
        ("Math env (light)", 50, 100),      # 50 epochs, ~100 steps/epoch
        ("F110 sim (medium)", 100, 200),     # 100 epochs, ~200 steps/epoch
        ("Robotic (heavy)", 200, 500),       # 200 epochs, ~500 steps/epoch
    ]

    # Use medium model size for gradient estimate
    medium_key = [k for k in grad_result if "Medium" in k][0]
    grad_clone_ms = grad_result[medium_key]["clone_ms"]
    grad_inplace_ms = grad_result[medium_key]["inplace_ms"]

    for label, epochs_per_erll, steps_per_epoch in configs:
        total_steps = epochs_per_erll * steps_per_epoch

        # Trainer construction: once per train() call = once per ERLL epoch
        trainer_overhead = trainer_ms  # ms per ERLL call

        # Gradient cloning: once per training step
        grad_clone_total = grad_clone_ms * total_steps
        grad_inplace_total = grad_inplace_ms * total_steps
        grad_savings = grad_clone_total - grad_inplace_total

        print(f"  {label} ({epochs_per_erll} epochs × {steps_per_epoch} steps/epoch = {total_steps} steps):")
        print(f"    Trainer construction:        {trainer_overhead:.1f} ms")
        print(f"    Grad cloning (current):      {grad_clone_total:.1f} ms ({grad_clone_total/1000:.2f} s)")
        print(f"    Grad cloning (in-place):     {grad_inplace_total:.1f} ms ({grad_inplace_total/1000:.2f} s)")
        print(f"    Potential grad savings:      {grad_savings:.1f} ms ({grad_savings/1000:.2f} s)")
        print(f"    Total overhead (current):    {(trainer_overhead + grad_clone_total)/1000:.2f} s")
        print()

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║      ModelTrainer Overhead Profiling — RLRC / mbrl-lib             ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  PyTorch: {torch.__version__:<20s}  Lightning: {pl.__version__:<20s}  ║")
    print(f"║  CUDA: {'available' if torch.cuda.is_available() else 'not available':<15s}  "
          f"MPS: {'available' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'not available':<16s}  ║")
    print(f"║  Device: {('cuda' if torch.cuda.is_available() else ('mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu')):<58s}  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    trainer_result = bench_trainer_construction()
    grad_result = bench_gradient_cloning()
    bench_normalizer_nonfinite()
    bench_cumulative_impact(trainer_result, grad_result)

    print("=" * 72)
    print("PROFILING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
