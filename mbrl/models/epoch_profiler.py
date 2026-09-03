# coding=utf-8
"""Per-epoch training profiler for :class:`~mbrl.models.ModelTrainer`.

RLRP-775 action **A11** (plan ``perf_tcn_ms2ss_training_speed_RLRP-775.md``): before
any epoch-level speed-up can be promised, one REAL training epoch must be profiled on
the actual training platform (Valeria / Compute Canada x86 + A100). The open questions
the plan cannot answer without this are:

* is the run GPU-bound at all, or is it waiting on the ``DataLoader``?
* how much of an epoch is *validation* (the AR unroll runs there too, without backward)?
* what share of a training step is the TCN encoder (``Conv1d``), i.e. what is the
  Amdahl ceiling of every encoder-side action (A1/A3/A4/A5/A9)?

The profiler is **fully opt-in and zero-overhead when disabled**: it is only attached
when the ``RLRC_PROFILE_EPOCH`` environment variable is set, so an unmodified
production job is bit-exact and unaffected. Enable it in a Slurm job with::

    --env RLRC_PROFILE_EPOCH=1          # (apptainer) or `export` before `python`

Environment variables
---------------------
``RLRC_PROFILE_EPOCH``
    ``0``/unset -> disabled (default). Anything else -> enabled.
``RLRC_PROFILE_EPOCH_MAX_EPOCHS``
    Stop reporting after this many epochs (default ``2``). Timing is cheap
    (``perf_counter`` per batch) so this mostly bounds the log volume.
``RLRC_PROFILE_EPOCH_TORCH_PROFILER``
    ``1`` -> additionally capture a ``torch.profiler`` window of
    ``RLRC_PROFILE_EPOCH_TORCH_WINDOW`` steps (default ``20``) on the FIRST profiled
    epoch and print the operator table plus the ``Conv1d`` CUDA-time share. Off by
    default because the profiler itself perturbs step time.
``RLRC_PROFILE_EPOCH_TRACE_DIR``
    If set together with the profiler, the Chrome trace is exported there.
"""
import os
import time
from typing import List, Optional

import pytorch_lightning as pl
import torch


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ("", "0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class EpochProfilerCallback(pl.Callback):
    """Measure where a training epoch's wall-clock actually goes.

    Per batch the callback splits the wall-clock into two disjoint parts:

    ``compute``
        ``batch_start -> batch_end``, i.e. forward + backward + optimizer step.
    ``loader wait``
        ``previous batch_end -> next batch_start``, i.e. everything Lightning does
        BETWEEN steps -- dominated by batch assembly / host-to-device transfer.
        This is the number that decides whether any model-side optimisation matters
        at all (A10 was rejected on exactly this measurement).

    Both are accumulated separately for the train and the validation loop, so the
    validation share of an epoch (which runs the same AR unroll, without backward)
    becomes explicit.
    """

    def __init__(
        self,
        max_epochs_reported: int = 2,
        use_torch_profiler: bool = False,
        torch_profiler_window: int = 20,
        trace_dir: Optional[str] = None,
    ):
        self._max_epochs_reported = max(int(max_epochs_reported), 1)
        self._use_torch_profiler = bool(use_torch_profiler)
        self._torch_profiler_window = max(int(torch_profiler_window), 1)
        self._trace_dir = trace_dir

        self._epochs_reported = 0
        self._profiler = None
        self._profiled_steps = 0

        self._reset_epoch_accumulators()

    # ---- bookkeeping -----------------------------------------------------
    def _reset_epoch_accumulators(self) -> None:
        self._t_epoch_start: Optional[float] = None
        self._t_last_batch_end: Optional[float] = None
        self._train_compute_s: float = 0.0
        self._train_wait_s: float = 0.0
        self._train_batches: int = 0
        self._val_compute_s: float = 0.0
        self._val_wait_s: float = 0.0
        self._val_batches: int = 0
        self._t_batch_start: Optional[float] = None

    @property
    def _reporting(self) -> bool:
        return self._epochs_reported < self._max_epochs_reported

    # ---- train loop ------------------------------------------------------
    def on_train_epoch_start(self, trainer, pl_module):
        if not self._reporting:
            return
        self._reset_epoch_accumulators()
        _sync()
        self._t_epoch_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self._use_torch_profiler and self._epochs_reported == 0:
            self._start_torch_profiler()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self._reporting:
            return
        now = time.perf_counter()
        if self._t_last_batch_end is not None:
            self._train_wait_s += now - self._t_last_batch_end
        self._t_batch_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._reporting:
            return
        # A GPU step is asynchronous: without a sync the measured "compute" would be
        # the Python dispatch time only and the queued kernels would leak into the
        # next interval (and be mis-attributed to loader wait).
        _sync()
        now = time.perf_counter()
        if self._t_batch_start is not None:
            self._train_compute_s += now - self._t_batch_start
        self._train_batches += 1
        self._t_last_batch_end = now
        self._step_torch_profiler()

    # ---- validation loop -------------------------------------------------
    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        if not self._reporting:
            return
        now = time.perf_counter()
        if self._t_last_batch_end is not None:
            self._val_wait_s += now - self._t_last_batch_end
        self._t_batch_start = now

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not self._reporting:
            return
        _sync()
        now = time.perf_counter()
        if self._t_batch_start is not None:
            self._val_compute_s += now - self._t_batch_start
        self._val_batches += 1
        self._t_last_batch_end = now

    # ---- report ----------------------------------------------------------
    def on_train_epoch_end(self, trainer, pl_module):
        if not self._reporting:
            return
        self._stop_torch_profiler()
        _sync()
        t_epoch = (
            time.perf_counter() - self._t_epoch_start
            if self._t_epoch_start is not None
            else float("nan")
        )
        self._epochs_reported += 1
        self._print_report(trainer, pl_module, t_epoch)

    def _print_report(self, trainer, pl_module, t_epoch: float) -> None:
        train_total = self._train_compute_s + self._train_wait_s
        val_total = self._val_compute_s + self._val_wait_s
        accounted = train_total + val_total

        def _pct(x: float) -> float:
            return 100.0 * x / t_epoch if t_epoch else float("nan")

        def _ms(x: float, n: int) -> float:
            return 1e3 * x / n if n else float("nan")

        device = getattr(pl_module, "device", "?")
        print(f"\n=== RLRP-775 A11 epoch profile (epoch {trainer.current_epoch}) ===")
        print(f"device                 : {device}")
        print(f"t_epoch                : {t_epoch:.3f} s")
        print(
            f"train  batches         : {self._train_batches}  "
            f"({_ms(self._train_compute_s, self._train_batches):.2f} ms/batch compute)"
        )
        print(
            f"  compute              : {self._train_compute_s:.3f} s  "
            f"({_pct(self._train_compute_s):.1f} % of epoch)"
        )
        print(
            f"  loader wait          : {self._train_wait_s:.3f} s  "
            f"({_pct(self._train_wait_s):.1f} % of epoch)   <-- A10 decision metric"
        )
        print(
            f"val    batches         : {self._val_batches}  "
            f"({_ms(self._val_compute_s, self._val_batches):.2f} ms/batch compute)"
        )
        print(
            f"  compute              : {self._val_compute_s:.3f} s  "
            f"({_pct(self._val_compute_s):.1f} % of epoch)   <-- A14 ceiling"
        )
        print(f"  loader wait          : {self._val_wait_s:.3f} s ({_pct(self._val_wait_s):.1f} %)")
        print(
            f"unaccounted            : {t_epoch - accounted:.3f} s "
            f"({_pct(t_epoch - accounted):.1f} % -- epoch setup, callbacks, "
            f"best-weight snapshot, logging)"
        )
        if torch.cuda.is_available():
            print(
                f"peak CUDA memory       : "
                f"{torch.cuda.max_memory_allocated() / 2**20:.1f} MiB"
            )
        print("=" * 62)

    # ---- optional torch.profiler window ----------------------------------
    def _start_torch_profiler(self) -> None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities, record_shapes=False, with_stack=False
        )
        self._profiler.__enter__()
        self._profiled_steps = 0

    def _step_torch_profiler(self) -> None:
        if self._profiler is None:
            return
        self._profiled_steps += 1
        if self._profiled_steps >= self._torch_profiler_window:
            self._stop_torch_profiler()

    def _stop_torch_profiler(self) -> None:
        if self._profiler is None:
            return
        profiler, self._profiler = self._profiler, None
        profiler.__exit__(None, None, None)

        sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        print(
            f"\n--- torch.profiler ({self._profiled_steps} training steps), "
            f"top 15 by {sort_by} ---"
        )
        print(profiler.key_averages().table(sort_by=sort_by, row_limit=15))

        # Encoder share: the Amdahl ceiling of every conv-stack action (A1/A3/A4/A5/A9).
        conv_us = 0.0
        total_us = 0.0
        for evt in profiler.key_averages():
            device_us = (
                getattr(evt, "self_device_time_total", None)
                or getattr(evt, "self_cuda_time_total", 0.0)
                or 0.0
            ) if torch.cuda.is_available() else evt.self_cpu_time_total
            total_us += float(device_us)
            name = evt.key.lower()
            if "conv" in name:
                conv_us += float(device_us)
        if total_us:
            print(
                f"conv* share of profiled device time: {100.0 * conv_us / total_us:.1f} % "
                f"(=> Amdahl ceiling of encoder-side actions)"
            )

        if self._trace_dir:
            os.makedirs(self._trace_dir, exist_ok=True)
            path = os.path.join(self._trace_dir, "rlrp775_a11_epoch_trace.json")
            profiler.export_chrome_trace(path)
            print(f"chrome trace exported: {path}")


def maybe_build_epoch_profiler_callbacks() -> List[pl.Callback]:
    """Return ``[EpochProfilerCallback]`` iff ``RLRC_PROFILE_EPOCH`` is set.

    Returns an EMPTY list otherwise, so a production run is completely unaffected
    (RLRP-775 action A11).
    """
    if not _env_flag("RLRC_PROFILE_EPOCH"):
        return []
    return [
        EpochProfilerCallback(
            max_epochs_reported=_env_int("RLRC_PROFILE_EPOCH_MAX_EPOCHS", 2),
            use_torch_profiler=_env_flag("RLRC_PROFILE_EPOCH_TORCH_PROFILER"),
            torch_profiler_window=_env_int("RLRC_PROFILE_EPOCH_TORCH_WINDOW", 20),
            trace_dir=os.environ.get("RLRC_PROFILE_EPOCH_TRACE_DIR"),
        )
    ]
