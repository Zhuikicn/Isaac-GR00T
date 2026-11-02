"""Low-frequency, per-rank training timings without per-step CUDA synchronization.

CUDA event intervals include host dispatch gaps and stream dependencies, not just
kernel activity. DataLoader host waits can overlap preceding GPU work. DeepSpeed
engine.step includes clipping, optimizer work and communication; it must not be
mistaken for the no-op optimizer.step exposed to HuggingFace Trainer.
"""

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import logging
import math
import time

import torch
from transformers.trainer_callback import TrainerCallback


class TimedIterator:
    """Measure actual next() calls, excluding Trainer's batch bookkeeping."""

    def __init__(self, iterator, timing):
        self.iterator = iterator
        self.timing = timing

    def __iter__(self):
        return self

    def __next__(self):
        with self.timing.phase("data_wait", always_cpu=True):
            return next(self.iterator)


class TrainingTiming(TrainerCallback):
    PHASES = ("data_wait", "input_prepare", "forward", "backward", "clip", "optimizer")

    def __init__(self, trainer, log_steps=100, sample_steps=5):
        if log_steps < 0 or sample_steps < 1 or (log_steps and sample_steps > log_steps):
            raise ValueError("Timing requires log_steps >= 0 and 1 <= sample_steps <= log_steps")
        self.trainer = trainer
        self.log_steps = log_steps
        self.enabled = log_steps > 0
        self.sample_offsets = (
            {(i * log_steps // sample_steps + 1) % log_steps for i in range(sample_steps)}
            if self.enabled
            else set()
        )
        self.cuda = trainer.args.device.type == "cuda"
        self.active = False
        self.sample = False
        self.current = None
        self.records = []
        self.wrappers = []
        self.last_finish = None
        self.window_start = None
        self.checkpoint_ms = 0.0
        self.deepspeed = False

    def _event(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _wrap(self, owner, name, phase):
        original = getattr(owner, name)

        @wraps(original)
        def measured(*args, **kwargs):
            with self.phase(phase):
                return original(*args, **kwargs)

        setattr(owner, name, measured)
        self.wrappers.append((owner, name, original))

    def close(self):
        for owner, name, original in reversed(self.wrappers):
            setattr(owner, name, original)
        self.wrappers.clear()
        self.active = self.sample = False

    def _log(self, metrics):
        # CallbackHandler.on_log clears should_log. These extra measurements
        # must not suppress Trainer's regular loss logging (and its collectives)
        # on only some ranks. Call on every rank; W&B itself only emits on rank 0.
        should_log = self.trainer.control.should_log
        try:
            self.trainer.log(metrics)
        finally:
            self.trainer.control.should_log = should_log

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        self.close()
        self.records.clear()
        self.reported_startup = False
        self.first_update = True
        self.last_finish = None
        self.checkpoint_ms = 0.0
        self.window_start = time.perf_counter()
        accelerator = self.trainer.accelerator
        engine_wrapper = getattr(accelerator, "deepspeed_engine_wrapped", None)
        self.deepspeed = engine_wrapper is not None
        if self.deepspeed:
            self._wrap(engine_wrapper.engine, "backward", "backward")
            self._wrap(engine_wrapper.engine, "step", "optimizer")
        else:
            self._wrap(accelerator, "backward", "backward")
            self._wrap(accelerator, "clip_grad_norm_", "clip")
            self._wrap(self.trainer.optimizer, "step", "optimizer")
        logging.info(
            "Training timings through Trainer/W&B: every %d updates, CUDA samples at %s",
            self.log_steps,
            sorted(self.sample_offsets),
        )

    def begin_update(self, global_step):
        if not self.enabled or self.window_start is None:
            return
        now = time.perf_counter()
        self.active = True
        # Offset regular samples from the default loss-logging steps (0,10,...).
        self.sample = self.first_update or global_step % self.log_steps in self.sample_offsets
        self.first_update = False
        self.current = {
            "start": now,
            "cpu": defaultdict(float),
            "events": defaultdict(list),
            "gap_ms": 0.0 if self.last_finish is None else (now - self.last_finish) * 1000,
            "sample": self.sample,
            "cuda_start": self._event() if self.sample and self.cuda else None,
        }

    @contextmanager
    def phase(self, name, always_cpu=False):
        if not self.active or not (self.sample or always_cpu):
            yield
            return
        start = time.perf_counter()
        event = self._event() if self.sample and self.cuda else None
        try:
            yield
        finally:
            self.current["cpu"][name] += (time.perf_counter() - start) * 1000
            if event is not None:
                self.current["events"][name].append((event, self._event()))

    def on_step_end(self, args, state, control, **kwargs):
        if not self.active:
            return
        now = time.perf_counter()
        record = self.current
        record["step_cpu_ms"] = (now - record["start"]) * 1000
        if record["cuda_start"] is not None:
            record["events"]["step"].append((record["cuda_start"], self._event()))
        self.records.append(record)
        self.last_finish = now
        self.active = self.sample = False
        # A separate startup row prevents cold DataLoader startup being hidden.
        if len(self.records) == 1 and not getattr(self, "reported_startup", False):
            self.reported_startup = True
            self.flush(state.global_step)
        elif state.global_step % self.log_steps == 0:
            self.flush(state.global_step)

    def checkpoint(self, elapsed_ms):
        if self.enabled and self.window_start is not None:
            self.checkpoint_ms += elapsed_ms
            # Emit at save time as well: the last checkpoint may be followed by
            # train_end without another timing window to carry this duration.
            packed = torch.tensor(
                [elapsed_ms], dtype=torch.float64, device=self.trainer.args.device
            )
            ranks = self.trainer.accelerator.gather(packed).cpu().tolist()
            self._log(
                {
                    **{
                        f"timing/rank{rank}/checkpoint_cpu_ms": value
                        for rank, value in enumerate(ranks)
                    },
                    "timing/max_rank/checkpoint_cpu_ms": max(ranks),
                }
            )

    def flush(self, step):
        if not self.records:
            return
        records = self.records
        samples = [record for record in records if record["sample"]]
        if self.cuda and samples:
            # Only one event synchronization per report; never synchronize each
            # phase or batch. All phase events use this rank's current stream.
            samples[-1]["events"]["step"][-1][1].synchronize()
        count = len(records)
        report_boundary = time.perf_counter()
        waits = sorted(record["cpu"]["data_wait"] for record in records)
        values = {
            "updates": count,
            "sampled_updates": len(samples),
            "wall_ms_per_update": (report_boundary - self.window_start) * 1000 / count,
            "step_cpu_ms": sum(r["step_cpu_ms"] for r in records) / count,
            "between_updates_cpu_ms": sum(r["gap_ms"] for r in records) / count,
            "data_wait_cpu_ms": sum(waits) / count,
            "data_wait_cpu_p95_ms": waits[max(0, math.ceil(0.95 * count) - 1)],
            "data_wait_cpu_max_ms": waits[-1],
            "checkpoint_cpu_ms_total": self.checkpoint_ms,
            "optimizer_includes_clip": int(self.deepspeed),
        }
        for name in self.PHASES[1:]:
            values[f"{name}_cpu_ms"] = sum(r["cpu"][name] for r in samples) / max(1, len(samples))
        if self.cuda:
            for name in (*self.PHASES, "step"):
                values[f"{name}_cuda_ms"] = sum(
                    start.elapsed_time(end)
                    for record in samples
                    for start, end in record["events"][name]
                ) / max(1, len(samples))
        keys = list(values)
        packed = torch.tensor(
            [list(values.values())], dtype=torch.float64, device=self.trainer.args.device
        )
        ranks = self.trainer.accelerator.gather(packed).cpu().tolist()
        metrics = {
            f"timing/rank{rank}/{key}": value
            for rank, row in enumerate(ranks)
            for key, value in zip(keys, row)
        }
        metrics["timing/world_size"] = len(ranks)
        for name in (
            "wall_ms_per_update",
            "data_wait_cpu_ms",
            "data_wait_cpu_max_ms",
            "step_cuda_ms",
        ):
            if name in keys:
                metrics[f"timing/max_rank/{name}"] = max(row[keys.index(name)] for row in ranks)
        self._log(metrics)
        self.records = []
        self.checkpoint_ms = 0.0
        # Include this report's gather/logging in the next wall-clock window.
        self.window_start = report_boundary

    def on_train_end(self, args, state, control, **kwargs):
        if self.enabled:
            self.flush(state.global_step)
        self.close()
