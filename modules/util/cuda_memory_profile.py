"""CUDA memory profiling helpers, gated on the ``LTX2_MEMORY_PROFILE`` env var.

When enabled (env var ``LTX2_MEMORY_PROFILE=1``), records every CUDA
allocation with its Python stack trace. After the caching pass and after
each sampling stage, dumps a snapshot pickle that can be loaded into
https://pytorch.org/memory_viz for a full timeline view, and prints a
human-readable summary to stdout at key checkpoints.

The recorder has measurable runtime cost (each allocation captures a
traceback), so it's strictly opt-in. Off by default.
"""

import os
import time
from pathlib import Path

import torch


_ENABLED = os.environ.get("LTX2_MEMORY_PROFILE", "0").lower() in ("1", "true", "yes")
_RECORDING = False
_DUMP_DIR = Path(os.environ.get("LTX2_MEMORY_PROFILE_DIR", "memory_profiles"))


def is_enabled() -> bool:
    return _ENABLED


def start(max_entries: int = 200_000) -> None:
    """Start recording allocation history. No-op when not enabled."""
    global _RECORDING
    if not _ENABLED or _RECORDING or not torch.cuda.is_available():
        return
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=max_entries,
    )
    _RECORDING = True
    print("[MemProfile] recording enabled (max_entries={})".format(max_entries), flush=True)


def stop() -> None:
    """Stop recording allocation history."""
    global _RECORDING
    if not _ENABLED or not _RECORDING:
        return
    torch.cuda.memory._record_memory_history(enabled=None)
    _RECORDING = False


def dump(label: str) -> Path | None:
    """Dump the current allocation history to a pickle.

    Returns the path written, or None if profiling is disabled. Open the
    pickle at https://pytorch.org/memory_viz for a visual timeline.
    """
    if not _ENABLED or not torch.cuda.is_available():
        return None
    _DUMP_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
    path = _DUMP_DIR / f"mem_{int(time.time())}_{safe_label}.pickle"
    torch.cuda.memory._dump_snapshot(str(path))
    print(f"[MemProfile] snapshot dumped: {path}", flush=True)
    return path


def print_stats(label: str) -> None:
    """Print a one-line summary of current allocator state.

    Format: alloc / peak_alloc / reserved / peak_reserved / num_segments / num_alloc_retries.
    """
    if not _ENABLED or not torch.cuda.is_available():
        return
    s = torch.cuda.memory_stats()
    line = (
        "[MemProfile {label}] "
        "alloc={a:.2f} peak={pa:.2f} reserved={r:.2f} peak_reserved={pr:.2f} "
        "segments={seg} retries={ret} oom={oom}"
    ).format(
        label=label,
        a=s.get("allocated_bytes.all.current", 0) / 1e9,
        pa=s.get("allocated_bytes.all.peak", 0) / 1e9,
        r=s.get("reserved_bytes.all.current", 0) / 1e9,
        pr=s.get("reserved_bytes.all.peak", 0) / 1e9,
        seg=s.get("num_alloc_retries", 0) and "?" or s.get("segment.all.current", 0),
        ret=s.get("num_alloc_retries", 0),
        oom=s.get("num_ooms", 0),
    )
    print(line, flush=True)


def print_summary(label: str, abbreviated: bool = True) -> None:
    """Print torch.cuda.memory_summary() — verbose breakdown by size class."""
    if not _ENABLED or not torch.cuda.is_available():
        return
    print(f"[MemProfile {label}] memory_summary:", flush=True)
    print(torch.cuda.memory_summary(abbreviated=abbreviated), flush=True)


def reset_peak() -> None:
    """Reset peak counters so the next print_stats reflects only what follows."""
    if not _ENABLED or not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()


def install_caching_probe(
    module: torch.nn.Module,
    label: str,
    stats_every: int = 25,
    dump_after: int = 10,
    summary_at_dump: bool = True,
) -> None:
    """Attach a forward hook to ``module`` that drives the profiler.

    Behavior on each forward call (only when profiling is enabled):
      - Print a one-line memory_stats summary every ``stats_every`` calls.
      - At the ``dump_after``-th call, dump a snapshot pickle and print a
        full memory_summary. After dumping, recording is stopped to bound
        the recorder's own memory cost.

    The hook auto-removes itself after dumping, so subsequent forwards are
    untouched.
    """
    if not _ENABLED:
        return

    state = {"calls": 0, "dumped": False, "handle": None}

    def hook(_mod, _inputs, _outputs):
        if state["dumped"]:
            return
        state["calls"] += 1
        n = state["calls"]
        if n == 1:
            print_stats(f"{label} call#1 (entry)")
        if n % stats_every == 0:
            print_stats(f"{label} call#{n}")
        if n == dump_after:
            print_stats(f"{label} call#{n} (pre-dump)")
            if summary_at_dump:
                print_summary(f"{label} call#{n}", abbreviated=False)
            dump(f"{label}_call{n}")
            stop()
            state["dumped"] = True
            if state["handle"] is not None:
                state["handle"].remove()

    state["handle"] = module.register_forward_hook(hook)
    print(
        f"[MemProfile] probe installed on '{label}' — stats every {stats_every} calls, "
        f"dump at call {dump_after}",
        flush=True,
    )
