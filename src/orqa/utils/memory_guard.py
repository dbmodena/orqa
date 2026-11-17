"""Memory guards for runs on a shared machine without swap.

On 2026-11-16 the nyc statements run froze the whole machine for two hours:
the kernel's OOM killer acts only after the page cache is gone, and with no
swap the machine thrashes long before that. The run's own cgroup cap (85 GB)
never applied, because the other users already held 25-60 GB of the 125.
So the guards here measure what the MACHINE still has free, not only the cap:

- ``check_load``: before a table is read, its in-memory size is estimated
  from a decoded sample of rows; a table that would leave less than the
  floor free is refused (``InsufficientMemoryError``), and the group is
  skipped with no entry written, so a later run retries it.
- ``sandbox_should_stop``: the code sandbox's parent stops a child whose
  run takes the machine under the floor (``QueryValidator.run_in_sandbox``).
- ``start_watchdog``: a thread that, under a lower hard floor, kills the
  sandbox children and then, if memory is still short, ends the run itself:
  a stopped run resumes from its saved groups, a frozen machine stops everyone.
- ``release_unused``: Arrow's allocator (mimalloc) keeps freed memory for
  reuse; after a 44 GB table the run held ~40 GB it no longer used through
  the next groups. Released after every group and before every check.

Floors: ``ORQA_MEMORY_FLOOR_GB`` (default 8) and ``ORQA_MEMORY_HARD_FLOOR_GB``
(default 3).
"""
from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

GB = 1024 ** 3


def _floor_from_env(name: str, default_gb: float) -> int:
    try:
        return int(float(os.environ.get(name, default_gb)) * GB)
    except ValueError:
        return int(default_gb * GB)


# Free memory the machine must keep: loads and sandboxed code stop above it.
FLOOR_BYTES = _floor_from_env("ORQA_MEMORY_FLOOR_GB", 8)
# Below this the run stops itself (after killing its sandbox children).
HARD_FLOOR_BYTES = _floor_from_env("ORQA_MEMORY_HARD_FLOOR_GB", 3)

# Peak while reading a table / the table's in-memory size: Arrow buffers and
# the pandas frame coexist during the conversion. Measured 1.54 on nyc
# 74wj-s5ij (69 / 44.8 GB) and 1.47 on 8h5j-fqxa (4.9 / 3.35 GB).
LOAD_PEAK_FACTOR = 1.6
# Rows decoded to estimate a parquet table's size per row (an instant read;
# 3.5 GB estimated vs 3.35 GB measured on 8h5j-fqxa, 42.0 vs 44.8 on 74wj-s5ij).
# The file's own metadata cannot say: dictionary-encoded text is 5x smaller
# in the file than decoded (8h5j-fqxa: 0.64 GB "uncompressed", 3.35 GB frame).
SAMPLE_ROWS = 20_000
# A CSV's frame per byte of file (text and numbers parsed from text).
CSV_FRAME_PER_FILE_BYTE = 2.0
# The sandbox stops a child only when the machine dropped this much since the
# child started: memory others took while it ran is not its doing.
SANDBOX_MIN_GROWTH_BYTES = 1 * GB
# Exit status of a run the watchdog stopped (run_portal.sh logs it).
WATCHDOG_EXIT_STATUS = 75

MEMINFO = Path("/proc/meminfo")
PROC_SELF_CGROUP = Path("/proc/self/cgroup")
CGROUP_ROOT = Path("/sys/fs/cgroup")


class InsufficientMemoryError(MemoryError):
    """A table that would not fit in the memory the machine has free."""


def gb(n: Optional[float]) -> str:
    return "?" if n is None else f"{n / GB:.1f} GB"


def _machine_available() -> Optional[int]:
    try:
        with open(MEMINFO) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cgroup_headroom() -> Optional[int]:
    """The smallest ``memory.max`` minus usage over this process's cgroup and
    its ancestors (``None`` when none is capped). Page cache counts as usage
    in ``memory.current`` but is reclaimable, so ``file`` is subtracted."""
    try:
        line = next(l for l in PROC_SELF_CGROUP.read_text().splitlines() if l.startswith("0::"))
    except (OSError, StopIteration):
        return None
    node = CGROUP_ROOT / line.split("::", 1)[1].strip().lstrip("/")
    headroom: Optional[int] = None
    while True:
        try:
            limit = (node / "memory.max").read_text().strip()
            if limit != "max":
                used = int((node / "memory.current").read_text())
                for stat in (node / "memory.stat").read_text().splitlines():
                    key, _, value = stat.partition(" ")
                    if key == "file":
                        used -= int(value)
                        break
                room = int(limit) - used
                headroom = room if headroom is None else min(headroom, room)
        except (OSError, ValueError):
            pass
        if node == CGROUP_ROOT or node.parent == node:
            break
        node = node.parent
    return headroom


def available_bytes() -> Optional[int]:
    """What this run can still allocate: the machine's ``MemAvailable``, or
    its cgroup's headroom when that is lower (``None`` when neither is known)."""
    values = [v for v in (_machine_available(), _cgroup_headroom()) if v is not None]
    return min(values) if values else None


def release_unused() -> None:
    """Hand freed memory back to the system (Python garbage, Arrow's pool)."""
    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except Exception:  # noqa: BLE001 — best effort
        pass


def estimate_load_bytes(path: Path) -> Optional[int]:
    """Peak memory of reading ``path`` whole, or ``None`` when unknown."""
    path = Path(path)
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            rows = parquet.metadata.num_rows
            if rows == 0:
                return 0
            batch = next(parquet.iter_batches(batch_size=SAMPLE_ROWS), None)
            if batch is None or batch.num_rows == 0:
                return None
            frame = batch.to_pandas()
            per_row = frame.memory_usage(deep=True).sum() / len(frame)
            return int(per_row * rows * LOAD_PEAK_FACTOR)
        if path.suffix == ".csv":
            return int(path.stat().st_size * CSV_FRAME_PER_FILE_BYTE * LOAD_PEAK_FACTOR)
    except Exception:  # noqa: BLE001 — an unreadable file fails in the reader
        return None
    return None


def check_load(path: Path) -> None:
    """Refuse to read ``path`` when it would leave the machine under the floor."""
    need = estimate_load_bytes(path)
    if not need:
        return
    release_unused()
    free = available_bytes()
    if free is not None and need > free - FLOOR_BYTES:
        raise InsufficientMemoryError(
            f"{Path(path).name} needs ~{gb(need)} to load and {gb(free)} is free "
            f"(the machine keeps {gb(FLOOR_BYTES)})"
        )


def sandbox_should_stop(free_at_start: Optional[int]) -> Optional[int]:
    """The free memory to report when a sandbox child must stop, else ``None``:
    the machine is under the floor and dropped by at least
    ``SANDBOX_MIN_GROWTH_BYTES`` since the child started."""
    free = available_bytes()
    if free is None or free >= FLOOR_BYTES:
        return None
    if free_at_start is not None and free_at_start - free < SANDBOX_MIN_GROWTH_BYTES:
        return None
    return free


def sandbox_memory_feedback(free: Optional[int]) -> str:
    return (
        f"The code was stopped: it took the machine down to {gb(free)} of free memory. "
        "Likely cause: a copy of the whole table (df = Table_0.copy()), parsing or "
        "converting a whole text column, or a join/groupby with a huge intermediate.\n"
        "Fix: select the columns you need and filter rows first, then convert only "
        "those; never copy the whole table."
    )


# ----------------------------------------------------------------------
# Watchdog
# ----------------------------------------------------------------------

_watchdog: Optional[threading.Thread] = None


def _stop_run(free: Optional[int]) -> None:
    message = (
        f"[memory] stopping the run: the machine has {gb(free)} free, under the "
        f"{gb(HARD_FLOOR_BYTES)} hard floor (ORQA_MEMORY_HARD_FLOOR_GB); "
        "saved groups are kept, a restart resumes after them."
    )
    logger.critical(message)
    for stream in (sys.stdout, sys.stderr):
        try:
            print(message, file=stream, flush=True)
        except Exception:  # noqa: BLE001
            pass
    os._exit(WATCHDOG_EXIT_STATUS)


def watchdog_step(settle_seconds: float = 2.0) -> None:
    """One check: under the hard floor, kill the sandbox children; still
    under it after ``settle_seconds``, stop the run."""
    free = available_bytes()
    if free is None or free >= HARD_FLOOR_BYTES:
        return
    children = mp.active_children()
    for child in children:
        logger.warning("[memory] %s free: killing sandbox process %s", gb(free), child.pid)
        child.kill()
    if children:
        time.sleep(settle_seconds)
        free = available_bytes()
        if free is None or free >= HARD_FLOOR_BYTES:
            return
    release_unused()
    free = available_bytes()
    if free is not None and free < HARD_FLOOR_BYTES:
        _stop_run(free)


def start_watchdog(interval: float = 0.5) -> None:
    """Run ``watchdog_step`` every ``interval`` seconds in a daemon thread
    (once per process)."""
    global _watchdog
    if _watchdog is not None and _watchdog.is_alive():
        return

    def loop() -> None:
        while True:
            try:
                watchdog_step()
            except Exception:  # noqa: BLE001 — the watchdog never dies
                logger.exception("[memory] watchdog check failed")
            time.sleep(interval)

    _watchdog = threading.Thread(target=loop, name="orqa-memory-watchdog", daemon=True)
    _watchdog.start()
