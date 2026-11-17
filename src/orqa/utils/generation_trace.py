"""Generation trace: a persistent record of one ``generate-statements`` run,
kept so failures can be investigated after the fact.

The console (``PipelineLogger``) shows a run while it happens, and
``generated_queries.json`` keeps the final state of every group that
finished. Neither keeps what a crashed group, a skipped table or an LLM-level
failure (a truncated answer, a schema violation, a timeout) leaves behind, and
neither is easy to scan across groups. The trace keeps all of it: every stage
of every group appends one JSON line as it happens, so a run killed mid-group
still leaves its trail.

Layout (one folder per run, under ``tasks.query_generation.trace.dir``)::

    <dir>/.gitignore          "*": the folder is never committed, wherever it lives
    <dir>/latest              symlink to the newest run
    <dir>/<run_id>/
        run.json              argv, config, git state, start/end, counts
        events.jsonl          every event in order, each tagged with its group
        groups.jsonl          one line per finished group: status and why
        console.log           the console output, colour codes stripped
        llm/<group>.txt       every LLM call of the group in order, as sent and
                              received: system prompt (a pointer), user
                              message, parameters, reasoning, raw answer
                              (``trace.llm_io``: all, failed, off)
        llm/system/*.txt      each distinct system prompt, written once

Events (the ``event`` field of each line):

    group_start / group_end / group_error   one table group's generation
    anchor_search / table_anchors           the verified keyword anchor (or the miss)
    question_stage_start                    slots, anchors, what the writer and judges see
    question_attempt                        one draft: gate verdict + votes, judge panel, outcome
    question_stage_end                      approved questions, abort status
    question_rewrite                        approved questions sent back after planning failed
    plans                                   questions handed to the planner vs plans returned
    plan_judge_attempt / plan_judge_result  every plan-judge round and each plan's verdict
    code_generated                          the queries one plan produced, with errors
    validator_round                         static validation in/out and its errors
    code_verdict                            one query's validation/judge/execution outcome
    empty_result                            the empty-result plan-level retry
    llm_call                                every structured LLM attempt; raw answer when it failed
    pipeline_log                            PipelineLogger info/warning/error lines
    log                                     Python logging records (WARNING+, INFO+ from OrQA)
    run_error                               the step itself raised

Code anywhere in the pipeline records through the module-level functions
(:func:`trace_event`, :func:`trace_group`, :func:`trace_llm_call`), which do
nothing while no trace is open, so the web app and the unit tests run
untraced. Nothing here may break generation: a failed write disables the
trace with one warning.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path, PurePath
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# Loggers whose INFO records are kept too (everything else: WARNING and up).
_OWN_LOGGERS = ("orqa", "solr", "conf")

# Structured-call outcomes that produced a usable answer
# (LLMClientStructured._validate / the repair stage).
_LLM_OK = {"strict", "coerced", "salvaged", "passthrough", "invalid->repaired"}
# Kept from every call record; the raw answer and the prompt only on failure.
_LLM_FIELDS = (
    "client", "kind", "judge", "attempt", "outcome", "model",
    "native_schema", "max_tokens", "finish_reason", "truncated", "usage", "elapsed_s",
    "raw_chars", "whitespace_ratio", "json_error", "strict_errors", "coerce_errors",
    "fixes", "salvage", "loop_unit", "error", "deployment",
)
_RAW_CAP = 50_000
_PROMPT_CAP = 100_000

# Which LLM calls are written in full to llm/<group>.txt.
LLM_IO_MODES = ("all", "failed", "off")
_RULE_WIDTH = 100

_QUESTION_LAYERS = ("readability", "topic_linkage", "grounding", "table_necessity", "difficulty")
_PLAN_LAYERS = (
    ("plan_approval", "plan"),
    ("table_usage_approval", "tables"),
    ("expected_result_approval", "expected-result"),
    ("metric_combination_approval", "combination"),
)

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_ACTIVE: Optional["GenerationTrace"] = None


# ----------------------------------------------------------------------
# Serialisation
# ----------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """``json.dumps`` fallback for what the pipeline carries around: pydantic
    models, dataclasses, numpy values, paths, exceptions."""
    if isinstance(obj, BaseException):
        return f"{type(obj).__name__}: {obj}"
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # noqa: BLE001
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return dataclasses.asdict(obj)
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def _dumps(record: dict) -> str:
    try:
        return json.dumps(record, ensure_ascii=False, default=_jsonable)
    except (TypeError, ValueError) as exc:
        # e.g. a dict keyed by tuples: keep the event, say why it is flat
        head = {k: record[k] for k in ("seq", "ts", "event", "group") if k in record}
        return json.dumps(
            {**head, "unserializable": f"{type(exc).__name__}: {exc}", "repr": repr(record)[:5000]},
            ensure_ascii=False,
        )


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=_jsonable)
    except (TypeError, ValueError):
        return repr(value)


def _clip(text: Any, cap: int) -> Any:
    if not isinstance(text, str) or len(text) <= cap:
        return text
    return text[:cap] + f"\n… [{len(text) - cap} more characters]"


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


# ----------------------------------------------------------------------
# LLM call files (llm/<group>.txt)
# ----------------------------------------------------------------------

def _section(title: str) -> str:
    head = f"──── {title} "
    return head + "─" * max(4, _RULE_WIDTH - len(head)) + "\n"


def _render_llm_call(seq: Optional[int], record: dict, schema: Optional[str], ok: bool,
                     system_file: Optional[str]) -> str:
    """One call as sent and received, for reading: a header (who, which
    attempt, outcome, deployment, parameters, tokens, what went wrong), then
    the system prompt's file, the user message, the reasoning and the raw
    answer, each exactly as sent or received."""
    who = str(record.get("client") or "?")
    if record.get("judge"):
        who += f" (judge {record['judge']})"
    step = "repair call" if record.get("kind") == "repair" else f"attempt {record.get('attempt', '?')}"
    outcome = str(record.get("outcome") or "?")
    lines = ["═" * _RULE_WIDTH,
             f"call {seq if seq is not None else '?'} · {_now()} · {who} · {step} · {outcome}"
             + ("" if ok else "  ✗ FAILED")]

    model = []
    if record.get("deployment"):
        model.append(f"deployment {record['deployment']}")
    if record.get("model") and record.get("model") != record.get("deployment"):
        model.append(f"served by {record['model']}")
    if schema:
        model.append(f"schema {schema} ({'native' if record.get('native_schema') else 'in prompt'})")
    params = record.get("params") or {}
    if params:
        model.append(", ".join(f"{k}={_as_text(v)}" for k, v in params.items()))
    if model:
        lines.append(" · ".join(model))

    usage = record.get("usage") or {}
    numbers = []
    if usage:
        numbers.append(f"{usage.get('prompt_tokens') or 0:,} prompt + "
                       f"{usage.get('completion_tokens') or 0:,} completion tokens")
    if record.get("elapsed_s") is not None:
        numbers.append(f"{record['elapsed_s']} s")
    if record.get("finish_reason"):
        numbers.append(f"finish {record['finish_reason']}")
    if record.get("truncated"):
        numbers.append("TRUNCATED at the output budget")
    if numbers:
        lines.append(" · ".join(numbers))

    for key, label in (("error", "error"), ("json_error", "JSON error"),
                       ("strict_errors", "schema errors"), ("coerce_errors", "after coercion"),
                       ("fixes", "coercion fixes"), ("salvage", "salvaged"),
                       ("loop_unit", "repeated text")):
        value = record.get(key)
        if value:
            if isinstance(value, list):
                lines.append(f"{label}:")
                lines.extend(f"  - {_as_text(v)}" for v in value)
            else:
                lines.append(f"{label}: {_as_text(value)}")

    out = "\n".join(lines) + "\n"
    system = record.get("system")
    if system_file:
        out += _section(f"SYSTEM → {system_file} ({len(system):,} chars)")
    elif system:
        out += _section(f"SYSTEM ({len(system):,} chars)") + system.rstrip("\n") + "\n"
    prompt = record.get("prompt")
    if prompt is not None:
        prompt = _as_text(prompt)
        out += _section(f"USER ({len(prompt):,} chars)") + prompt.rstrip("\n") + "\n"
    reasoning = record.get("reasoning")
    if reasoning:
        out += _section(f"REASONING ({len(reasoning):,} chars)") + reasoning.rstrip("\n") + "\n"
    if "raw" in record:
        raw = _as_text(record.get("raw") or "")
        out += _section(f"ANSWER ({len(raw):,} chars)") + (raw.rstrip("\n") if raw else "(empty)") + "\n"
    else:
        out += _section("NO ANSWER") + _as_text(record.get("error") or "(no response)") + "\n"
    return out + "\n"


# ----------------------------------------------------------------------
# Group summaries (groups.jsonl)
# ----------------------------------------------------------------------

def question_attempt_failure(attempt: dict) -> Optional[str]:
    """Where one question attempt stopped: ``empty``, a gate (``leak``,
    ``retrieval``, ...), ``judge:<failed layers>``; ``None`` when it passed."""
    if attempt.get("failed_at"):
        return str(attempt["failed_at"])
    gate = attempt.get("gate") or {}
    if gate and not gate.get("approved", False):
        return str(gate.get("failed_at") or "gate")
    judge = attempt.get("judge") or {}
    if judge and not judge.get("approved", False):
        layers = [name for name in _QUESTION_LAYERS if judge.get(f"{name}_approval") is False]
        return "judge" + (f":{','.join(layers)}" if layers else "")
    return None


def plan_attempt_failure(attempt: dict) -> Optional[str]:
    """The plan-judge layers one round failed (``None`` when approved)."""
    if attempt.get("approved"):
        return None
    layers = [label for field, label in _PLAN_LAYERS if attempt.get(field) is False]
    return ",".join(layers) or "rejected"


def summarize_generation(content: Optional[dict]) -> dict:
    """One group's outcome from its generation output: the status, how many
    questions/plans/queries survived, and where the rest stopped."""
    content = content or {}
    result = content.get("result") or {}
    queries = result.get("queries") or []
    failed = result.get("failed_queries") or []
    question_slots = result.get("question_feedback") or []
    plan_entries = result.get("plan_feedback") or []
    errors = content.get("errors") or []

    question_attempts = [
        attempt
        for slot in question_slots
        for attempt in [
            *(slot.get("attempts") or []),
            *(a for replan in slot.get("replans") or [] for a in replan.get("attempts") or []),
        ]
    ]
    plan_attempts = [a for entry in plan_entries for a in entry.get("attempts") or []]
    return {
        "status": result.get("status") or ("success" if queries else "failure"),
        "queries": len(queries),
        "failed_queries": dict(Counter(str(f.get("status", "?")) for f in failed)),
        "questions": {
            "slots": len(question_slots),
            "approved": sum(1 for s in question_slots if s.get("approved")),
            "attempts": len(question_attempts),
            "rewrites": sum(len(s.get("replans") or []) for s in question_slots),
            "failures": dict(Counter(
                f for f in map(question_attempt_failure, question_attempts) if f
            )),
        },
        "plans": {
            "judged": len(plan_entries),
            "approved": sum(1 for e in plan_entries if e.get("approved")),
            "kept_as_fallback": sum(1 for e in plan_entries if e.get("kept_as_fallback")),
            "unjustified_tables": sum(1 for e in plan_entries if e.get("rejected_for_unjustified_tables")),
            "rounds": len(plan_attempts),
            "failures": dict(Counter(
                f for f in map(plan_attempt_failure, plan_attempts) if f
            )),
        },
        "tokens": content.get("token_usage"),
        "model": content.get("model"),
        "n_errors": len(errors),
        "errors": [_clip(_as_text(e), 400) for e in errors[:5]],
    }


# ----------------------------------------------------------------------
# The trace
# ----------------------------------------------------------------------

class _Group:
    """Handle of one running group: :meth:`finish` records its outcome."""

    def __init__(self, trace: Optional["GenerationTrace"], group_id: str, info: dict):
        self.trace = trace
        self.group_id = group_id
        self.info = info
        self.started = time.perf_counter()
        self.finished = False

    def finish(self, content: Optional[dict], **extra: Any) -> None:
        if self.trace is not None and not self.finished:
            self.trace._finish_group(self, summarize_generation(content), extra)
        self.finished = True


class GenerationTrace:
    """One run's trace folder. Usually opened through :func:`open_trace`."""

    def __init__(self, root: Path, run_info: Optional[dict] = None, llm_io: str = "all"):
        if llm_io not in LLM_IO_MODES:
            raise ValueError(f"llm_io must be one of {list(LLM_IO_MODES)}, got {llm_io!r}")
        self.llm_io = llm_io
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        ignore = self.root / ".gitignore"
        if not ignore.exists():
            ignore.write_text("# generation traces are local investigation data\n*\n")

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id, n = stamp, 1
        while (self.root / run_id).exists():
            n += 1
            run_id = f"{stamp}-{n}"
        self.run_id = run_id
        self.dir = self.root / run_id
        self.dir.mkdir()

        self._lock = threading.Lock()
        self._console_lock = threading.Lock()
        self._seq = 0
        self._group_id: Optional[str] = None
        self._events_by_kind: Counter = Counter()
        self._groups_by_status: Counter = Counter()
        self._closed = False
        self._disabled = False
        self._llm_lock = threading.Lock()
        self._llm_dir = self.dir / "llm"
        self._systems: dict[str, str] = {}
        # The running group's banner, written above its first call in llm/.
        self._llm_banner: Optional[str] = None

        self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8", buffering=1)
        self._groups = open(self.dir / "groups.jsonl", "a", encoding="utf-8", buffering=1)
        self.console = open(self.dir / "console.log", "a", encoding="utf-8", buffering=1)

        self._run = {
            "run_id": run_id,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "pid": os.getpid(),
            "git": _git_state(),
            "llm_io": llm_io,
            **(run_info or {}),
        }
        self._write_run()
        self._point_latest()

    # -- events ---------------------------------------------------------

    def event(self, kind: str, /, **payload: Any) -> Optional[int]:
        """Append one event, tagged with the running group; returns its
        ``seq`` (``None`` when nothing was written)."""
        if self._closed or self._disabled:
            return None
        try:
            with self._lock:
                self._seq += 1
                record = {"seq": self._seq, "ts": _now(), "event": kind, "group": self._group_id}
                record.update((k, v) for k, v in payload.items() if k not in record)
                self._events.write(_dumps(record) + "\n")
                self._events_by_kind[kind] += 1
                return self._seq
        except Exception as exc:  # noqa: BLE001
            self._disable(exc)
            return None

    # -- LLM calls in full (llm/) ---------------------------------------

    def llm_file(self) -> str:
        """The running group's LLM call file, relative to the run folder."""
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", self._group_id) if self._group_id else "_run"
        return f"llm/{name}.txt"

    def write_llm_call(self, seq: Optional[int], record: dict, schema: Optional[str], ok: bool) -> None:
        """Append one call, as sent and received, to the group's file."""
        if self._closed or self._disabled or self.llm_io == "off":
            return
        try:
            with self._llm_lock:
                path = self.dir / self.llm_file()
                path.parent.mkdir(parents=True, exist_ok=True)
                block = _render_llm_call(seq, record, schema, ok, self._system_file(record))
                with open(path, "a", encoding="utf-8") as handle:
                    if self._llm_banner:
                        handle.write(self._llm_banner)
                        self._llm_banner = None
                    handle.write(block)
        except Exception as exc:  # noqa: BLE001 — the rest of the trace goes on
            self.llm_io = "off"
            logger.warning("Generation trace: LLM call files disabled after a write error: %s", exc)

    def _system_file(self, record: dict) -> Optional[str]:
        """Each distinct system prompt is written once to llm/system/; the
        call names the file (relative to llm/)."""
        text = record.get("system")
        if not isinstance(text, str) or not text:
            return None
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        name = self._systems.get(digest)
        if name is None:
            label = re.sub(r"[^A-Za-z0-9_-]", "_", str(record.get("client") or "system"))[:40]
            name = f"system/{label}-{digest}.txt"
            target = self._llm_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            self._systems[digest] = name
        return name

    @contextmanager
    def group(self, group_id: str, /, **info: Any) -> Iterator[_Group]:
        """Tag every event inside with ``group_id``; a group that raises is
        recorded (``group_error`` + its ``groups.jsonl`` line) and re-raised."""
        handle = _Group(self, str(group_id), info)
        previous, self._group_id = self._group_id, handle.group_id
        self.event("group_start", **info)
        details = " · ".join(f"{k} {_as_text(v)}" for k, v in info.items())
        title = f" group {handle.group_id}{' · ' + details if details else ''} · started {_now()} "
        self._llm_banner = "\n" + "█" * 4 + title + "█" * max(4, _RULE_WIDTH - len(title) - 4) + "\n\n"
        try:
            yield handle
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "exception"
            self.event(
                "group_error",
                status=status,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            self._finish_group(handle, {"status": status, "error": f"{type(exc).__name__}: {exc}"}, {})
            handle.finished = True
            raise
        finally:
            if not handle.finished:
                self._finish_group(handle, {"status": "unknown"}, {})
            self._group_id = previous
            self._llm_banner = None

    def _finish_group(self, handle: _Group, summary: dict, extra: dict) -> None:
        elapsed = round(time.perf_counter() - handle.started, 2)
        if summary.get("status") != "exception" and summary.get("status") != "interrupted":
            self.event("group_end", elapsed_s=elapsed, **summary, **extra)
        if self._closed or self._disabled:
            return
        line = {"ts": _now(), "group": handle.group_id, **handle.info,
                "elapsed_s": elapsed, **summary, **extra}
        try:
            with self._lock:
                self._groups.write(_dumps(line) + "\n")
                self._groups_by_status[str(summary.get("status"))] += 1
        except Exception as exc:  # noqa: BLE001
            self._disable(exc)

    # -- lifecycle ------------------------------------------------------

    def close(self, status: str = "finished", error: Optional[str] = None) -> None:
        if self._closed:
            return
        self._run.update(
            status=status,
            finished_at=_now(),
            events=dict(self._events_by_kind),
            groups=dict(self._groups_by_status),
        )
        if error:
            self._run["error"] = error
        self._write_run()
        self._closed = True
        for handle in (self._events, self._groups, self.console):
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass

    def write_console(self, text: str) -> None:
        if self._closed or not text:
            return
        try:
            with self._console_lock:
                self.console.write(_ANSI.sub("", text))
        except Exception:  # noqa: BLE001 — the terminal still has it
            pass

    def _disable(self, exc: Exception) -> None:
        if self._disabled:
            return
        self._disabled = True
        logger.warning("Generation trace disabled after a write error: %s", exc)

    def _write_run(self) -> None:
        try:
            tmp = self.dir / "run.json.tmp"
            tmp.write_text(json.dumps(self._run, indent=2, ensure_ascii=False, default=_jsonable))
            tmp.replace(self.dir / "run.json")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write the trace's run.json: %s", exc)

    def _point_latest(self) -> None:
        latest = self.root / "latest"
        try:
            if latest.is_symlink() or not latest.exists():
                if latest.is_symlink():
                    latest.unlink()
                latest.symlink_to(self.run_id, target_is_directory=True)
        except OSError:
            pass


def _git_state() -> Optional[dict]:
    """Commit and uncommitted-file count of the code that ran (None outside
    a git checkout)."""
    repo = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if commit.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        return {
            "commit": commit.stdout.strip(),
            "uncommitted_files": len([line for line in status.stdout.splitlines() if line.strip()]),
        }
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------
# Console and logging capture
# ----------------------------------------------------------------------

class _Tee:
    """A stream that also copies what it writes into the trace's console.log."""

    def __init__(self, stream: Any, trace: GenerationTrace):
        self._stream = stream
        self._trace = trace

    def write(self, text: str) -> int:
        written = self._stream.write(text)
        self._trace.write_console(text)
        return written

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _TraceLogHandler(logging.Handler):
    """Python logging records as ``log`` events: WARNING and up from any
    logger, INFO and up from OrQA's own modules."""

    def __init__(self, trace: GenerationTrace):
        super().__init__(logging.INFO)
        self.trace = trace

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.WARNING and not record.name.startswith(_OWN_LOGGERS):
            return
        if record.name == __name__:
            return  # the trace's own complaints would loop back into it
        try:
            payload = {"level": record.levelname, "logger": record.name, "message": record.getMessage()}
            if record.exc_info:
                payload["exception"] = "".join(traceback.format_exception(*record.exc_info))
            self.trace.event("log", **payload)
        except Exception:  # noqa: BLE001
            self.handleError(record)


class _Capture:
    """Install/uninstall the console tee and the logging handler."""

    def __init__(self, trace: GenerationTrace):
        self.trace = trace
        self._streams: tuple = ()
        self._handlers: list[logging.Handler] = []
        self._levels: list[tuple[logging.Logger, int]] = []

    def install(self) -> "_Capture":
        self._streams = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(sys.stdout, self.trace)
        sys.stderr = _Tee(sys.stderr, self.trace)

        root = logging.getLogger()
        if not root.handlers:
            # Python prints WARNING+ through logging.lastResort only while no
            # handler exists anywhere; ours would silence it, so stand in.
            console = logging.StreamHandler(sys.stderr)
            console.setLevel(logging.WARNING)
            root.addHandler(console)
            self._handlers.append(console)
        handler = _TraceLogHandler(self.trace)
        root.addHandler(handler)
        self._handlers.append(handler)

        # Let OrQA's INFO records reach the handler — only when no console
        # handler would start printing them too.
        if all(h.level >= logging.WARNING for h in root.handlers if h is not handler):
            for name in _OWN_LOGGERS:
                own = logging.getLogger(name)
                if own.getEffectiveLevel() > logging.INFO:
                    self._levels.append((own, own.level))
                    own.setLevel(logging.INFO)
        return self

    def uninstall(self) -> None:
        for own, level in self._levels:
            own.setLevel(level)
        root = logging.getLogger()
        for handler in self._handlers:
            root.removeHandler(handler)
        if self._streams:
            sys.stdout, sys.stderr = self._streams


# ----------------------------------------------------------------------
# Module-level API (no-ops while no trace is open)
# ----------------------------------------------------------------------

def active_trace() -> Optional[GenerationTrace]:
    return _ACTIVE


def trace_event(kind: str, /, **payload: Any) -> None:
    trace = _ACTIVE
    if trace is not None:
        trace.event(kind, **payload)


@contextmanager
def trace_group(group_id: str, /, **info: Any) -> Iterator[_Group]:
    """See :meth:`GenerationTrace.group`; without a trace, a handle whose
    ``finish`` does nothing."""
    trace = _ACTIVE
    if trace is None:
        yield _Group(None, str(group_id), info)
        return
    with trace.group(group_id, **info) as handle:
        yield handle


def trace_llm_call(record: dict, schema: Optional[str] = None) -> None:
    """One structured-call attempt (``LLMClientStructured._observe``): its
    outcome, tokens and timing always; the raw answer and the prompt only
    when the attempt failed, since that is when they are needed."""
    trace = _ACTIVE
    if trace is None:
        return
    try:
        outcome = str(record.get("outcome") or "")
        ok = outcome in _LLM_OK or outcome.startswith("repaired:")
        payload = {k: record[k] for k in _LLM_FIELDS if k in record}
        payload.update(ok=ok, schema=schema)
        if record.get("reasoning"):
            payload["reasoning_chars"] = len(record["reasoning"])
        if not ok and outcome not in ("error", "repair_error", "skipped_too_long", ""):
            payload["raw"] = _clip(record.get("raw"), _RAW_CAP)
            payload["prompt"] = _clip(record.get("prompt"), _PROMPT_CAP)
        # The whole exchange goes to llm/<group>.txt (trace.llm_io) — only
        # for a request that was actually sent.
        in_full = (record.get("prompt") is not None
                   and (trace.llm_io == "all" or (trace.llm_io == "failed" and not ok)))
        if in_full:
            payload["io"] = trace.llm_file()
        seq = trace.event("llm_call", **payload)
        if in_full:
            trace.write_llm_call(seq, record, schema, ok)
    except Exception:  # noqa: BLE001 — observers never break a call
        pass


@contextmanager
def open_trace(root: Path, run_info: Optional[dict] = None,
               llm_io: str = "all") -> Iterator[GenerationTrace]:
    """Open a run folder under ``root`` and make it the active trace: the
    console is mirrored into it and logging records are captured until the
    block exits. ``llm_io`` picks the LLM calls written in full to
    ``llm/<group>.txt``: ``all``, ``failed`` (the attempts that produced no
    usable answer) or ``off``."""
    global _ACTIVE
    trace = GenerationTrace(root, run_info, llm_io=llm_io)
    previous, _ACTIVE = _ACTIVE, trace
    capture = _Capture(trace).install()
    print(f"Generation trace: {trace.dir}")
    status, error = "finished", None
    try:
        yield trace
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        trace.event("run_error", error=error, traceback=traceback.format_exc())
        raise
    finally:
        capture.uninstall()
        _ACTIVE = previous
        trace.close(status, error)
