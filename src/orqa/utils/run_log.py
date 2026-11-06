"""Run log: one rotating file per target that keeps what every run did.

Each ``src/main.py`` run appends to ``<write_path>/<logging.dir>/<logging.file>``
(``data/bologna/log/orqa.log`` for bologna), configured by the top-level
``logging`` block of the workflow yaml (``conf.config.RunLogConfig``):

* everything the run prints to the terminal — step banners, the
  PipelineLogger's progress, warnings, a progress bar's final state —
  colour codes stripped, one timestamped line each (``stdout`` / ``stderr``);
* Python logging records: OrQA's own (``orqa``, ``solr``, ``conf``,
  ``cleaning``) from ``logging.level`` up — INFO by default, which the
  terminal does not show — and every library's from WARNING up.

The file rotates at ``logging.max_size``: ``orqa.log`` is always the live
file and the previous ones are ``orqa.log.1`` (newer) and ``orqa.log.2``
(older) — ``logging.files`` files in all. Once they are all full, the oldest
is overwritten, so the log never takes more than ``files x max_size``. Each
run opens with a ``run start`` line and closes with a ``run end`` line saying
how it ended (an unhandled exception's traceback is written before it), so
the runs sharing a file can be told apart.

The terminal output does not change: the run log only copies it.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import shlex
import sys
import threading
import time
import traceback
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

# Loggers whose INFO records the run log keeps (other loggers: WARNING up).
OWN_LOGGERS = ("orqa", "solr", "conf", "cleaning")

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S%z"
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# An unfinished console line longer than this is written as it stands.
_MAX_PENDING = 64 * 1024
_STREAM_EMIT = logging.StreamHandler.emit.__code__


def _is_own(name: str) -> bool:
    return any(name == own or name.startswith(own + ".") for own in OWN_LOGGERS)


def _reaches_root(name: str) -> bool:
    """Whether a record of logger ``name`` propagates up to the root logger."""
    logger = logging.getLogger(name)
    while logger.propagate and logger.parent is not None:
        logger = logger.parent
    return logger is logging.root


def _handled_below_root(name: str) -> bool:
    """Whether a handler between logger ``name`` and the root takes its records."""
    logger = logging.getLogger(name)
    while logger is not None and logger is not logging.root:
        if logger.handlers:
            return True
        if not logger.propagate:
            return False
        logger = logger.parent
    return False


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _ANSI.sub("", super().format(record))


class RunLogHandler(logging.handlers.RotatingFileHandler):
    """The run log's file: ``files`` files of at most ``max_bytes``.

    Only the process that opened it rotates it: a forked worker (the
    validator's sandbox) writes to the same file but never renames it under
    the parent. With ``files == 1`` the file starts over when full (a plain
    RotatingFileHandler with no backups grows forever). A write error (disk
    full) disables the log with one warning on the terminal instead of a
    traceback per line.
    """

    def __init__(self, path: Path, max_bytes: int, files: int, own_level: int):
        super().__init__(path, maxBytes=max_bytes, backupCount=files - 1, encoding="utf-8")
        self._owner = os.getpid()
        self.own_level = own_level
        self.disabled = False
        self.setLevel(min(own_level, logging.WARNING))
        self.setFormatter(_Formatter(_FORMAT, _DATEFMT))

    def wants(self, record: logging.LogRecord) -> bool:
        if getattr(record, "orqa_run_log", False):
            return True  # a console line or a run start/end line
        if _is_own(record.name):
            return record.levelno >= self.own_level
        return record.levelno >= logging.WARNING

    def filter(self, record):
        return not self.disabled and self.wants(record) and super().filter(record)

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        return os.getpid() == self._owner and super().shouldRollover(record)

    def doRollover(self) -> None:
        if self.backupCount > 0:
            super().doRollover()
            return
        if self.stream:
            self.stream.close()
        self.stream, self.mode = None, "w"
        try:
            self.stream = self._open()
        finally:
            self.mode = "a"

    def handleError(self, record: logging.LogRecord) -> None:
        if self.disabled:
            return
        self.disabled = True
        exc = sys.exc_info()[1]
        stream = getattr(sys, "__stderr__", None) or sys.stderr
        try:
            stream.write(f"Run log disabled: cannot write {self.baseFilename}: {exc}\n")
        except Exception:  # noqa: BLE001 — nowhere left to report it
            pass


class _LastResort(logging.StreamHandler):
    """Prints WARNING+ records on the terminal exactly as ``logging.lastResort``
    does — the bare message on stderr — and only when it would have: no
    handler of their own below the root. Python uses its last resort only
    while a record finds no handler at all, which the run log's handler on the
    root would otherwise end."""

    def __init__(self, stream):
        super().__init__(stream)
        self.setLevel(logging.WARNING)

    def filter(self, record):
        if _handled_below_root(record.name):
            return False
        for handler in logging.root.handlers:
            if handler is not self and type(handler) is logging.StreamHandler:
                return False  # someone else prints the root's records
        return super().filter(record)


_LIVE_COPIES: "weakref.WeakSet[_ConsoleCopy]" = weakref.WeakSet()


def _after_fork_in_child() -> None:
    # A forked child starts with fresh locks and none of the parent's
    # unfinished lines (it would write them a second time).
    for copy in list(_LIVE_COPIES):
        copy._lock = threading.Lock()
        copy._pending.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


class _ConsoleCopy:
    """``sys.stdout`` / ``sys.stderr`` stand-in: writes through to the real
    stream unchanged and copies each finished line into the run log.

    Lines are assembled per thread, so concurrent prints keep their own lines;
    a carriage-return redraw (a progress bar) keeps only its latest state.
    What a logging StreamHandler prints is not copied when the record reaches
    the run log's handler anyway (it is written there with its level and
    logger)."""

    def __init__(self, stream, name: str, log: "RunLog"):
        self._stream = stream
        self._name = name
        self._log = log
        self._pending: dict[int, str] = {}
        self._lock = threading.Lock()
        self._busy = threading.local()
        _LIVE_COPIES.add(self)

    def write(self, text):
        written = self._stream.write(text)
        if text and isinstance(text, str) and not getattr(self._busy, "on", False):
            self._busy.on = True
            try:
                if not self._from_logging_handler():
                    for line in self._lines(text):
                        self._log.console(self._name, line)
            except Exception:  # noqa: BLE001 — the terminal still has it
                pass
            finally:
                self._busy.on = False
        return written

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):
        if name == "_stream":  # not set yet (copy/unpickle)
            raise AttributeError(name)
        return getattr(self._stream, name)

    def _lines(self, text: str) -> list[str]:
        key = threading.get_ident()
        with self._lock:
            pending = (self._pending.pop(key, "") + text).replace("\r\n", "\n")
            *done, rest = pending.split("\n")
            rest = _latest_redraw(rest)
            if len(rest) > _MAX_PENDING:
                done.append(rest)
                rest = ""
            if rest:
                self._pending[key] = rest
        return [_visible(line).rstrip() for line in done]

    def drain(self) -> None:
        """Write the unfinished lines (end of the run)."""
        with self._lock:
            rest = [_visible(text).rstrip() for text in self._pending.values()]
            self._pending.clear()
        for line in rest:
            if line:
                self._log.console(self._name, line)

    def _from_logging_handler(self) -> bool:
        frame = sys._getframe(2)  # 0 this, 1 write, 2 write's caller
        for _ in range(4):  # the caller, or a wrapping stream's write above it
            if frame is None:
                return False
            if frame.f_code is _STREAM_EMIT:
                record = frame.f_locals.get("record")
                return isinstance(record, logging.LogRecord) and self._log.takes(record)
            frame = frame.f_back
        return False


def _visible(line: str) -> str:
    """What a carriage-return redraw leaves on screen: the last non-empty part."""
    if "\r" not in line:
        return line
    for part in reversed(line.split("\r")):
        if part:
            return part
    return ""


def _latest_redraw(rest: str) -> str:
    """An unfinished line keeps only its latest redraw (and a trailing \\r)."""
    cut = rest.rfind("\r", 0, len(rest) - 1)
    return rest[cut + 1:] if cut >= 0 else rest


class RunLog:
    """The installed run log: the file handler, the last-resort stand-in,
    the console copies, and what to restore when the run ends."""

    def __init__(self, path: Path, max_bytes: int, files: int, level: str = "INFO"):
        self.path = Path(path)
        self.own_level = logging.getLevelName(level.upper())
        if not isinstance(self.own_level, int):
            raise ValueError(f"unknown logging level {level!r}")
        self.max_bytes = max_bytes
        self.files = files
        self.handler: Optional[RunLogHandler] = None
        self._added: list[logging.Handler] = []
        self._levels: list[tuple[logging.Logger, int]] = []
        self._streams: tuple = ()
        self._copies: list[_ConsoleCopy] = []

    @classmethod
    def from_config(cls, config) -> "RunLog":
        return cls(config.path, config.max_bytes, config.files, config.level)

    # -- records -------------------------------------------------------------

    def takes(self, record: logging.LogRecord) -> bool:
        """Whether the file handler writes ``record`` when it is logged."""
        handler = self.handler
        return (handler is not None and not handler.disabled
                and record.levelno >= handler.level
                and _reaches_root(record.name) and handler.wants(record))

    def console(self, stream_name: str, line: str) -> None:
        self._write(logging.LogRecord(stream_name, logging.INFO, "", 0, line, None, None))

    def note(self, message: str, level: int = logging.INFO) -> None:
        """A line of the run log's own (``run start`` / ``run end``)."""
        self._write(logging.LogRecord("run", level, "", 0, message, None, None))

    def _write(self, record: logging.LogRecord) -> None:
        record.orqa_run_log = True
        if self.handler is not None:
            self.handler.handle(record)

    # -- install / uninstall -------------------------------------------------

    def install(self) -> "RunLog":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handler = RunLogHandler(self.path, self.max_bytes, self.files, self.own_level)
        root = logging.getLogger()
        if not root.handlers:
            # Bound to the real stderr, so what it prints is not copied twice.
            self._add(root, _LastResort(sys.stderr))
        # Let OrQA's records below WARNING be created — only when no console
        # handler on the root would start printing them too.
        if all(h.level >= logging.WARNING or isinstance(h, logging.FileHandler)
               for h in root.handlers):
            for name in OWN_LOGGERS:
                own = logging.getLogger(name)
                if own.getEffectiveLevel() > self.own_level:
                    self._levels.append((own, own.level))
                    own.setLevel(self.own_level)
        self._add(root, self.handler)

        self._streams = (sys.stdout, sys.stderr)
        self._copies = [_ConsoleCopy(sys.stdout, "stdout", self),
                        _ConsoleCopy(sys.stderr, "stderr", self)]
        sys.stdout, sys.stderr = self._copies
        return self

    def _add(self, root: logging.Logger, handler: logging.Handler) -> None:
        root.addHandler(handler)
        self._added.append(handler)

    def uninstall(self) -> None:
        for copy in self._copies:
            try:
                copy.drain()
            except Exception:  # noqa: BLE001
                pass
        if self._streams:
            sys.stdout, sys.stderr = self._streams
            self._streams = ()
        for own, level in self._levels:
            own.setLevel(level)
        self._levels.clear()
        root = logging.getLogger()
        for handler in self._added:
            root.removeHandler(handler)
        self._added.clear()
        if self.handler is not None:
            self.handler.close()
            self.handler = None


def _elapsed(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


@contextmanager
def run_log(config, argv: Optional[Sequence[str]] = None) -> Iterator[Optional[RunLog]]:
    """Keep the run log for the body of the ``with``: a ``run start`` line,
    everything printed and logged, then ``run end`` with how the run ended.

    ``config`` is a ``conf.config.RunLogConfig``; when it is missing or
    disabled, or the file cannot be opened (a read-only mount), the run goes
    on without a log (the last case says so on stderr).
    """
    if config is None or not getattr(config, "enabled", False) or getattr(config, "path", None) is None:
        yield None
        return
    log = RunLog.from_config(config)
    try:
        log.install()
    except OSError as exc:
        log.uninstall()
        print(f"Run log disabled: cannot open {config.path}: {exc}", file=sys.stderr)
        yield None
        return

    command = shlex.join(["src/main.py", *(sys.argv[1:] if argv is None else argv)])
    log.note(f"run start: {command} (pid {os.getpid()})")
    started = time.monotonic()
    outcome = "ok"
    try:
        yield log
    except SystemExit as exc:
        if exc.code not in (None, 0):
            outcome = f"exit status {exc.code}"
        raise
    except KeyboardInterrupt:
        outcome = "interrupted"
        raise
    except BaseException as exc:
        log.note("run failed:\n" + "".join(traceback.format_exception(exc)).rstrip(), logging.ERROR)
        outcome = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        log.note(f"run end: {outcome} after {_elapsed(time.monotonic() - started)}")
        log.uninstall()
