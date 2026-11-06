"""Run a LOCAL Elasticsearch for a workflow run — no Docker, no service to start.

Elasticsearch is a JVM server; it cannot be embedded in the Python process.
What can be done is what ``src/solr/solr.py`` does for Solr: let OrQa start the
server itself when a step needs it, and stop it when the run ends. With
``tasks.mcp_search.elasticsearch_managed.enabled: true``, ``load_index`` (so
``generate-statements``, ``solve-benchmark`` and the retrievability report)
calls :func:`ensure_running` before it connects.

Everything lives next to the portal's other index files, under
``<data_path>/index/elasticsearch/``::

    elasticsearch-<version>/   the unpacked distribution (bundles its own JDK)
    data/                      path.data — the index itself
    logs/                      path.logs, and orqa-elasticsearch.out (stdout)

The distribution is downloaded once, on first use (``auto_install``), or with
``python -m orqa.benchmark.es_local install``. The download is unpacked WHILE it
streams — the ~640 MB tarball is never written to disk — and checked against
Elastic's published SHA-512 before it is accepted.

The server is started as a single node on loopback with security off (the
client connects without credentials), and with Elasticsearch's DISK WATERMARKS
OFF: on a nearly full disk they would refuse to allocate the index's shard, or
flip it read-only, which is not what a local development server should do.

If this run started the server it stops it at exit (``keep_running: true``
leaves it up). A server that was already running is reused and never stopped.
Two runs at once should therefore set ``keep_running: true``, or start the
server once and use the plain ``elasticsearch_url``.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_VERSION = "8.17.0"
_DOWNLOAD_ROOT = "https://artifacts.elastic.co/downloads/elasticsearch"
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

# Free space required before installing: the unpacked distribution is ~1.5 GB
# and the index a few hundred MB more. Refusing early beats filling the disk.
MIN_FREE_BYTES = 3 * 1024**3


def announce(message: str) -> None:
    """Say what the index server is doing, where a person running the pipeline
    can see it. ``main.py`` leaves logging unconfigured, so a ``logger.info``
    never reaches the terminal; a print does."""
    try:
        print(f"[elasticsearch] {message}", flush=True)
    except Exception:  # stdout may be gone while the interpreter shuts down
        pass


# ---------------------------------------------------------------------------
# install


def download_url(version: str = DEFAULT_VERSION, system: Optional[str] = None,
                 machine: Optional[str] = None) -> str:
    """The distribution tarball for this platform (Linux and macOS)."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(machine)
    if system not in ("linux", "darwin") or arch is None:
        raise RuntimeError(
            f"No Elasticsearch tarball for {system}/{machine}. Unpack a distribution "
            "yourself and set tasks.mcp_search.elasticsearch_managed.home."
        )
    return f"{_DOWNLOAD_ROOT}/elasticsearch-{version}-{system}-{arch}.tar.gz"


class _HashingReader:
    """A file-like object that feeds everything it reads to a digest, so the
    checksum covers the whole download even though it is never saved."""

    def __init__(self, raw: Any, digest: Any, total: Optional[int], log: Callable[[str], None]):
        self._raw, self._digest, self._total, self._log = raw, digest, total, log
        self._seen = 0
        self._next_report = 10

    def read(self, size: int = -1) -> bytes:
        chunk = self._raw.read(size)
        self._digest.update(chunk)
        self._seen += len(chunk)
        if self._total:
            percent = 100 * self._seen // self._total
            if percent >= self._next_report:
                self._log(f"  {percent}% of {self._total // 2**20} MB")
                self._next_report = percent // 10 * 10 + 10
        return chunk


def install(
    dest: Path,
    version: str = DEFAULT_VERSION,
    url: Optional[str] = None,
    log: Callable[[str], None] = print,
    min_free_bytes: int = MIN_FREE_BYTES,
) -> Path:
    """Download and unpack Elasticsearch into ``dest/elasticsearch-<version>``;
    returns that directory. Idempotent: an existing install is returned as is.

    The archive is unpacked as it streams (no tarball on disk) and only kept if
    its SHA-512 matches the ``.sha512`` file published next to it.
    """
    dest = Path(dest)
    home = dest / f"elasticsearch-{version}"
    if (home / "bin" / "elasticsearch").exists():
        return home

    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if free < min_free_bytes:
        raise RuntimeError(
            f"Only {free / 2**30:.1f} GB free at {dest}; unpacking Elasticsearch needs about "
            f"{min_free_bytes / 2**30:.0f} GB. Free some space, or set "
            "tasks.mcp_search.elasticsearch_managed.home to an install elsewhere."
        )

    url = url or download_url(version)
    log(f"Installing Elasticsearch {version} into {home} (one time)…")
    with urllib.request.urlopen(url + ".sha512", timeout=60) as response:
        expected = response.read().decode().split()[0].strip().lower()

    staging = dest / f".installing-{version}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0) or None
            digest = hashlib.sha512()
            reader = _HashingReader(response, digest, total, log)
            with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                for member in archive:
                    archive.extract(member, staging, filter="tar")
            while reader.read(1 << 20):  # the digest must cover the trailing bytes too
                pass
        if digest.hexdigest() != expected:
            raise RuntimeError(
                f"Checksum mismatch for {url}: got {digest.hexdigest()[:16]}…, "
                f"expected {expected[:16]}…. Nothing was installed."
            )
        unpacked = [p for p in staging.iterdir() if p.is_dir()]
        if len(unpacked) != 1 or not (unpacked[0] / "bin" / "elasticsearch").exists():
            raise RuntimeError(f"Unexpected archive layout in {url}; nothing was installed.")
        os.replace(unpacked[0], home)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log(f"Installed: {home}")
    return home


# ---------------------------------------------------------------------------
# run


class ManagedElasticsearch:
    """A local Elasticsearch process OrQa starts, waits for, and stops."""

    def __init__(
        self,
        home: Path,
        url: str,
        data_dir: Path,
        heap: str = "2g",
        startup_timeout: float = 180.0,
        keep_running: bool = False,
        ping: Optional[Callable[[], bool]] = None,
        health: Optional[Callable[[], bool]] = None,
        popen: Callable[..., Any] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.home = Path(home)
        self.url = url
        self.data_dir = Path(data_dir)
        self.heap = heap
        self.startup_timeout = startup_timeout
        self.keep_running = keep_running
        self._ping = ping or self._default_ping
        # An injected ping stands for "the server is fully up" unless a health
        # probe is injected too.
        self._health = health or (self._default_health if ping is None else (lambda: True))
        self._popen = popen
        self._sleep = sleep
        self._clock = clock
        self.process: Any = None

    @property
    def port(self) -> int:
        return urlparse(self.url).port or 9200

    @property
    def log_file(self) -> Path:
        return self.data_dir / "logs" / "orqa-elasticsearch.out"

    def _default_ping(self) -> bool:
        """Is ANYTHING answering HTTP at the URL? A plain GET, so polling a server
        that is still starting is silent (the Elasticsearch client logs a warning
        and a traceback per failed attempt). Any response — even a 401 from a
        secured server — counts: something is there, so we must not start another."""
        import urllib.error

        try:
            with urllib.request.urlopen(self.url, timeout=3):
                return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            return False

    def _default_health(self) -> bool:
        """Can the cluster SERVE searches? Answering HTTP is not enough: after a
        restart the server responds while the shards of an existing index are
        still being recovered from disk, and a search in that window fails with
        a 503. Ask the server to wait (up to 5 s per call) for status yellow —
        for a single node with no replicas, the primary shard being started."""
        import json

        try:
            url = f"{self.url.rstrip('/')}/_cluster/health?wait_for_status=yellow&timeout=5s"
            with urllib.request.urlopen(url, timeout=15) as response:
                body = json.loads(response.read().decode())
            return not body.get("timed_out", False) and body.get("status") in ("yellow", "green")
        except Exception:  # 408 on timeout, connection errors, a secured server...
            return False

    def running(self) -> bool:
        try:
            return bool(self._ping())
        except Exception:
            return False

    def command(self) -> list[str]:
        settings = {
            "discovery.type": "single-node",
            "network.host": "127.0.0.1",
            "http.port": str(self.port),
            "xpack.security.enabled": "false",
            # ML needs a native controller that is pointless here and can fail to
            # start on some hosts; the GeoIP downloader phones home at startup.
            "xpack.ml.enabled": "false",
            "ingest.geoip.downloader.enabled": "false",
            # A local dev server on a nearly full disk must still allocate its
            # shard and stay writable (see the module docstring).
            "cluster.routing.allocation.disk.threshold_enabled": "false",
            "path.data": str(self.data_dir / "data"),
            "path.logs": str(self.data_dir / "logs"),
        }
        command = [str(self.home / "bin" / "elasticsearch")]
        for key, value in settings.items():
            command += ["-E", f"{key}={value}"]
        return command

    def environment(self) -> dict[str, str]:
        return {**os.environ, "ES_JAVA_OPTS": f"-Xms{self.heap} -Xmx{self.heap}"}

    def _log_tail(self, lines: int = 25) -> str:
        try:
            return "\n".join(self.log_file.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return "(no log)"

    def start(self) -> None:
        launcher = self.home / "bin" / "elasticsearch"
        if not launcher.exists():
            raise FileNotFoundError(
                f"No Elasticsearch at {self.home} (missing bin/elasticsearch). Install it with "
                "`python -m orqa.benchmark.es_local install`, or set "
                "tasks.mcp_search.elasticsearch_managed.home."
            )
        (self.data_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "logs").mkdir(parents=True, exist_ok=True)
        announce(
            f"starting a local server from {self.home} (heap {self.heap}, data in "
            f"{self.data_dir / 'data'}) — this can take up to 30 s…"
        )
        with open(self.log_file, "ab") as out:
            self.process = self._popen(
                self.command(), env=self.environment(), stdout=out, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )

    def wait_until_ready(self) -> None:
        deadline = self._clock() + self.startup_timeout
        announced = False
        while self._clock() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"Elasticsearch exited during startup (code {self.process.returncode}). "
                    f"Last lines of {self.log_file}:\n{self._log_tail()}"
                )
            if self.running():
                if self._health():
                    return
                if not announced:
                    announce("the server is answering; waiting for its index to recover from disk…")
                    announced = True
            self._sleep(1.0)
        self.stop()
        raise RuntimeError(
            f"Elasticsearch did not become ready at {self.url} within {self.startup_timeout:.0f}s "
            f"(it must answer AND report cluster health yellow or green). "
            f"Last lines of {self.log_file}:\n{self._log_tail()}"
        )

    def stop(self) -> None:
        """Stop the server this object started (a no-op for one it did not)."""
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        announce(f"stopping the server this run started (pid {process.pid})…")
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=60)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                pass
        announce("stopped.")

    def ensure_running(self) -> bool:
        """Make sure the server answers. Returns True when THIS call started it."""
        if self.running():
            if not self._health():
                announce("already running, but its index is still recovering — waiting…")
                self.wait_until_ready()
            announce(f"already running at {self.url} — reusing it (this run will not stop it).")
            return False
        began = self._clock()
        self.start()
        self.wait_until_ready()
        announce(
            f"started at {self.url} (pid {getattr(self.process, 'pid', '?')}, ready in "
            f"{self._clock() - began:.0f}s)"
            + ("; it stays up when this run ends (keep_running)." if self.keep_running
               else "; it will be stopped when this run ends.")
        )
        if not self.keep_running:
            atexit.register(self.stop)
        return True


# ---------------------------------------------------------------------------
# config glue

_started: dict[str, ManagedElasticsearch] = {}


def resolve_paths(cfg: Any) -> tuple[Path, Path, bool]:
    """``(home, data_dir, home_is_default)`` — everything under
    ``<data_path>/index/elasticsearch`` unless the config or ``$ES_HOME``
    says otherwise."""
    managed = cfg.mcp_search.elasticsearch_managed
    base = Path(cfg.mcp_search.index_filepath).parent / "elasticsearch"
    explicit = managed.home or os.environ.get("ES_HOME", "").strip()
    home = Path(explicit) if explicit else base / f"elasticsearch-{managed.version}"
    data_dir = Path(managed.data_dir) if managed.data_dir else base
    return home, data_dir, not explicit


def ensure_running(cfg: Any, es_url: str) -> Optional[ManagedElasticsearch]:
    """Start (and, if needed, first install) the local Elasticsearch a
    workflow's config asks for. A no-op when ``elasticsearch_managed`` is off.
    """
    managed = cfg.mcp_search.elasticsearch_managed
    if not managed.enabled:
        return None
    host = (urlparse(es_url).hostname or "").lower()
    if host not in _LOCAL_HOSTS:
        raise ValueError(
            f"tasks.mcp_search.elasticsearch_managed only manages a LOCAL Elasticsearch, but the "
            f"URL is {es_url!r}. Turn it off to use a remote cluster."
        )
    if es_url in _started and _started[es_url].running():
        return _started[es_url]

    home, data_dir, home_is_default = resolve_paths(cfg)
    server = ManagedElasticsearch(
        home, es_url, data_dir, heap=managed.heap,
        startup_timeout=managed.startup_timeout, keep_running=managed.keep_running,
    )
    if not server.running() and not (home / "bin" / "elasticsearch").exists():
        if not (managed.auto_install and home_is_default):
            raise FileNotFoundError(
                f"No Elasticsearch at {home}. Run `python -m orqa.benchmark.es_local install "
                f"--dest {home.parent}` or fix tasks.mcp_search.elasticsearch_managed.home."
            )
        install(home.parent, managed.version, log=announce)
    server.ensure_running()
    _started[es_url] = server
    return server


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Install the local Elasticsearch OrQa can manage.")
    sub = parser.add_subparsers(dest="command", required=True)
    inst = sub.add_parser("install", help="Download and unpack Elasticsearch (streamed, checksum-verified).")
    inst.add_argument("--dest", type=Path, required=True, help="Directory to unpack into, e.g. data/uk/index/elasticsearch")
    inst.add_argument("--version", default=DEFAULT_VERSION)
    args = parser.parse_args(argv)
    home = install(args.dest, args.version)
    print(f"Elasticsearch is at {home}")


if __name__ == "__main__":
    main(sys.argv[1:])
