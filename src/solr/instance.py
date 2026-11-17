"""OrQA's own Solr: the server that holds the portal cores.

OrQA builds the cores LakeGen is scored on (``solr.build``)
and serves them from this instance; ``solr.index`` queries them. It is a standalone
(user-managed) Solr started from the installed Solr as the current user, and
everything it writes lives under one folder:

    $ORQA_SOLR_HOME/
        solr.xml, solr.in.sh       written by :meth:`SolrInstance.prepare`
        configsets/portal/conf/    copy of conf/solr/portal, every core's config
        <core>/                    one folder per core (core.properties, data/)
        logs/, solr-<port>.pid

Settings (environment or ``.env``):
    ORQA_SOLR_HOME  the folder above; required.
    ORQA_SOLR_PORT  its port, default 8984.
    ORQA_SOLR_HEAP  JVM heap, default 2g.
    SOLR_PATH       the Solr installation, default /opt/solr.

LakeGen's shared Solr (port 8983, home /var/solr/data) is another server.
Every write first asks the server at :attr:`SolrInstance.url` for its Solr
home and refuses unless it is ``ORQA_SOLR_HOME``
(:meth:`SolrInstance.assert_own`), so no port or URL setting can make OrQA
write to that server.
"""

from __future__ import annotations

import filecmp
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import requests

CONFIGSET = "portal"
CONFIGSET_SOURCE = Path(__file__).resolve().parents[2] / "conf" / "solr" / CONFIGSET
DEFAULT_PORT = 8984
DEFAULT_HEAP = "2g"
DEFAULT_INSTALL = Path("/opt/solr")
# Documents per update request: a uk document with its 1024-d vector is about
# 15 KB of JSON, so a batch is a few MB.
POST_BATCH = 500

_CORE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")


class SolrInstanceError(RuntimeError):
    """OrQA's Solr is not set up, not running, or answered with an error."""


def default_url() -> str:
    """The URL of OrQA's Solr from ``ORQA_SOLR_PORT``, without needing
    ``ORQA_SOLR_HOME`` (the retrieval side only queries)."""
    port = int(os.environ.get("ORQA_SOLR_PORT") or DEFAULT_PORT)
    return f"http://localhost:{port}/solr"


def check_core_name(name: str) -> str:
    if not _CORE_NAME.match(name or ""):
        raise ValueError(f"invalid Solr core name {name!r}")
    return name


def build_core_name(core: str) -> str:
    """Where a build of ``core`` is made before :meth:`SolrInstance.publish`."""
    return f"{check_core_name(core)}_build"


class SolrInstance:
    """OrQA's Solr server: its files, its process and its cores.

    Args:
        home: The Solr home (``ORQA_SOLR_HOME``).
        port: The HTTP port (``ORQA_SOLR_PORT``).
        install: The Solr installation whose ``bin/solr`` runs it.
        heap: The JVM heap (``bin/solr -m``).
        timeout: Seconds per admin or update request.
    """

    def __init__(
        self,
        home: Path,
        port: int = DEFAULT_PORT,
        install: Path = DEFAULT_INSTALL,
        heap: str = DEFAULT_HEAP,
        timeout: float = 300.0,
    ) -> None:
        self.home = Path(home).expanduser().resolve()
        self.port = int(port)
        self.install = Path(install)
        self.heap = heap
        self.timeout = timeout
        self._session = requests.Session()

    @classmethod
    def from_env(cls) -> "SolrInstance":
        home = os.environ.get("ORQA_SOLR_HOME", "").strip()
        if not home:
            raise SolrInstanceError(
                "ORQA_SOLR_HOME is not set: point it (e.g. in .env) at the folder "
                "OrQA's Solr keeps its cores in, on a disk with room for the indexes."
            )
        return cls(
            Path(home),
            port=int(os.environ.get("ORQA_SOLR_PORT") or DEFAULT_PORT),
            install=Path(os.environ.get("SOLR_PATH") or DEFAULT_INSTALL),
            heap=os.environ.get("ORQA_SOLR_HEAP") or DEFAULT_HEAP,
        )

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/solr"

    @property
    def configset_dir(self) -> Path:
        return self.home / "configsets" / CONFIGSET / "conf"

    @property
    def _include(self) -> Path:
        return self.home / "solr.in.sh"

    # ------------------------------------------------------------- files

    def prepare(self) -> None:
        """Create the home: ``solr.xml``, the include file ``bin/solr``
        reads instead of the system service's, and the configset (copied
        only when missing: a changed one is :meth:`configset_drift`)."""
        if not CONFIGSET_SOURCE.is_dir():
            raise SolrInstanceError(f"the configset {CONFIGSET_SOURCE} is missing")
        (self.home / "logs").mkdir(parents=True, exist_ok=True)
        solr_xml = self.home / "solr.xml"
        if not solr_xml.exists():
            shutil.copy2(self.install / "server" / "solr" / "solr.xml", solr_xml)
        if not self.configset_dir.exists():
            shutil.copytree(CONFIGSET_SOURCE, self.configset_dir)
        # bin/solr sources the first readable solr.in.sh it finds (its own
        # folder, ~/.solr.in.sh, /etc/default, /var/solr, ...) unless
        # SOLR_INCLUDE names one; this one keeps the pid file and the logs
        # in the home and binds to localhost only.
        self._include.write_text(
            "# Written by OrQA (src/solr/instance.py); rewritten on every start.\n"
            f'SOLR_PID_DIR="{self.home}"\n'
            f'SOLR_LOGS_DIR="{self.home / "logs"}"\n'
            f'SOLR_HEAP="{self.heap}"\n'
            "SOLR_JETTY_HOST=127.0.0.1\n"
        )

    def configset_drift(self) -> list[str]:
        """Files of ``conf/solr/portal`` that differ from the home's copy,
        or exist on one side only."""
        if not self.configset_dir.exists():
            return []
        drift: list[str] = []

        def walk(cmp: filecmp.dircmp, prefix: str) -> None:
            for name in (*cmp.left_only, *cmp.right_only, *cmp.diff_files, *cmp.funny_files):
                drift.append(prefix + name)
            for name, sub in cmp.subdirs.items():
                walk(sub, f"{prefix}{name}/")

        walk(filecmp.dircmp(CONFIGSET_SOURCE, self.configset_dir), "")
        return sorted(drift)

    def sync_configset(self) -> list[str]:
        """Copy ``conf/solr/portal`` over the home's configset and reload
        every core so it takes effect. Returns the files that changed."""
        drift = self.configset_drift()
        if not drift:
            return []
        self.assert_own()
        shutil.rmtree(self.configset_dir)
        shutil.copytree(CONFIGSET_SOURCE, self.configset_dir)
        for name in self.cores():
            self._admin(action="RELOAD", core=name)
        return drift

    # ----------------------------------------------------------- process

    def _bin(self, *args: str) -> None:
        env = {**os.environ, "SOLR_INCLUDE": str(self._include)}
        # The service's settings must not leak in from the caller's shell.
        for name in ("SOLR_HOME", "SOLR_PORT", "SOLR_PID_DIR", "SOLR_LOGS_DIR", "SOLR_DATA_HOME"):
            env.pop(name, None)
        subprocess.run([str(self.install / "bin" / "solr"), *args], env=env, check=True)

    def start(self) -> bool:
        """Start the server unless it already runs; True when started."""
        if self.running():
            return False
        self.prepare()
        self._bin(
            "start", "--user-managed", "-p", str(self.port),
            "--solr-home", str(self.home), "-m", self.heap,
        )
        self.assert_own()
        self._wait_for_cores()
        return True

    def _wait_for_cores(self, timeout: float = 180.0) -> None:
        """Solr answers before it has loaded its cores, and STATUS lists
        only the loaded ones: wait until every core folder of the home is
        listed, so nobody mistakes a loading core for a missing one."""
        expected = {
            line.split("=", 1)[1].strip()
            for path in self.home.glob("*/core.properties")
            for line in path.read_text().splitlines()
            if line.startswith("name=")
        }
        deadline = time.monotonic() + timeout
        while True:
            missing = expected - set(self.cores())
            if not missing:
                return
            if time.monotonic() > deadline:
                raise SolrInstanceError(f"cores not loaded after {timeout:.0f} s: {sorted(missing)}")
            time.sleep(1)

    def stop(self) -> bool:
        """Stop the server; True when it was running."""
        if not self.running():
            return False
        self._bin("stop", "-p", str(self.port))
        return True

    def system_info(self) -> Optional[dict[str, Any]]:
        """``/admin/info/system`` of the server on this port; None when
        nothing answers."""
        try:
            response = self._session.get(
                f"{self.url}/admin/info/system", params={"wt": "json"}, timeout=5
            )
        except (requests.ConnectionError, requests.Timeout):
            return None
        if response.status_code != 200:
            return None
        return response.json()

    def running(self) -> bool:
        """Whether OrQA's Solr answers on this port. Another Solr on the
        port is an error, not "not running"."""
        info = self.system_info()
        if info is None:
            return False
        self._check_home(info)
        return True

    def assert_own(self) -> None:
        """Refuse unless the server at :attr:`url` is this instance (its
        Solr home is :attr:`home`) — called before every write."""
        info = self.system_info()
        if info is None:
            raise SolrInstanceError(
                f"OrQA's Solr is not running at {self.url}: start it with "
                "`python src/solr/cli.py start`."
            )
        self._check_home(info)

    def _check_home(self, info: dict[str, Any]) -> None:
        served = Path(str(info.get("solr_home") or "")).resolve()
        if served != self.home:
            raise SolrInstanceError(
                f"the Solr at {self.url} serves {served}, not ORQA_SOLR_HOME {self.home}: "
                "OrQA writes only to its own instance (is ORQA_SOLR_PORT another server's port?)."
            )

    # ------------------------------------------------------------- cores

    def _admin(self, **params: str) -> dict[str, Any]:
        response = self._session.get(
            f"{self.url}/admin/cores", params={**params, "wt": "json"}, timeout=self.timeout
        )
        if response.status_code != 200:
            raise SolrInstanceError(
                f"Solr core admin {params} failed: HTTP {response.status_code}: "
                f"{_solr_message(response)}"
            )
        return response.json()

    def cores(self) -> dict[str, dict[str, Any]]:
        """``{core: {"docs": int, "size": str, "instance_dir": str}}``."""
        status = self._admin(action="STATUS").get("status", {})
        return {
            name: {
                "docs": (info.get("index") or {}).get("numDocs"),
                "size": (info.get("index") or {}).get("size"),
                "instance_dir": info.get("instanceDir"),
            }
            for name, info in sorted(status.items())
        }

    def create_core(self, name: str) -> None:
        """A new, empty core on the portal configset. Its folder is the
        name, or the name and a timestamp when that folder is taken: a
        RENAME keeps a core's folder, so a published build still lives in
        ``<core>_build/``."""
        check_core_name(name)
        self.assert_own()
        if name in self.cores():
            raise SolrInstanceError(f"core {name!r} already exists")
        folder = name
        if (self.home / folder).exists():
            folder = f"{name}.{time.strftime('%Y%m%dT%H%M%S')}"
        self._admin(action="CREATE", name=name, instanceDir=folder, configSet=CONFIGSET)

    def unload_core(self, name: str) -> None:
        """Remove a core and its index files."""
        check_core_name(name)
        self.assert_own()
        self._admin(action="UNLOAD", core=name, deleteInstanceDir="true")

    def rename_core(self, name: str, new_name: str) -> None:
        self.assert_own()
        self._admin(action="RENAME", core=check_core_name(name), other=check_core_name(new_name))

    def publish(self, core: str) -> Optional[str]:
        """Serve ``<core>_build`` as ``core``. The core it replaces stays as
        ``<core>_previous`` (an older ``_previous`` is dropped). Returns the
        name the replaced core now has, or None when ``core`` did not exist.

        Two RENAMEs, no SWAP: in user-managed Solr 9.10 a RENAME right after
        a SWAP dropped the swapped-in core from the running server (its
        files were right; a restart brought it back). ``core`` is missing
        between the two renames, for milliseconds."""
        build, previous = build_core_name(core), f"{core}_previous"
        cores = self.cores()
        built = (cores.get(build) or {}).get("docs")
        if not built:
            raise SolrInstanceError(f"no built core {build!r} to publish (missing or empty)")
        replaced = None
        if core in cores:
            if previous in cores:
                self.unload_core(previous)
            self.rename_core(core, previous)
            replaced = previous
        self.rename_core(build, core)
        after = self.cores()
        if (after.get(core) or {}).get("docs") != built or build in after:
            raise SolrInstanceError(
                f"publishing {build!r} as {core!r} left the server in an unexpected state: "
                f"{ {n: after.get(n) for n in (core, build, previous)} }"
            )
        return replaced

    def add_documents(
        self, core: str, documents: Iterable[dict[str, Any]], batch_size: int = POST_BATCH
    ) -> int:
        """Post documents in batches, then commit once; the number posted."""
        check_core_name(core)
        self.assert_own()
        posted = 0
        for batch in _batches(documents, batch_size):
            self._update(core, batch)
            posted += len(batch)
        self._update(core, {"commit": {}})
        return posted

    def _update(self, core: str, body: Any) -> None:
        response = self._session.post(
            f"{self.url}/{core}/update", params={"wt": "json"}, json=body, timeout=self.timeout
        )
        if response.status_code != 200:
            raise SolrInstanceError(
                f"posting to core {core!r} failed: HTTP {response.status_code}: "
                f"{_solr_message(response)}"
            )


def _batches(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _solr_message(response: requests.Response) -> str:
    try:
        return str(response.json()["error"]["msg"])
    except Exception:
        return response.text[:300]
