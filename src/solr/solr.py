import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

# main.py (sibling of this file's parent) is the single source of truth for
# target -> data layout resolution (TargetSpec, write_path, flat_layout, ...).
# Importing it here — rather than hand-rolling a second DATADIR-based path
# per domain — is the only way this file's notion of "where is a domain's
# normalized metadata" cannot silently drift from what the actual workflow
# (main.py + conf/config.py's write_path split) resolves it to.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as _main


class Solr:
    def __init__(self, domain: str, use_sudo: bool = False):
        self.domain = domain
        self._use_sudo = use_sudo
        self._solr_url = os.environ.get(
            "SOLR_URL", "http://localhost:8983/solr"
        ).rstrip("/")
        self._solr_config_owner = os.environ.get("SOLR_CONFIG_OWNER", "solr")
        self._solr_config_group = os.environ.get("SOLR_CONFIG_GROUP", "solr")
        self._solr_path = self._load_optional_path("SOLR_PATH")
        self._solr_home = self._load_solr_home()
        self._solr_bin = self._load_solr_bin()
        self._collection_conf_path = self._resolve_collection_conf_path()

    def _resolve_collection_conf_path(self) -> Path:
        """Where this collection's managed-schema.xml/solrconfig.xml actually
        live.

        ``bin/solr create -c <name>`` (no ``-d``) gives every collection its
        OWN embedded copy at ``<solr_home>/<name>/conf`` — what this used to
        assume unconditionally. But a collection can instead be built on a
        named, possibly-shared CONFIGSET (``bin/solr create -c <name> -d
        <configset>``), which lives at ``<solr_home>/configsets/<configset>/
        conf`` instead and is merely referenced from the collection's
        ``core.properties`` (``configSet=<configset>``) — true for every
        collection on this project's shared Solr instance. Reading that
        reference back is the only way to find the real path for an existing
        collection; a not-yet-created one falls back to the embedded default,
        matching what ``create_cluster`` (no ``-d``) will actually produce.
        """
        core_properties = self._solr_home / self.domain / "core.properties"
        if core_properties.exists():
            for line in core_properties.read_text().splitlines():
                if line.startswith("configSet="):
                    configset = line.split("=", 1)[1].strip()
                    configset_path = self._solr_home / "configsets" / configset / "conf"
                    if configset_path.exists():
                        return configset_path
        return self._solr_home / self.domain / "conf"

    @staticmethod
    def _load_optional_path(env_name: str) -> Path | None:
        value = os.environ.get(env_name)
        if not value:
            return None
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Path from {env_name} does not exist: {path}")
        return path

    def _load_solr_home(self) -> Path:
        solr_home = self._load_optional_path("SOLR_HOME")
        if solr_home is not None:
            return solr_home
        if self._solr_path is not None:
            solr_home = self._solr_path / "server" / "solr"
            if not solr_home.exists():
                raise FileNotFoundError(
                    f"Default Solr home not found: {solr_home}. "
                    "Set SOLR_HOME to the active Solr home."
                )
            return solr_home
        raise OSError(
            "SOLR_HOME or SOLR_PATH is required. For a system service, "
            "SOLR_HOME is usually the directory containing collection folders."
        )

    def _load_solr_bin(self) -> str:
        solr_bin = os.environ.get("SOLR_BIN")
        if solr_bin:
            return solr_bin
        if self._solr_path is not None:
            return str(self._solr_path / "bin" / "solr")
        return "solr"

    def _exec(self, *args):
        subprocess.run([str(arg) for arg in args], check=True)

    def _copy_config_file(self, source: Path, destination: Path) -> None:
        if not source.exists():
            raise FileNotFoundError(f"Config file not found: {source}")

        if self._use_sudo:
            self._exec(
                "sudo",
                "install",
                "-o",
                self._solr_config_owner,
                "-g",
                self._solr_config_group,
                "-m",
                "0644",
                source,
                destination,
            )
            return

        try:
            shutil.copy2(source, destination)
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot write {destination}. Re-run with --sudo, or give your user "
                f"write access to {self._collection_conf_path}."
            ) from exc

    def _admin_request(self, path: str, params: dict[str, str]) -> str:
        query = urlencode({**params, "wt": "json"})
        url = f"{self._solr_url}/{path}?{query}"
        try:
            with urlopen(url, timeout=30) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Solr admin request failed ({exc.code}) for {url}: {body}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach Solr at {self._solr_url}: {exc.reason}"
            ) from exc

    def start(self):
        self._exec(self._solr_bin, "start", "--user-managed")

    def stop(self):
        self._exec(self._solr_bin, "stop")

    def create_cluster(self):
        self._exec(self._solr_bin, "create", "-c", self.domain)

    def delete_cluster(self):
        self._exec(self._solr_bin, "delete", "-c", self.domain)

    def apply_config(
        self,
        new_managed_schema_xml: Path,
        new_solrconfig_xml: Path,
    ) -> None:
        if not self._use_sudo and not self._collection_conf_path.exists():
            raise FileNotFoundError(
                f"Collection config directory not found: {self._collection_conf_path}. "
                "Create the collection first, or set SOLR_HOME to the active Solr home."
            )

        config_files = {
            new_managed_schema_xml: self._collection_conf_path / "managed-schema.xml",
            new_solrconfig_xml: self._collection_conf_path / "solrconfig.xml",
        }
        for source, destination in config_files.items():
            self._copy_config_file(source, destination)

    def reload_collection(self) -> None:
        self._admin_request(
            "admin/cores",
            {
                "action": "RELOAD",
                "core": self.domain,
            },
        )

    def apply_config_and_reload(
        self,
        new_managed_schema_xml: Path,
        new_solrconfig_xml: Path,
    ) -> None:
        self.apply_config(new_managed_schema_xml, new_solrconfig_xml)
        self.reload_collection()

    def create_cluster_with_metadata(
        self,
        new_managed_schema_xml: Path,
        new_solrconfig_xml: Path,
        metadata_path: Path,
    ):
        self.create_cluster()
        self.apply_config_and_reload(new_managed_schema_xml, new_solrconfig_xml)
        self.post_metadata(metadata_path)

    def reload_collection_with_metadata(
        self,
        new_managed_schema_xml: Path,
        new_solrconfig_xml: Path,
        metadata_path: Path,
    ) -> None:
        self.apply_config_and_reload(new_managed_schema_xml, new_solrconfig_xml)
        self.post_metadata(metadata_path)

    def post_metadata(self, metadata_path: Path):
        self._exec(
            self._solr_bin,
            "post",
            "-c",
            self.domain,
            str(metadata_path.resolve()),
        )


# --- domain helpers -----------------------------------------------------------

# Every target main.py knows about is a valid Solr domain — its target_id
# doubles as the Solr core/collection name (see e.g. conf/workflow/*.yaml's
# tasks.mcp_search.retrieval_contract.solr_core, which is set to the same
# name for bologna/valencia/paris/nyc). A domain with no conf/solr/<domain>/
# folder yet just fails `create`/`reload` with a clear FileNotFoundError.
DOMAINS = tuple(spec.target_id for spec in _main.TARGETS)


def _conf_dir(domain: str) -> Path:
    return (Path(__file__) / ".." / ".." / ".." / "conf" / "solr" / domain).resolve()


def _target_spec(domain: str) -> "_main.TargetSpec":
    for spec in _main.TARGETS:
        if spec.target_id == domain:
            return spec
    raise ValueError(f"Unknown domain {domain!r}. Supported: {', '.join(DOMAINS)}")


def _metadata_path(domain: str) -> Path:
    # Same OrQAConfig main.py itself builds for this target, so this always
    # points at exactly what `normalize-metadata` last wrote — on write_path
    # when a workflow yaml sets one, on DATADIR otherwise.
    cfg = _main.load_cfg(_target_spec(domain))
    return cfg.normalized_metadata_filepath


def cmd_create(domain: str, use_sudo: bool = False) -> None:
    solr = Solr(domain, use_sudo=use_sudo)
    conf_dir = _conf_dir(domain)
    solr.create_cluster_with_metadata(
        new_managed_schema_xml=conf_dir / "managed-schema.xml",
        new_solrconfig_xml=conf_dir / "solrconfig.xml",
        metadata_path=_metadata_path(domain),
    )


def cmd_delete(domain: str, use_sudo: bool = False) -> None:
    Solr(domain, use_sudo=use_sudo).delete_cluster()


def cmd_post(domain: str, use_sudo: bool = False) -> None:
    Solr(domain, use_sudo=use_sudo).post_metadata(_metadata_path(domain))


def cmd_reload(domain: str, use_sudo: bool = False) -> None:
    solr = Solr(domain, use_sudo=use_sudo)
    conf_dir = _conf_dir(domain)
    solr.apply_config_and_reload(
        new_managed_schema_xml=conf_dir / "managed-schema.xml",
        new_solrconfig_xml=conf_dir / "solrconfig.xml",
    )


def cmd_refresh(domain: str, use_sudo: bool = False) -> None:
    solr = Solr(domain, use_sudo=use_sudo)
    conf_dir = _conf_dir(domain)
    solr.reload_collection_with_metadata(
        new_managed_schema_xml=conf_dir / "managed-schema.xml",
        new_solrconfig_xml=conf_dir / "solrconfig.xml",
        metadata_path=_metadata_path(domain),
    )


def cmd_start(domain: str, use_sudo: bool = False) -> None:
    Solr(domain, use_sudo=use_sudo).start()


def cmd_stop(domain: str, use_sudo: bool = False) -> None:
    Solr(domain, use_sudo=use_sudo).stop()


# --- CLI ----------------------------------------------------------------------

COMMANDS = {
    "create": cmd_create,
    "delete": cmd_delete,
    "post": cmd_post,
    "reload": cmd_reload,
    "refresh": cmd_refresh,
    "start": cmd_start,
    "stop": cmd_stop,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solr_cli",
        description="Manage Solr clusters for supported domains.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=COMMANDS,
        help=(
            "create  – create collection, apply custom config, index metadata\n"
            "delete  – delete the collection\n"
            "post    – (re-)index metadata into an existing collection\n"
            "reload  – copy XML config to the live collection and reload it\n"
            "refresh – reload XML config and then (re-)index metadata\n"
            "start   – start the Solr server manually\n"
            "stop    – stop the Solr server manually"
        ),
    )
    parser.add_argument(
        "domain",
        choices=DOMAINS,
        help="Target domain / collection name.",
    )
    parser.add_argument(
        "--sudo",
        action="store_true",
        help=(
            "Use sudo install when copying XML files into SOLR_HOME. "
            "Useful when Solr runs as a system service under /var/solr/data."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    COMMANDS[args.command](args.domain, use_sudo=args.sudo)


if __name__ == "__main__":
    main()
