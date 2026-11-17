"""OrQA's Solr from the command line (``solr.instance``).

    python src/solr/cli.py start | stop | status                # Solr and the embedding server
    python src/solr/cli.py reembed <core>... --model bge-m3     # new vectors into <core>_build
    python src/solr/cli.py publish <core>                       # serve <core>_build as <core>
    python src/solr/cli.py drop <core> --yes
    python src/solr/cli.py sync-config                          # conf/solr/portal -> the home

A portal's core is built from OrQA's data by the workflow step
``build-solr-core``. Settings come from ``.env``: ``ORQA_SOLR_HOME``
(required), ``ORQA_SOLR_PORT`` (8984), ``ORQA_SOLR_HEAP`` (2g),
``SOLR_PATH`` (/opt/solr); for the embedding server (``solr.embedder``)
``ORQA_OLLAMA_MODELS`` (required) and ``ORQA_OLLAMA_PORT`` (11435). Every command that writes acts on OrQA's own
server only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    # Run as a script, this folder is sys.path[0]; the `solr` package lives in src/.
    sys.path[0] = str(_SRC)

from dotenv import load_dotenv  # noqa: E402

from solr.embedder import EmbedderError, OllamaServer  # noqa: E402
from solr.instance import SolrInstance, SolrInstanceError  # noqa: E402


def _status(instance: SolrInstance, embedder: OllamaServer) -> None:
    if embedder.running():
        print(f"Embedding server: {embedder.url} (models {embedder.models}): {', '.join(embedder.models_available())}")
    else:
        print(f"Embedding server: {embedder.url} not running")
    print(f"OrQA Solr: {instance.url} (home {instance.home})")
    if not instance.running():
        print("  not running")
        return
    for name, info in instance.cores().items():
        docs = "?" if info["docs"] is None else info["docs"]
        print(f"  {name:20s} {docs:>7} documents  {info['size'] or '(not loaded)'}")
    drift = instance.configset_drift()
    if drift:
        print(f"  configset differs from conf/solr/portal: {drift} (run sync-config)")


def _reembed(instance: SolrInstance, embedder: OllamaServer, args: argparse.Namespace) -> int:
    from orqa.agent.llm_client.EmbeddingClient import solr_embedding_client
    from solr.build import reembed_core
    from solr.client import LocalSolrClient

    instance.start()
    if not args.model.startswith(("cohere.", "ocid1.")):
        embedder.ensure(args.model)
    options = {"num_thread": args.threads} if args.threads else None
    embedding_client = solr_embedding_client(
        args.model, None, _SRC.parent / "conf" / "llm", batch_size=16, options=options,
    )
    client = LocalSolrClient(base_url=instance.url, timeout=300, retries=3)
    cores = instance.cores()
    for core in args.cores:
        if core not in cores:
            raise SolrInstanceError(f"no core {core!r} on OrQA's Solr")
    for core in args.cores:
        report = reembed_core(instance, client, core, embedding_client)
        comparison = report["comparison"]
        print(
            f"{core}: {report['build']} holds {comparison['documents']['built']} documents "
            f"({comparison['added']} added, {comparison['removed']} removed, field changes "
            f"{comparison['kept_with_field_changes'] or 'none'}), vectors {report['vectors']} "
            f"(report {report['path']})", flush=True,
        )
        if args.publish:
            # The vectors' labels change with the model; nothing else may.
            changed = set(comparison["kept_with_field_changes"]) - {"embedding_model", "representation_version"}
            if comparison["added"] or comparison["removed"] or changed:
                raise SolrInstanceError(f"{core}: the build's documents differ from the core's; not published")
            previous = instance.publish(core)
            print(f"{core}: published; the replaced core is {previous}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python src/solr/cli.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("start", help="start OrQA's Solr and its embedding server")
    commands.add_parser("stop", help="stop OrQA's Solr and its embedding server")
    commands.add_parser("status", help="cores and document counts")
    commands.add_parser("sync-config", help="copy conf/solr/portal to the home and reload the cores")
    reembed = commands.add_parser("reembed", help="re-make cores' vectors with another model, into <core>_build")
    reembed.add_argument("cores", nargs="+")
    reembed.add_argument("--model", required=True, help="e.g. bge-m3 (OrQA's Ollama) or cohere.embed-v4.0 (OCI)")
    reembed.add_argument(
        "--threads", type=int, default=0,
        help="CPU threads of the embedding requests (Ollama num_thread; 24 is ~35%% faster than the "
        "default 8). Ollama reloads the model whenever a request without it (a question) comes in "
        "between, so use it only while nothing else queries the embedding server",
    )
    reembed.add_argument("--publish", action="store_true", help="publish each build whose documents are unchanged")
    publish = commands.add_parser("publish", help="serve <core>_build as <core>, keeping <core>_previous")
    publish.add_argument("core")
    drop = commands.add_parser("drop", help="delete a core of OrQA's Solr and its index")
    drop.add_argument("core")
    drop.add_argument("--yes", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(_SRC.parent / ".env")
    args = build_parser().parse_args(argv)
    try:
        instance = SolrInstance.from_env()
        embedder = OllamaServer.from_env()
        if args.command == "start":
            print("Solr", "started" if instance.start() else "already running", instance.url)
            print("embedding server", "started" if embedder.start() else "already running", embedder.url)
        elif args.command == "stop":
            print("Solr", "stopped" if instance.stop() else "not running")
            print("embedding server", "stopped" if embedder.stop() else "not running")
        elif args.command == "status":
            _status(instance, embedder)
        elif args.command == "reembed":
            return _reembed(instance, embedder, args)
        elif args.command == "sync-config":
            print("updated:", instance.sync_configset() or "nothing")
        elif args.command == "publish":
            previous = instance.publish(args.core)
            print(f"{args.core} now serves the build" + (f"; the replaced core is {previous}" if previous else ""))
        elif args.command == "drop":
            instance.unload_core(args.core)
            print(f"dropped {args.core}")
    except (SolrInstanceError, EmbedderError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
