#!/usr/bin/env python3
"""
run_monitor.py — Browser view of the portal runs in progress.

A READ-ONLY monitor: which models the runs use (litellm.yaml's LLM and
judges, the embedding model of discovery and of the Solr gate, what OrQA's
Ollama has loaded and on which processor) and how far each portal's three
phases are — candidates-discovery (embeddings, clusters, task selection
against its token and dataset caps), generate-query-candidates and
generate-statements (groups, approved questions, tokens, ETA). It reads the
workflow yamls, the launcher logs (``$ORQA_RUNS_DIR/<portal>.log``, written
by ``run_portal.sh``, one ``[run_portal] ... step <step> start|exit``
marker per step), the step outputs and the generation trace; it never
writes, and asks Solr and Ollama only for their status.

Run (from src/):
    uvicorn run_monitor:app --port 8001

Served through a public tunnel, set ``ORQA_MONITOR_TOKEN``: every request
must then carry it, as ``?token=`` once (it is kept in a cookie after that).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
import yaml
from dotenv import dotenv_values, load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
load_dotenv(_PROJECT_ROOT / ".env")

WORKFLOWS = _PROJECT_ROOT / "conf" / "workflow"
LITELLM = _PROJECT_ROOT / "conf" / "llm" / "litellm.yaml"
RUNS_DIR = Path(os.environ.get("ORQA_RUNS_DIR", "/data/bilel/orqa/runs"))
# portal -> (--country, --city) of main.py
PORTALS = {
    "valencia": ("spain", "valencia"),
    "paris": ("france", "paris"),
    "bologna": ("italy", "bologna"),
    "nyc": ("usa", "nyc"),
    "uk": ("uk", None),
}
STEPS = ("candidates-discovery", "generate-query-candidates", "generate-statements")
LOG_TAIL_LINES = 14
# The access token a public tunnel needs (empty: no check, for localhost).
TOKEN = os.environ.get("ORQA_MONITOR_TOKEN", "").strip()
_TOKEN_COOKIE = "orqa_monitor"

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_MARKER = re.compile(r"^\[run_portal\] (\S+) step (\S+) (start|exit)(?: (-?\d+))?")
_BUDGET = re.compile(r"budget — tokens left: ([\d,-]+), datasets left: (-?\d+)")
_BATCH = re.compile(r"batch (\d+)/(\d+)")
# litellm's line for each model a process's Router registers.
_REGISTERED = re.compile(r"register_model: model=(\S+)")
_ROUTER_ALIAS = re.compile(r"^(primary|fallback_\d+|\w+_judge_\d+)$")


# ------------------------------------------------------------------ files

class _Tail:
    """New lines of a growing file since the last call (restarts when the
    file is replaced or shrinks), so a refresh never re-reads a long log."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.inode = None

    def new_lines(self) -> tuple[bool, list[str]]:
        """(restarted, lines); ``restarted`` tells the caller to drop what it
        derived from the previous file."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            restarted = self.offset != 0
            self.offset, self.inode = 0, None
            return restarted, []
        restarted = stat.st_ino != self.inode or stat.st_size < self.offset
        if restarted:
            self.offset, self.inode = 0, stat.st_ino
        with open(self.path, "rb") as handle:
            handle.seek(self.offset)
            data = handle.read()
        # Keep a partly written last line for the next call.
        end = data.rfind(b"\n") + 1
        self.offset += end
        text = data[:end].decode("utf-8", errors="replace")
        return restarted, [_ANSI.sub("", line).rstrip("\r") for line in text.splitlines()]


_tails: dict[Path, _Tail] = {}
_derived: dict[Path, dict] = {}


def _follow(path: Path, fold, initial) -> Any:
    """Fold the new lines of ``path`` into its derived state."""
    tail = _tails.setdefault(path, _Tail(path))
    restarted, lines = tail.new_lines()
    if restarted or path not in _derived:
        _derived[path] = initial()
    state = _derived[path]
    for line in lines:
        fold(state, line)
    return state


_json_cache: dict[Path, tuple[float, Any]] = {}


def _load_json(path: Path) -> Any:
    """A JSON file, re-read only when it changes; None while missing or
    half-written."""
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    cached = _json_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return cached[1] if cached else None
    _json_cache[path] = (mtime, value)
    return value


def _parse_time(text: Optional[str]) -> Optional[float]:
    """Epoch seconds of an ISO time, naive ones read as local time."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


# ------------------------------------------------------------------ models and servers

def _expand(model: str) -> str:
    """A deployment's model string with ${NAME} read from the environment."""
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), model)


def _short(model: str) -> str:
    # An endpoint OCID is long; its tail identifies it.
    return f"{model[:18]}…{model[-12:]}" if len(model) > 40 else model


def _references(config: dict) -> list[str]:
    """Every model reference of the LLM config: primary, fallbacks, judges."""
    refs = [config.get("model")] + list(config.get("fallback_models") or [])
    for judges in (config.get("judge_profiles") or {}).values():
        refs += [j.get("model") if isinstance(j, dict) else j for j in judges or []]
    return [str(r) for r in refs if r]


def deployment_names() -> dict[str, str]:
    """The LiteLLM model string each deployment resolves to -> its name, to
    name what a process registered."""
    config = yaml.safe_load(LITELLM.read_text()) or {}
    names: dict[str, str] = {}
    for name, entry in (config.get("deployments") or {}).items():
        model = _expand(str((entry or {}).get("model") or ""))
        # Twins on one endpoint (qwen3, qwen3-thinking) share the string: keep the first.
        names.setdefault(model, name)
    return names


def llm_status() -> dict[str, Any]:
    config = yaml.safe_load(LITELLM.read_text()) or {}
    model = config.get("model")
    deployment = (config.get("deployments") or {}).get(model) or {}
    served = _short(_expand(str(deployment.get("model") or model or "")))
    # Read fresh: the pin may be set after the monitor started.
    pinned = (dotenv_values(_PROJECT_ROOT / ".env").get("ORQA_ONLY_LLM") or "").strip() or None
    references = _references(config)
    thinking = (((deployment.get("extra_body") or {}).get("chat_template_kwargs") or {}).get("enable_thinking"))
    base = str(deployment.get("api_base") or "")
    if base.startswith("os.environ/"):
        base = os.environ.get(base.split("/", 1)[1], "")
    return {
        "model": model,
        "served_as": served,
        "endpoint": re.sub(r"^https?://([^/]+).*$", r"\1", base) if base else None,
        "thinking": None if thinking is None else bool(thinking),
        "judges": config.get("judge_profiles") or {},
        "fallback_models": config.get("fallback_models") or [],
        "litellm_embedding_model": config.get("embedding_model"),
        "pinned": pinned,
        "references": sorted(set(references)),
        "config_matches_pin": None if pinned is None else all(r == pinned for r in references),
    }


def embedding_server_status() -> dict[str, Any]:
    port = int(os.environ.get("ORQA_OLLAMA_PORT") or 11435)
    url = f"http://127.0.0.1:{port}"
    status: dict[str, Any] = {"url": url, "up": False, "loaded": [], "models": []}
    try:
        loaded = requests.get(f"{url}/api/ps", timeout=3).json().get("models", [])
        models = requests.get(f"{url}/api/tags", timeout=3).json().get("models", [])
    except (requests.RequestException, ValueError):
        return status
    status["up"] = True
    status["loaded"] = [
        {
            "name": m.get("name"),
            "processor": "GPU" if m.get("size_vram") else "CPU",
            "size_gb": round((m.get("size") or 0) / 1e9, 2),
        }
        for m in loaded
    ]
    status["models"] = [m.get("name") for m in models]
    return status


def solr_status() -> dict[str, Any]:
    port = int(os.environ.get("ORQA_SOLR_PORT") or 8984)
    url = f"http://localhost:{port}/solr"
    status: dict[str, Any] = {"url": url, "up": False, "cores": {}}
    try:
        payload = requests.get(
            f"{url}/admin/cores", params={"action": "STATUS", "wt": "json"}, timeout=5
        ).json()
    except (requests.RequestException, ValueError):
        return status
    status["up"] = True
    status["cores"] = {
        name: (core.get("index") or {}).get("numDocs")
        for name, core in sorted((payload.get("status") or {}).items())
    }
    return status


def disk_status() -> dict[str, float]:
    disks = {}
    for mount in ("/", "/data"):
        try:
            disks[mount] = round(shutil.disk_usage(mount).free / 1e9, 1)
        except OSError:
            pass
    return disks


# ------------------------------------------------------------------ processes

def _main_processes() -> dict[str, dict[str, Any]]:
    """portal -> its running ``main.py`` (pid, steps, start time)."""
    found: dict[str, dict[str, Any]] = {}
    ticks = os.sysconf("SC_CLK_TCK")
    boot = time.time() - float(Path("/proc/uptime").read_text().split()[0])
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        if not any(a.endswith("main.py") for a in argv) or "--country" not in argv:
            continue
        country = argv[argv.index("--country") + 1] if argv.index("--country") + 1 < len(argv) else ""
        city = argv[argv.index("--city") + 1] if "--city" in argv and argv.index("--city") + 1 < len(argv) else None
        portal = next((p for p, target in PORTALS.items() if target == (country, city)), None)
        if portal is None:
            continue
        steps = argv[argv.index("--steps") + 1:] if "--steps" in argv else []
        started = boot + int(stat.rsplit(")", 1)[1].split()[19]) / ticks
        found[portal] = {"pid": int(entry.name), "steps": steps, "started": started}
    return found


# ------------------------------------------------------------------ one portal

def _settings(portal: str) -> dict[str, Any]:
    parsed = yaml.safe_load((WORKFLOWS / f"{portal}.yaml").read_text()) or {}
    tasks = parsed.get("tasks") or {}
    discovery = tasks.get("candidates_discovery") or {}
    generation = tasks.get("query_generation") or {}
    solr = (tasks.get("search") or {}).get("solr") or {}
    return {
        "write_path": Path(parsed.get("write_path") or ""),
        "tokens_budget": int(discovery.get("tokens_budget") or 0),
        "max_datasets": int(discovery.get("max_datasets_to_process") or 0),
        "neighbor_threshold": discovery.get("cosine_similarity_threshold"),
        "discovery_embedding_model": discovery.get("embedding_model"),
        "gate_embedding_model": solr.get("embedding_model"),
        "solr_core": solr.get("core"),
        "single_target": int(generation.get("single_table_query_count") or 0),
        "multi_target": int(generation.get("multi_table_query_count") or 0),
    }


def _fold_launcher(state: dict, line: str) -> None:
    marker = _MARKER.match(line)
    if marker:
        when, step, event, code = marker.groups()
        entry = state["steps"].setdefault(step, {})
        if event == "start":
            state["waiting"] = None
            entry.clear()
            entry["started"] = _parse_time(when)
            state["current"] = step
            if step == "candidates-discovery":
                state["discovery"] = {"datasets_done": 0, "tokens_left": None, "datasets_left": None,
                                      "embed_batch": None, "last_dataset_at": None}
        else:
            entry["ended"] = _parse_time(when)
            entry["exit"] = int(code) if code is not None else None
        return
    if line.startswith("[run_portal]"):
        state["events"].append(line)
        # A waiter holding the next step back (nyc_after_others.sh).
        if " waiting: " in line:
            state["waiting"] = line.split(" waiting: ", 1)[1]
        return
    registered = _REGISTERED.search(line)
    # The Router's own aliases (primary, fallback_N, plan_judge_N) name no model.
    if registered and not _ROUTER_ALIAS.match(registered.group(1)):
        state["registered"].add(registered.group(1))
    discovery = state.get("discovery")
    if discovery is not None and state.get("current") == "candidates-discovery":
        if "DISCOVERY" in line:
            discovery["datasets_done"] += 1
            discovery["last_dataset_at"] = time.time()
        budget = _BUDGET.search(line)
        if budget:
            discovery["tokens_left"] = int(budget.group(1).replace(",", ""))
            discovery["datasets_left"] = int(budget.group(2))
        batch = _BATCH.search(line)
        if batch:
            discovery["embed_batch"] = [int(batch.group(1)), int(batch.group(2))]
    if line.strip():
        state["tail"].append(line[:400])
        del state["tail"][:-LOG_TAIL_LINES]


def _count_jsonl(path: Path) -> dict[str, Any]:
    """Line count, task kinds and tokens of a JSON-lines discovery file."""
    def fold(state: dict, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            entry = json.loads(line)
        except ValueError:
            return
        state["lines"] += 1
        if "task" in entry:
            state["tasks"][entry["task"]] = state["tasks"].get(entry["task"], 0) + 1
        usage = entry.get("token_usage") or {}
        state["tokens"] += int(usage.get("total_tokens") or 0)
        if "dataset" in entry:
            state["datasets"].add(entry["dataset"])

    state = _follow(path, fold, lambda: {"lines": 0, "tasks": {}, "tokens": 0, "datasets": set()})
    return {"lines": state["lines"], "tasks": state["tasks"], "tokens": state["tokens"],
            "datasets": len(state["datasets"])}


def _discovery(settings: dict, launcher: dict) -> dict[str, Any]:
    folder = settings["write_path"] / "candidates_discovery"
    manifest = _load_json(folder / "embeddings_manifest.json") or {}
    models: dict[str, int] = {}
    for entry in manifest.values():
        models[entry.get("model")] = models.get(entry.get("model"), 0) + 1
    proposed = _count_jsonl(folder / "proposed_tasks_semantic.json")
    results = _count_jsonl(folder / "tasks_results_semantic.json")
    progress = launcher.get("discovery") or {}
    step = launcher["steps"].get("candidates-discovery") or {}
    tokens_left, datasets_left = progress.get("tokens_left"), progress.get("datasets_left")
    done = (settings["max_datasets"] - datasets_left) if datasets_left is not None else progress.get("datasets_done", 0)
    rate = None
    if step.get("started") and done:
        rate = done / max(1.0, (step.get("ended") or time.time()) - step["started"])
    return {
        "embeddings": {"vectors": len(manifest), "models": models, "batch": progress.get("embed_batch")},
        "clusters": (folder / "clusters.json").exists(),
        "datasets_done": done,
        "max_datasets": settings["max_datasets"],
        "tokens_spent": (settings["tokens_budget"] - tokens_left) if tokens_left is not None else proposed["tokens"],
        "tokens_budget": settings["tokens_budget"],
        "pairs_judged": proposed["lines"],
        "candidates": results["lines"],
        "candidate_kinds": results["tasks"],
        "datasets_per_hour": round(rate * 3600, 1) if rate else None,
    }


def _query_candidates(settings: dict) -> dict[str, Any]:
    folder = settings["write_path"] / "candidates_discovery"
    groups = _load_json(folder / "query_candidates_semantic.json")
    final = _load_json(folder / "final_generation_candidates_semantic.json")
    return {
        "groups": len(groups) if isinstance(groups, (list, dict)) else None,
        "final": len(final) if isinstance(final, (list, dict)) else None,
    }


def _statements(settings: dict, launcher: dict, query_candidates: dict) -> dict[str, Any]:
    trace = settings["write_path"] / "log" / "generation" / "latest"
    run = _load_json(trace / "run.json") or {}
    step = launcher["steps"].get("generate-statements") or {}
    started = _parse_time(run.get("started_at"))
    # Only the trace of this launch's statements step, not an older run's.
    if not step.get("started") or not started or started < step["started"] - 120:
        return {"started": False}

    def fold(state: list, line: str) -> None:
        try:
            state.append(json.loads(line))
        except ValueError:
            pass

    rows = _follow(trace.resolve() / "groups.jsonl", fold, list)
    by_mode: dict[str, dict[str, Any]] = {}
    statuses: dict[str, int] = {}
    models: dict[str, int] = {}
    tokens = 0
    for row in rows:
        mode = by_mode.setdefault(row.get("mode") or "?", {"groups": 0, "questions": 0, "elapsed": []})
        mode["groups"] += 1
        mode["questions"] += int(row.get("queries") or 0)
        if row.get("elapsed_s"):
            mode["elapsed"].append(float(row["elapsed_s"]))
        statuses[row.get("status") or "?"] = statuses.get(row.get("status") or "?", 0) + 1
        if row.get("model"):
            models[row["model"]] = models.get(row["model"], 0) + 1
        tokens += int((row.get("tokens") or {}).get("total_tokens") or 0)
    multi_target = settings["multi_target"]
    if query_candidates.get("groups") is not None:
        multi_target = min(multi_target, query_candidates["groups"])
    targets = {"single": settings["single_target"], "multi": multi_target}
    eta = 0.0
    for mode, target in targets.items():
        info = by_mode.get(mode) or {"groups": 0, "elapsed": []}
        known = by_mode.get(mode, {}).get("elapsed") or [e for m in by_mode.values() for e in m["elapsed"]]
        mean = sum(known) / len(known) if known else (180.0 if mode == "single" else 420.0)
        eta += max(0, target - info["groups"]) * mean
    return {
        "started": True,
        "run_id": run.get("run_id"),
        "run_status": run.get("status"),
        "modes": {
            mode: {
                "groups": by_mode.get(mode, {}).get("groups", 0),
                "target": target,
                "questions": by_mode.get(mode, {}).get("questions", 0),
                "median_s": _median(by_mode.get(mode, {}).get("elapsed") or []),
            }
            for mode, target in targets.items()
        },
        "statuses": statuses,
        "models": models,
        "tokens": tokens,
        "last_group_at": _parse_time(rows[-1].get("ts")) if rows else None,
        "eta_s": round(eta),
    }


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return round(ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2, 1)


def portal_status(portal: str, processes: dict[str, dict], names: dict[str, str]) -> dict[str, Any]:
    settings = _settings(portal)
    launcher = _follow(RUNS_DIR / f"{portal}.log", _fold_launcher,
                       lambda: {"steps": {}, "current": None, "tail": [], "events": [], "discovery": None,
                                "registered": set()})
    steps = {}
    for step in STEPS:
        entry = launcher["steps"].get(step) or {}
        if launcher.get("waiting") and entry.get("exit") not in (None, 0):
            state = "waiting"
        elif "exit" in entry:
            state = "done" if entry["exit"] == 0 else "failed"
        elif entry.get("started"):
            state = "running" if portal in processes else "stopped"
        else:
            state = "pending"
        steps[step] = {"state": state, "started": entry.get("started"), "ended": entry.get("ended"),
                       "exit": entry.get("exit")}
    query_candidates = _query_candidates(settings)
    return {
        "portal": portal,
        "process": processes.get(portal),
        "launched": bool(launcher["steps"]),
        "current": launcher.get("current"),
        "waiting": launcher.get("waiting"),
        "steps": steps,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in settings.items()},
        "discovery": _discovery(settings, launcher),
        "query_candidates": query_candidates,
        "statements": _statements(settings, launcher, query_candidates),
        "events": launcher["events"][-4:],
        # The LLMs this portal's processes registered (litellm's log lines), named.
        "llm_registered": sorted(names.get(m, _short(m)) for m in launcher["registered"]),
        "log_tail": launcher["tail"],
    }


# ------------------------------------------------------------------ app

app = FastAPI(title="OrQA run monitor")


@app.middleware("http")
async def require_token(request: Request, call_next):
    if not TOKEN:
        return await call_next(request)
    given = request.query_params.get("token") or request.cookies.get(_TOKEN_COOKIE) or ""
    if not secrets.compare_digest(given.encode(), TOKEN.encode()):
        return PlainTextResponse("Forbidden: open the monitor with its ?token= link.", status_code=403)
    response = await call_next(request)
    if request.query_params.get("token"):
        response.set_cookie(_TOKEN_COOKIE, TOKEN, max_age=30 * 86400, httponly=True, secure=True, samesite="strict")
    return response


@app.get("/api/status")
def status() -> dict[str, Any]:
    processes = _main_processes()
    return {
        "generated_at": time.time(),
        "llm": llm_status(),
        "embedding_server": embedding_server_status(),
        "solr": solr_status(),
        "disk_free_gb": disk_status(),
        "portals": [portal_status(portal, processes, deployment_names()) for portal in PORTALS],
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_HERE / "static" / "monitor.html").read_text()
