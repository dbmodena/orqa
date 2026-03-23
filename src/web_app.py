#!/usr/bin/env python3
"""
web_app.py — Browser UI for the OrQA pipeline.

Run:
    uvicorn web_app:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
os.environ["DATADIR"] = "D:\\"

from conf import load_config
from orqa.statement_generation import stream_generate_statements
from orqa.statement_judge import QueryResponsePipeline
from orqa.utils import load_json

_HERE         = Path(os.path.dirname(__file__))
_PROJECT_ROOT = _HERE.parent
_STATIC_DIR   = _HERE / "static"

_SOURCE_REGISTRY = [
    (
        "nyc",
        "New York City",
        "socrata",
        Path(_PROJECT_ROOT, "conf", "workflow", "nyc.yaml"),
        Path(os.environ["DATADIR"], "orqa", "socrata", "nyc"),
    ),
    (
        "uk",
        "United Kingdom",
        "ckan",
        Path(_PROJECT_ROOT, "conf", "workflow", "uk.yaml"),
        Path(os.environ["DATADIR"], "orqa", "ckan", "uk"),
    ),
    (
        "modena",
        "Modena",
        "ckan",
        Path(_PROJECT_ROOT, "conf", "workflow", "modena.yaml"),
        Path(os.environ["DATADIR"], "open_data", "ckan", "modena"),
    ),
    (
        "canada",
        "Canada",
        "ckan",
        Path(_PROJECT_ROOT, "conf", "workflow", "canada.yaml"),
        Path(os.environ["DATADIR"], "open_data", "ckan", "canada_small"),
    ),
]


@dataclass
class SourceEntry:
    id:    str
    label: str
    type:  str
    cfg:   object | None = None
    error: str    | None = None


def _load_all_sources() -> dict[str, SourceEntry]:
    sources: dict[str, SourceEntry] = {}
    for src_id, label, src_type, yaml_path, data_path in _SOURCE_REGISTRY:
        entry = SourceEntry(id=src_id, label=label, type=src_type)
        if yaml_path.exists():
            try:
                data_path.mkdir(parents=True, exist_ok=True)
                entry.cfg = load_config(yaml_path, data_path)
            except Exception as exc:
                entry.error = str(exc)
        else:
            entry.error = f"Config not found: {yaml_path}"
        sources[src_id] = entry
    return sources


_SOURCES: dict[str, SourceEntry] = _load_all_sources()


# ── Kind helper ───────────────────────────────────────────────────────────────

def _cfg_kind(cfg) -> str:
    """
    Read cfg.statement_generation.kind directly.
    Falls back to 'PANDAS' if the attribute is missing or falsy.
    Works for both instance and class attributes.
    """
    sg = cfg.statement_generation
    # Try direct attribute access — works for class attrs, instance attrs, properties
    try:
        kind = sg.kind
    except AttributeError:
        kind = None
    return (kind or "PANDAS").upper()


# ── Readiness ─────────────────────────────────────────────────────────────────

def _required_files(entry: SourceEntry) -> list[dict]:
    if entry.error or entry.cfg is None:
        return []
    cfg = entry.cfg
    return [
        {
            "name":   "Metadata",
            "path":   str(cfg.metadata_path / "metadata.json"),
            "exists": (cfg.metadata_path / "metadata.json").exists(),
        },
        {
            "name":   "Query candidates (matches.json)",
            "path":   str(cfg.statement_generation.query_candidates_path),
            "exists": cfg.statement_generation.query_candidates_path.exists(),
        },
        {
            "name":   "Datasets folder",
            "path":   str(cfg.datasets_path),
            "exists": cfg.datasets_path.exists() and any(cfg.datasets_path.iterdir()),
        },
        {
            "name":   "LiteLLM config",
            "path":   str(cfg.llm_config_path / "litellm.yaml"),
            "exists": (cfg.llm_config_path / "litellm.yaml").exists(),
        },
    ]


def _statements_info(entry: SourceEntry) -> dict:
    """Count queries across ALL languages stored in the file."""
    if entry.error or entry.cfg is None:
        return {"exists": False, "query_count": 0}
    queries_path: Path = entry.cfg.statement_generation.queries_path
    if not queries_path.exists():
        return {"exists": False, "query_count": 0}
    try:
        data  = load_json(queries_path)
        count = 0
        for model_entries in data.values():
            for kind_entries in model_entries.values():
                for ent in kind_entries.values():
                    queries = ent.get("data", {}).get("queries", [])
                    if queries:
                        count += len(queries)
        return {"exists": count > 0, "query_count": count}
    except Exception:
        return {"exists": False, "query_count": 0}


def _generation_status(entry: SourceEntry) -> dict:
    """Progress for cfg.statement_generation.kind only."""
    empty = {"kind": "PANDAS", "last_idx": -1, "total": 0,
             "is_complete": False, "done_count": 0, "query_count": 0}
    if entry.error or entry.cfg is None:
        return empty
    cfg  = entry.cfg
    kind = _cfg_kind(cfg)
    try:
        candidates_path: Path = cfg.statement_generation.query_candidates_path
        total = len(load_json(candidates_path)) if candidates_path.exists() else 0
    except Exception:
        total = 0
    queries_path: Path = cfg.statement_generation.queries_path
    if not queries_path.exists() or total == 0:
        return {**empty, "kind": kind, "total": total}
    try:
        data = load_json(queries_path)
        done_indices: set[int] = set()
        query_count: int = 0
        for model_entries in data.values():
            for stored_idx, ent in model_entries.get(kind, {}).items():
                done_indices.add(int(stored_idx))
                query_count += len(ent.get("data", {}).get("queries", []))
        done_count = len(done_indices)
        last_idx   = max(done_indices) if done_indices else -1
        return {
            "kind":        kind,
            "last_idx":    last_idx,
            "total":       total,
            "is_complete": done_count >= total,
            "done_count":  done_count,
            "query_count": query_count,
        }
    except Exception:
        return {**empty, "kind": kind, "total": total}


def _get_source_or_404(source: str) -> SourceEntry:
    entry = _SOURCES.get(source)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown source: {source!r}")
    if entry.error:
        raise HTTPException(status_code=400, detail=f"Source config error: {entry.error}")
    return entry


def _read_queries(cfg) -> list[dict]:
    """Return queries for ALL languages stored in the file."""
    queries_path: Path = cfg.statement_generation.queries_path
    if not queries_path.exists():
        return []
    data  = load_json(queries_path)
    items = []
    for _model_key, model_entries in data.items():
        for kind_key, kind_entries in model_entries.items():
            for entry_key, entry in kind_entries.items():
                queries = entry.get("data", {}).get("queries", [])
                if not queries:
                    continue
                tables = entry.get("tables", {})
                for q in queries:
                    items.append({
                        "entry_key":       entry_key,
                        "query_id":        q.get("id"),
                        "question":        q.get("question", ""),
                        "difficulty":      q.get("difficulty", ""),
                        "language":        kind_key,
                        "query_code":      q.get("query") or q.get("code") or q.get("pandas_query", ""),
                        "tables":          tables,
                        "response":        q.get("response"),
                        "produce_result":  q.get("produce_result"),
                        "execution_error": q.get("execution_error"),
                    })
    return items


def _find_query_in_store(cfg, entry_key: str, query_id: int, language: str | None = None):
    """Search across ALL languages, optionally filtered by language."""
    data = load_json(cfg.statement_generation.queries_path)
    for _model_key, model_entries in data.items():
        for kind_key, kind_entries in model_entries.items():
            if language and kind_key != language:
                continue
            q_entry = kind_entries.get(str(entry_key))
            if q_entry is None:
                continue
            query = next(
                (q for q in q_entry.get("data", {}).get("queries", [])
                 if q.get("id") == query_id),
                None,
            )
            if query is not None:
                return q_entry, query, kind_key
    return None, None, None


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="OrQA Pipeline UI")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _statements_to_sse(cfg, resume_from: int = 0):
    kind = _cfg_kind(cfg)
    async for event in stream_generate_statements(cfg, kind=kind, resume_from=resume_from):
        yield _sse(event["type"], event)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/sources")
async def get_sources() -> list[dict]:
    result = []
    for e in _SOURCES.values():
        kind = _cfg_kind(e.cfg) if e.cfg and not e.error else None
        gen  = _generation_status(e) if e.cfg and not e.error else {
            "kind": kind or "PANDAS", "last_idx": -1, "total": 0,
            "is_complete": False, "done_count": 0,
        }
        result.append({
            "id":         e.id,
            "label":      e.label,
            "type":       e.type,
            "error":      e.error,
            "kind":       kind,       # <-- cfg.statement_generation.kind
            "files":      _required_files(e),
            "statements": _statements_info(e),
            "generation": gen,
        })
    return result


@app.get("/api/debug-kind/{source}")
async def debug_kind(source: str) -> dict:
    """
    Call this in your browser to verify what _cfg_kind is reading:
        /api/debug-kind/nyc
    """
    e = _SOURCES.get(source)
    if e is None:
        return {"error": f"Unknown source: {source}"}
    if e.error:
        return {"error": e.error}
    cfg = e.cfg
    sg  = cfg.statement_generation

    # Inspect the statement_generation object thoroughly
    kind_direct   = getattr(sg, "kind", "<<AttributeError>>")
    sg_type       = type(sg).__name__
    sg_dict       = vars(sg) if hasattr(sg, "__dict__") else "<<no __dict__>>"
    sg_class_vars = {k: v for k, v in vars(type(sg)).items()
                     if not k.startswith("__") and not callable(v)}

    return {
        "source":              source,
        "cfg_kind_result":     _cfg_kind(cfg),
        "kind_direct":         str(kind_direct),
        "statement_generation_type":      sg_type,
        "statement_generation_instance_vars": str(sg_dict),
        "statement_generation_class_vars":    str(sg_class_vars),
    }


@app.post("/api/run/statements/{source}")
async def run_statements(source: str) -> StreamingResponse:
    entry = _get_source_or_404(source)
    return StreamingResponse(
        _statements_to_sse(entry.cfg, resume_from=0),
        media_type="text/event-stream",
    )


@app.post("/api/resume/statements/{source}")
async def resume_statements(source: str) -> StreamingResponse:
    entry  = _get_source_or_404(source)
    status = _generation_status(entry)
    return StreamingResponse(
        _statements_to_sse(entry.cfg, resume_from=status["last_idx"] + 1),
        media_type="text/event-stream",
    )


@app.get("/api/queries/{source}")
async def get_queries(source: str) -> list[dict]:
    entry = _get_source_or_404(source)
    return _read_queries(entry.cfg)


@app.get("/api/debug/{source}")
async def debug_source(source: str) -> dict:
    entry = _get_source_or_404(source)
    cfg   = entry.cfg
    queries_path: Path = cfg.statement_generation.queries_path
    if not queries_path.exists():
        return {"error": f"File not found: {queries_path}"}
    data        = load_json(queries_path)
    summary     = {}
    total_all   = 0
    total_shown = 0
    for model_key, model_entries in data.items():
        summary[model_key] = {}
        for kind_key, kind_entries in model_entries.items():
            summary[model_key][kind_key] = {}
            for entry_key, ent in kind_entries.items():
                queries = ent.get("data", {}).get("queries", [])
                summary[model_key][kind_key][entry_key] = {
                    "status":      ent.get("status", "MISSING"),
                    "query_count": len(queries),
                }
                total_all += len(queries)
                if queries:
                    total_shown += len(queries)
    return {
        "queries_path":              str(queries_path),
        "active_kind":               _cfg_kind(cfg),
        "total_queries_in_file":     total_all,
        "total_queries_shown_in_ui": total_shown,
        "breakdown":                 summary,
    }


@app.post("/api/response/{source}/{entry_key}/{query_id}")
async def generate_response(source: str, entry_key: str, query_id: int, language: str | None = None) -> dict:
    entry    = _get_source_or_404(source)
    pipeline = QueryResponsePipeline(entry.cfg)
    q_entry, query, kind = _find_query_in_store(entry.cfg, entry_key, query_id, language)
    if q_entry is None:
        raise HTTPException(status_code=404, detail=f"Query {entry_key}/{query_id} not found")
    return await pipeline.run_single_async(q_entry, query, entry_key, language=kind, generate_nl=True)


@app.post("/api/execute/{source}/{entry_key}/{query_id}")
async def execute_query(source: str, entry_key: str, query_id: int, language: str | None = None) -> dict:
    entry    = _get_source_or_404(source)
    pipeline = QueryResponsePipeline(entry.cfg)
    q_entry, query, kind = _find_query_in_store(entry.cfg, entry_key, query_id, language)
    if q_entry is None:
        raise HTTPException(status_code=404, detail=f"Query {entry_key}/{query_id} not found")
    return await pipeline.run_single_async(q_entry, query, entry_key, language=kind, generate_nl=False)