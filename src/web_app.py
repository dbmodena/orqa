#!/usr/bin/env python3
"""
web_app.py — Browser UI for generated OrQA queries.

A READER and EXECUTOR over ``generated_queries*.json``: browse every
generated query per source, inspect its full judging history (plan panel
and code panel, attempt by attempt), and re-execute its code against the
real datasets. Query GENERATION is not exposed here — it runs through the
pipeline CLI, and this app never mutates the queries file.

Run:
    uvicorn web_app:app --reload --port 8000
"""

from __future__ import annotations

import math
import os
import traceback
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from conf import load_config
from orqa.benchmark.index import load_index
from orqa.benchmark.questions import META_KEY, iter_questions, match_plan_feedback
from orqa.embedding_discovery.clustering import compute_cluster_projection, save_cluster_projection
from orqa.queries.query_execution import QueryExecutor
from orqa.utils import load_json

_HERE = Path(os.path.dirname(__file__))
_PROJECT_ROOT = _HERE.parent
_STATIC_DIR = _HERE / "static"

_DATADIR = Path(os.environ.get("DATADIR", str(_PROJECT_ROOT / "data")))

_SOURCE_REGISTRY = [
    (
        "nyc",
        "New York City",
        "socrata",
        Path(_PROJECT_ROOT, "conf", "workflow", "nyc.yaml"),
        Path(_DATADIR, "orqa", "socrata", "nyc"),
    ),
    (
        "uk",
        "United Kingdom",
        "ckan",
        Path(_PROJECT_ROOT, "conf", "workflow", "uk.yaml"),
        Path(_DATADIR, "orqa", "ckan", "uk"),
    ),
    (
        "modena",
        "Modena",
        "ckan",
        Path(_PROJECT_ROOT, "conf", "workflow", "modena.yaml"),
        Path(_DATADIR, "open_data", "ckan", "modena"),
    ),
    (
        "canada",
        "Canada",
        "ckan",
        Path(_PROJECT_ROOT, "conf", "workflow", "canada.yaml"),
        Path(_DATADIR, "open_data", "ckan", "canada_small"),
    ),
]

# Rows shipped to the browser per executed result — the full frame stays
# server-side; the UI reports "showing N of M".
_MAX_RESULT_ROWS = 100


@dataclass
class SourceEntry:
    id: str
    label: str
    type: str
    cfg: object | None = None
    error: str | None = None


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


def _get_source_or_404(source: str) -> SourceEntry:
    entry = _SOURCES.get(source)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown source: {source!r}")
    if entry.error:
        raise HTTPException(status_code=400, detail=f"Source config error: {entry.error}")
    return entry


# Built lazily (first keyword-search request per source), not at startup like
# _SOURCES: an index needs the metadata already crawled/indexed and, for the
# elasticsearch backend, a reachable cluster — neither should block the app
# from serving everything else when they're not there yet.
_INDEX_CACHE: dict[str, Any] = {}


def _get_index(source: str, cfg) -> Optional[Any]:
    if source not in _INDEX_CACHE:
        _INDEX_CACHE[source] = load_index(cfg)
    return _INDEX_CACHE[source]


# ── Queries file reading ──────────────────────────────────────────────────────


def _query_status(q: dict) -> str:
    """Coarse per-query status chip for the list view."""
    history = q.get("attempt_history") or []
    for att in reversed(history):
        outcome = str(att.get("outcome", "")).lower()
        if outcome:
            return outcome
    return "unknown"


def _read_queries(cfg) -> list[dict]:
    """Flatten the queries file into list items addressed by
    (kind, section, entry_key, qnum)."""
    queries_path: Path = cfg.statement_generation.queries_path
    if not queries_path.exists():
        return []
    data = load_json(queries_path)
    items: list[dict] = []
    for kind_key in data:
        for section, entry_key, qnum, q, meta in iter_questions(data, kind_key):
            plan_fb = match_plan_feedback(meta or {}, q.get("question", ""))
            plan_attempts = (plan_fb or {}).get("attempts") or []
            # The plan the query was ultimately generated from — the last
            # attempt's proposal (earlier ones were revised away).
            final_plan = (plan_attempts[-1].get("proposed_plan") if plan_attempts else None) or {}
            items.append(
                {
                    "kind": kind_key,
                    "section": section,
                    "entry_key": entry_key,
                    "qnum": qnum,
                    "question": q.get("question", ""),
                    "question_keywords": q.get("question_keywords") or [],
                    "difficulty": q.get("difficulty", ""),
                    "code": q.get("code", ""),
                    "topic": q.get("topic", ""),
                    "story": q.get("story", ""),
                    "status": _query_status(q),
                    "task_types": final_plan.get("task_types") or [],
                    "expected_result_type": q.get("expected_result_type")
                    or final_plan.get("expected_result_type")
                    or "",
                    "expected_result_description": q.get("expected_result_description")
                    or final_plan.get("expected_result_description")
                    or "",
                    "tables": q.get("tables") or [],
                    "tables_map": (meta or {}).get("tables") or {},
                    "attempt_history": q.get("attempt_history") or [],
                    "plan_attempts": plan_attempts,
                    "model": (meta or {}).get("model", ""),
                }
            )
    return items


# ── Stats aggregation ──────────────────────────────────────────────────────────


def _judge_name(raw: str) -> str:
    """Strip the ``oci/`` provider prefix judge model names carry."""
    raw = str(raw or "judge")
    return raw.split("/", 1)[1] if raw.startswith("oci/") else raw


def _bump(buckets: dict, key: str, approved: bool) -> None:
    b = buckets.setdefault(key, {"total": 0, "rejected": 0})
    b["total"] += 1
    if not approved:
        b["rejected"] += 1


def _finalize_rates(buckets: dict) -> dict:
    return {
        k: {
            "total": v["total"],
            "rejected": v["rejected"],
            "rate": round(v["rejected"] / v["total"], 4) if v["total"] else 0.0,
        }
        for k, v in sorted(buckets.items())
    }



def _compute_stats(items: list[dict]) -> dict:
    """Failure/rejection-rate breakdowns over already-flattened queries
    (see ``_read_queries``).

    - ``failure_rate``: the FINAL per-query outcome (approved vs not),
      broken down by skill / difficulty / kind (programming language).
    - ``plan_rejection`` / ``code_rejection``: EVERY individual judge vote
      across every attempt of every query, broken down by judge model /
      difficulty / skill / kind — a per-judge rejection rate, not a
      per-query one, so a query with 3 correction rounds contributes up to
      3x the votes of one approved on the first try.

    Each breakdown axis is independent (not a cross product): "by model"
    and "by skill" are two separate tables, not a model-x-skill matrix.
    """
    failure_by_skill: dict = {}
    failure_by_difficulty: dict = {}
    failure_by_kind: dict = {}

    plan_by_model: dict = {}
    plan_by_difficulty: dict = {}
    plan_by_skill: dict = {}
    plan_by_kind: dict = {}

    code_by_model: dict = {}
    code_by_difficulty: dict = {}
    code_by_skill: dict = {}
    code_by_kind: dict = {}

    for q in items:
        skill_keys = q.get("task_types") or ["plain"]
        difficulty = (q.get("difficulty") or "unknown").lower() or "unknown"
        kind = q.get("kind") or "unknown"
        query_approved = q.get("status") == "approved"

        for sk in skill_keys:
            _bump(failure_by_skill, sk, query_approved)
        _bump(failure_by_difficulty, difficulty, query_approved)
        _bump(failure_by_kind, kind, query_approved)

        for att in q.get("plan_attempts") or []:
            for vote in (att.get("panel") or {}).get("votes") or []:
                if vote.get("error"):
                    continue
                ok = bool(vote.get("approved"))
                _bump(plan_by_model, _judge_name(vote.get("judge")), ok)
                _bump(plan_by_difficulty, difficulty, ok)
                for sk in skill_keys:
                    _bump(plan_by_skill, sk, ok)
                _bump(plan_by_kind, kind, ok)

        for att in q.get("attempt_history") or []:
            for vote in (att.get("panel") or {}).get("votes") or []:
                if vote.get("error"):
                    continue
                ok = bool(vote.get("approved"))
                _bump(code_by_model, _judge_name(vote.get("judge")), ok)
                _bump(code_by_difficulty, difficulty, ok)
                for sk in skill_keys:
                    _bump(code_by_skill, sk, ok)
                _bump(code_by_kind, kind, ok)

    return {
        "totals": {"queries": len(items)},
        "failure_rate": {
            "by_skill": _finalize_rates(failure_by_skill),
            "by_difficulty": _finalize_rates(failure_by_difficulty),
            "by_kind": _finalize_rates(failure_by_kind),
        },
        "plan_rejection": {
            "by_model": _finalize_rates(plan_by_model),
            "by_difficulty": _finalize_rates(plan_by_difficulty),
            "by_skill": _finalize_rates(plan_by_skill),
            "by_kind": _finalize_rates(plan_by_kind),
        },
        "code_rejection": {
            "by_model": _finalize_rates(code_by_model),
            "by_difficulty": _finalize_rates(code_by_difficulty),
            "by_skill": _finalize_rates(code_by_skill),
            "by_kind": _finalize_rates(code_by_kind),
        },
    }


# ── Random walks ───────────────────────────────────────────────────────────────


def _normalize_walk_groups(data) -> list[dict]:
    """Flatten either shape ``final_generation_candidates(_semantic).json``
    has carried into one uniform ``{seed, operation_type, datasets}`` list.

    Older shape: ``{seed_id: [{"path_length": int, "paths": [{"operation_type",
    "datasets"}]}]}``. Current shape: a flat list of ``{"seed",
    "operation_type", "datasets", "steps"}``.
    """
    walks: list[dict] = []
    if isinstance(data, list):
        for w in data:
            if isinstance(w, dict) and "datasets" in w:
                walks.append(
                    {
                        "seed": w.get("seed", ""),
                        "operation_type": w.get("operation_type", []),
                        "datasets": w.get("datasets", []),
                    }
                )
    elif isinstance(data, dict):
        for seed_id, groups in data.items():
            for group in groups or []:
                for path in group.get("paths", []) or []:
                    walks.append(
                        {
                            "seed": seed_id,
                            "operation_type": path.get("operation_type", []),
                            "datasets": path.get("datasets", []),
                        }
                    )
    return walks


def _read_random_walks(cfg, limit: int = 300) -> dict:
    cd = cfg.candidates_discovery
    path: Path = cd.candidates_path
    if not path.exists():
        return {"available": False, "path": str(path), "walks": [], "total": 0}
    try:
        data = load_json(path)
    except Exception as exc:
        return {"available": False, "path": str(path), "error": str(exc), "walks": [], "total": 0}
    walks = _normalize_walk_groups(data)
    return {
        "available": True,
        "path": str(path),
        "total": len(walks),
        "walks": walks[:limit],
    }


# ── Clusters ─────────────────────────────────────────────────────────────────


def _dataset_name_lookup(cfg) -> dict[str, str]:
    """Best-effort dataset_id -> display name, from the crawled metadata.
    Missing/unreadable metadata just means ids are shown as-is."""
    path: Path = cfg.original_metadata_filepath
    if not path.exists():
        return {}
    try:
        data = load_json(path)
    except Exception:
        return {}
    names: dict[str, str] = {}
    for entry in data if isinstance(data, list) else []:
        resource = (entry or {}).get("resource") or {}
        rid = resource.get("id")
        name = resource.get("name")
        if rid and name:
            names[rid] = name
    return names


def _read_clusters(cfg) -> dict:
    """Cluster assignments + 2D projection for the Stats page's cluster map.

    Loads the persisted ``clusters.json`` when present (written by a live
    ``candidates-discovery-semantic`` run — see
    ``embedding_discovery.pipeline.pipeline``); otherwise computes it
    on-demand from the cached embeddings (deterministic given the same
    seed) and memoizes the result to that same path, so the expensive part
    — the embeddings themselves — is never recomputed, only the cheap
    KMeans+PCA step.
    """
    cd = cfg.candidates_discovery
    clusters_path: Optional[Path] = getattr(cd, "clusters_path", None)

    projection = None
    if clusters_path and clusters_path.exists():
        try:
            projection = load_json(clusters_path)
        except Exception:
            projection = None

    if projection is None:
        embeddings_path: Optional[Path] = getattr(cd, "embeddings_cache_path", None)
        if not embeddings_path or not embeddings_path.exists():
            return {"available": False}
        try:
            npz = np.load(embeddings_path, allow_pickle=True)
            ids = [str(i) for i in npz["ids"]]
            vectors = npz["vectors"]
            projection = compute_cluster_projection(
                ids,
                vectors,
                target_cluster_size=cd.target_cluster_size,
                overlap_margin=cd.cluster_overlap_margin,
                max_cluster_size=cd.max_cluster_size,
                seed=1,
            )
            if clusters_path:
                save_cluster_projection(clusters_path, projection)
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    names = _dataset_name_lookup(cfg)
    return {
        "available": True,
        "n_clusters": projection.get("n_clusters", 0),
        "points": [
            {
                "id": dataset_id,
                "name": names.get(dataset_id, dataset_id),
                "cluster": projection.get("assignments", {}).get(dataset_id, 0),
                "x": xy[0],
                "y": xy[1],
            }
            for dataset_id, xy in (projection.get("points") or {}).items()
        ],
        "clusters": {
            cid: [names.get(d, d) for d in members]
            for cid, members in (projection.get("clusters") or {}).items()
        },
    }


# ── Result serialization ──────────────────────────────────────────────────────


def _json_safe(v):
    """A cell value the browser's JSON parser will accept.

    Executed query results aren't always plain Python scalars: a
    ``number``-typed plan's aggregate is usually a numpy scalar (``.sum()``/
    ``.mean()`` return ``np.int64``/``np.float64``/``np.bool_``, none of
    which are instances of the Python types they resemble — a plain
    ``isinstance(v, (int, float, bool, str))`` check misses all three and
    used to fall through to ``str(v)``, silently turning a numeric result
    into a JSON string), and a ``list``-typed plan's per-group aggregation
    (e.g. ``df.groupby(...).apply(list)``) puts an actual Python
    list/ndarray in a DataFrame cell. ``pd.isna()`` on one of those returns
    an elementwise array rather than a single bool, so calling it inside an
    ``if`` used to crash the whole export with "the truth value of an array
    is ambiguous" instead of ever reaching the fallback.
    """
    if v is None:
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, (int, bool, str)):
        return v
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return [_json_safe(x) for x in list(v)]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (pd.Timestamp, datetime, date)):
        try:
            return None if pd.isna(v) else v.isoformat()
        except TypeError:
            return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _serialize_frame(df: pd.DataFrame) -> dict:
    total = len(df)
    shown = df.head(_MAX_RESULT_ROWS)
    return {
        "columns": [str(c) for c in shown.columns],
        "rows": [[_json_safe(v) for v in row] for row in shown.itertuples(index=False, name=None)],
        "total_rows": total,
        "shown_rows": len(shown),
    }


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="OrQA Query Browser")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/sources")
async def get_sources() -> list[dict]:
    result = []
    for e in _SOURCES.values():
        info = {"exists": False, "query_count": 0, "path": None}
        if e.cfg and not e.error:
            queries_path: Path = e.cfg.statement_generation.queries_path
            info["path"] = str(queries_path)
            if queries_path.exists():
                try:
                    data = load_json(queries_path)
                    info["query_count"] = sum(
                        1 for kind in data for _ in iter_questions(data, kind)
                    )
                    info["exists"] = info["query_count"] > 0
                except Exception:
                    pass
        result.append(
            {
                "id": e.id,
                "label": e.label,
                "type": e.type,
                "error": e.error,
                "statements": info,
            }
        )
    return result


@app.get("/api/queries/{source}")
async def get_queries(source: str) -> list[dict]:
    entry = _get_source_or_404(source)
    return _read_queries(entry.cfg)


@app.get("/api/stats/{source}")
def get_stats(source: str) -> dict:
    """Failure/rejection-rate breakdowns for the Stats page. Sync ``def``
    (threadpool) — aggregates over every query's full judge history, which
    for a large source is real work."""
    entry = _get_source_or_404(source)
    items = _read_queries(entry.cfg)
    return _compute_stats(items)


@app.get("/api/random-walks/{source}")
async def get_random_walks(source: str) -> dict:
    entry = _get_source_or_404(source)
    return _read_random_walks(entry.cfg)


@app.get("/api/clusters/{source}")
def get_clusters(source: str) -> dict:
    """Cluster map for the Stats page. Sync ``def`` (threadpool) — the
    first call for a source without a persisted ``clusters.json`` runs
    KMeans+PCA on demand."""
    entry = _get_source_or_404(source)
    return _read_clusters(entry.cfg)


@app.get("/api/keyword-search/{source}")
def keyword_search(
    source: str,
    keywords: list[str] = Query(...),
    num_tables: int = 1,
) -> dict:
    """Reverse-index search for the "Search Index" play button on a query's
    detail view: same ``DatasetIndex``/``ESDatasetIndex`` the plan judge's
    keyword-searchability check runs (see
    ``orqa.agent.utility.keyword_searchability``), exposed here so a human
    can re-run the exact same retrieval by hand for one query's
    ``question_keywords`` and see which tables actually come back.

    ``num_tables`` (the query's own table count, from the caller) drives the
    SAME adaptive top-K the gate itself used —
    ``round(num_tables * keyword_search_top_k_coefficient)`` — rather than a
    fixed K, so this manual re-run matches exactly what the plan judge saw
    for this query rather than a different, unrelated constant.

    Sync ``def`` (threadpool): the elasticsearch backend does network I/O.
    """
    entry = _get_source_or_404(source)
    index = _get_index(source, entry.cfg)
    top_k = max(
        1,
        round(max(1, num_tables) * entry.cfg.mcp_search.keyword_search_top_k_coefficient),
    )
    if index is None:
        return {"available": False, "results": [], "top_k": top_k}
    results = index.search(keywords, top_k=top_k)
    return {"available": True, "results": [r.to_dict() for r in results], "top_k": top_k}


@app.post("/api/execute/{source}/{kind}/{entry_key}/{qnum}")
def execute_query(source: str, kind: str, entry_key: str, qnum: str) -> dict:
    """Re-execute one stored query against the real datasets.

    Sync ``def`` on purpose: FastAPI runs it on the threadpool, so a long
    pandas execution never blocks the event loop. Read-only with
    respect to the queries file — nothing is written back.
    """
    entry = _get_source_or_404(source)
    cfg = entry.cfg
    queries_path: Path = cfg.statement_generation.queries_path
    if not queries_path.exists():
        raise HTTPException(status_code=404, detail=f"Queries file not found: {queries_path}")

    data = load_json(queries_path)
    query, meta = None, None
    for section, e_key, q_num, q, m in iter_questions(data, kind):
        if str(e_key) == str(entry_key) and str(q_num) == str(qnum):
            query, meta = q, m
            break
    if query is None:
        raise HTTPException(
            status_code=404, detail=f"Query {kind}/{entry_key}/{qnum} not found"
        )

    executor = QueryExecutor(cfg.datasets_path)
    try:
        frame = executor.execute({"tables": (meta or {}).get("tables", {})}, query, kind)
        if frame is None:
            return {"ok": False, "error": "Execution returned no result."}
        result = _serialize_frame(frame)
    except Exception as exc:
        # Covers both execution errors AND serialization errors (an
        # unexpected cell type _json_safe doesn't know how to handle) —
        # either way the UI gets a graceful "execution failed" message
        # instead of an unhandled 500 with no body the frontend can parse.
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=4),
        }
    return {"ok": True, "result": result}

