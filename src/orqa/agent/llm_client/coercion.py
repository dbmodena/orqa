"""Schema-driven coercion of an LLM's JSON into the shape a Pydantic model
expects — SHAPE only, never content.

Every structured answer in the pipeline is validated with ``model_validate``
and, on failure, re-requested. Most of those failures are shape slips a
program can undo without inventing anything: a list given as a comma string,
``table_reasons`` as an ``{alias: reason}`` mapping instead of a list of
objects, ``"slot": "1"``, a vote written as ``"yes"``, an ``op`` of
``"groupby"`` instead of ``group``, ``null`` for a field that has a default,
the whole answer wrapped in ``{"result": ...}``, camelCase keys. This module
fixes exactly those, logs each fix, and leaves everything else (a missing
vote, an unknown alias, an empty reason) to the model's retry.

Used by ``orqa.agent.llm_client.LLMClientStructured`` (every structured
call), ``orqa.agent.agents.JudgePanel`` (every vote) and
``orqa.agent.agents.QueryPlanner`` (every raw plan before assembly).
Measured in experiments/structured_stability (2026-11-02/03): on
gpt-oss-120b the shapes handled here were never needed on a clean answer —
they matter for the tail of a budget-cut answer and for weaker models.

Entry points:

- ``coerce_to_model(data, Model)`` — one proactive pass over the data,
  driven by the model's field annotations.
- ``apply_error_fixes(data, errors, Model)`` — a reactive pass driven by the
  ``ValidationError.errors()`` list (location + type), for what the
  proactive pass could not anticipate.
- ``validate_leniently(data, Model)`` — strict validation first, then the
  two passes above, up to a few rounds; returns the instance or the
  remaining errors.
- ``salvage_items(data, Model)`` — a set model keeps its valid items.
- ``unwrap_top(data, Model)`` — undo a wrapper the model's output does not have.
"""
from __future__ import annotations

import difflib
import json
import re
import types
import typing
from typing import Any, Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

TRUE_WORDS = {"true", "yes", "y", "1", "approved", "approve", "pass", "passed", "ok", "correct", "valid", "accept", "accepted"}
FALSE_WORDS = {"false", "no", "n", "0", "rejected", "reject", "fail", "failed", "incorrect", "invalid", "none", "null"}
NUMBER_WORDS = {w: i for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])}

# Synonyms for Literal-typed fields, keyed by FIELD NAME (normalized value ->
# canonical option). Only applied when the canonical option is among the
# field's own options, so the table is safe to share across models.
LITERAL_SYNONYMS: dict[str, dict[str, str]] = {
    "op": {
        "groupby": "group", "grouping": "group", "groupon": "group",
        "agg": "aggregate", "aggregation": "aggregate", "summarize": "aggregate", "summarise": "aggregate",
        "sum": "aggregate", "count": "aggregate", "mean": "aggregate", "average": "aggregate",
        "avg": "aggregate", "max": "aggregate", "min": "aggregate", "total": "aggregate",
        "where": "filter", "subset": "filter", "selectrows": "filter", "restrict": "filter",
        "orderby": "sort", "order": "sort", "arrange": "sort", "sortby": "sort",
        "top": "limit", "head": "limit", "topn": "limit", "first": "limit",
        "merge": "join", "innerjoin": "join", "leftjoin": "join", "lookup": "join",
        "concat": "union", "concatenate": "union", "append": "union", "stack": "union", "unionall": "union",
        "project": "select", "projection": "select", "pick": "select", "columns": "select", "distinct": "select",
        "compute": "derive", "calculate": "derive", "mutate": "derive", "transform": "derive",
        "derivecolumn": "derive", "flag": "derive", "bucket": "derive", "bin": "derive", "extract": "derive",
        "correlation": "correlate", "corr": "correlate", "pearson": "correlate", "spearman": "correlate",
        "ranking": "rank", "rankby": "rank",
        "cleaning": "clean", "cleanse": "clean", "cast": "clean", "impute": "clean", "dropna": "clean", "fillna": "clean",
    },
    "expected_result_type": {
        "dataframe": "table", "df": "table", "rows": "table", "tabular": "table", "records": "table", "matrix": "table",
        "integer": "number", "int": "number", "float": "number", "numeric": "number", "scalar": "number",
        "value": "number", "count": "number", "decimal": "number", "percentage": "number", "ratio": "number",
        "string": "text", "str": "text", "name": "text", "label": "text", "category": "text",
        "bool": "boolean", "yesno": "boolean", "flag": "boolean", "truefalse": "boolean",
        "array": "list", "series": "list", "sequence": "list", "values": "list", "column": "list",
    },
    "difficulty": {
        "simple": "easy", "basic": "easy", "low": "easy", "trivial": "easy",
        "moderate": "medium", "intermediate": "medium", "mid": "medium", "average": "medium",
        "difficult": "hard", "complex": "hard", "advanced": "hard", "high": "hard", "challenging": "hard",
    },
    "type": {
        "joincorrelation": "join-correlation", "correlation": "join-correlation", "correlate": "join-correlation",
        "merge": "join", "innerjoin": "join", "leftjoin": "join", "lookup": "join",
        "concat": "union", "append": "union", "stack": "union",
    },
    "kind": {"snapshot": "snapshot", "interval": "span", "range": "span", "period": "span", "feed": "ongoing", "live": "ongoing"},
}

# When a dict is handed where a list of objects is expected, the dict KEY is
# injected into the first of these fields the object model has (then into
# the first required string field).
KEY_FIELD_PREFERENCE = ("alias", "name", "table", "column", "key", "id", "slot")


# ---------------------------------------------------------------- helpers
def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _is_model(ann: Any) -> bool:
    return isinstance(ann, type) and issubclass(ann, BaseModel)


def _unwrap_optional(ann: Any) -> tuple[Any, bool]:
    """``Optional[X]`` -> ``(X, True)``; a Union keeps its non-None arms."""
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):
        arms = [a for a in get_args(ann) if a is not type(None)]
        if len(arms) == 1:
            return arms[0], True
        return Union[tuple(arms)], len(arms) < len(get_args(ann))  # type: ignore[return-value]
    return ann, False


def _kind(ann: Any) -> str:
    """Coarse kind of an annotation: model | list | dict | literal | str |
    bool | int | float | union | any."""
    ann, _ = _unwrap_optional(ann)
    origin = get_origin(ann)
    if origin is Literal:
        return "literal"
    if origin in (Union, types.UnionType):
        return "union"
    if _is_model(ann):
        return "model"
    if ann is list or origin in (list, typing.List, tuple, set):
        return "list"
    if ann is dict or origin in (dict, typing.Dict):
        return "dict"
    if ann is str:
        return "str"
    if ann is bool:
        return "bool"
    if ann is int:
        return "int"
    if ann is float:
        return "float"
    return "any"


def _item_ann(ann: Any) -> Any:
    ann, _ = _unwrap_optional(ann)
    args = get_args(ann)
    return args[0] if args else Any


def _dict_anns(ann: Any) -> tuple[Any, Any]:
    ann, _ = _unwrap_optional(ann)
    args = get_args(ann)
    return (args[0], args[1]) if len(args) == 2 else (Any, Any)


def _field_names(model: type[BaseModel]) -> dict[str, str]:
    """``{candidate key: field name}`` over names and aliases."""
    out = {}
    for name, info in model.model_fields.items():
        out[name] = name
        if info.alias:
            out[info.alias] = name
    return out


def _match_key(key: str, model: type[BaseModel]) -> Optional[str]:
    names = _field_names(model)
    if key in names:
        return names[key]
    lowered = {k.lower(): v for k, v in names.items()}
    if str(key).lower() in lowered:
        return lowered[str(key).lower()]
    nk = _norm(key)
    if not nk:
        return None
    normed: dict[str, str] = {}
    for k, v in names.items():
        normed.setdefault(_norm(k), v)
    if nk in normed:
        return normed[nk]
    # singular / plural
    for cand in (nk + "s", nk[:-1] if nk.endswith("s") else None):
        if cand and cand in normed:
            return normed[cand]
    # an unambiguous suffix ("keywords" -> "table_keywords" when it is the only one)
    if len(nk) >= 5:
        hits = {v for k, v in normed.items() if k.endswith(nk) or k.endswith(nk + "s")}
        if len(hits) == 1:
            return hits.pop()
    return None


def _single_list_field(model: type[BaseModel]) -> Optional[tuple[str, Any]]:
    """``(field name, item annotation)`` when the model has exactly one
    list-typed field (a QuestionDraftSet / QuerySet / PlanSet shape)."""
    lists = [(n, _item_ann(i.annotation)) for n, i in model.model_fields.items() if _kind(i.annotation) == "list"]
    return lists[0] if len(lists) == 1 else None


def _required_str_fields(model: type[BaseModel]) -> list[str]:
    return [n for n, i in model.model_fields.items() if _kind(i.annotation) == "str" and i.is_required()]


def _str_fields(model: type[BaseModel]) -> list[str]:
    return [n for n, i in model.model_fields.items() if _kind(i.annotation) == "str"]


def _default_for(ann: Any) -> Any:
    kind = _kind(ann)
    return {"list": [], "dict": {}, "str": ""}.get(kind, None)


# ------------------------------------------------------------ unwrapping
def unwrap_top(data: Any, model: type[BaseModel], log: Optional[list] = None) -> Any:
    """Undo a wrapper the model's output does not have: ``{"result": {...}}``,
    ``{"QuestionDraftSet": {...}}``, a bare list where the model has one list
    field, or a single item where a list of items is expected."""
    if isinstance(data, dict):
        matched = [k for k in data if _match_key(k, model)]
        if matched:
            return data
        if len(data) == 1:
            (key, value), = data.items()
            if isinstance(value, (dict, list)):
                if log is not None:
                    log.append(f"$: unwrapped {key!r}")
                return unwrap_top(value, model, log)
        single = _single_list_field(model)
        if single and _is_model(single[1]) and any(_match_key(k, single[1]) for k in data):
            if log is not None:
                log.append(f"$: wrapped a single item into {single[0]!r}")
            return {single[0]: [data]}
        return data
    if isinstance(data, list):
        single = _single_list_field(model)
        if single:
            if log is not None:
                log.append(f"$: wrapped a bare list into {single[0]!r}")
            return {single[0]: data}
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except (TypeError, ValueError):
            return data
        if isinstance(parsed, (dict, list)):
            if log is not None:
                log.append("$: parsed a JSON string")
            return unwrap_top(parsed, model, log)
    return data


# ------------------------------------------------------------- scalars
def coerce_str(value: Any) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        parts = [coerce_str(v) for v in value]
        return "; ".join(p for p in parts if p)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def coerce_bool(value: Any) -> Optional[bool]:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower().rstrip(".!")
        if text in TRUE_WORDS:
            return True
        if text in FALSE_WORDS:
            return False
        head = re.split(r"[\s,;:(\-–—]+", text, maxsplit=1)[0]
        if head in TRUE_WORDS:
            return True
        if head in FALSE_WORDS:
            return False
        return None
    if isinstance(value, dict):
        for key in ("approved", "value", "vote", "result"):
            if key in value:
                return coerce_bool(value[key])
    return None


def coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return int(value) if isinstance(value, bool) else None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in NUMBER_WORDS:
            return NUMBER_WORDS[text]
        m = re.search(r"-?\d+", text)
        if m:
            return int(m.group())
        for word, n in NUMBER_WORDS.items():
            if re.search(rf"\b{word}\b", text):
                return n
    return None


def coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if m:
            return float(m.group())
    return None


def coerce_literal(value: Any, options: tuple, field_name: str = "") -> Any:
    if value in options:
        return value
    if not isinstance(value, str):
        if all(isinstance(o, bool) for o in options):
            b = coerce_bool(value)
            return b if b in options else value
        if all(isinstance(o, int) for o in options):
            i = coerce_int(value)
            return i if i in options else value
        return value
    text = value.strip()
    by_norm = {_norm(o): o for o in options if isinstance(o, str)}
    nv = _norm(text)
    if nv in by_norm:
        return by_norm[nv]
    synonyms = LITERAL_SYNONYMS.get(field_name, {})
    if nv in synonyms and synonyms[nv] in options:
        return synonyms[nv]
    # a compound op ("groupby_sum", "aggregate_count"): the longest option it starts with or contains
    containing = sorted((o for n, o in by_norm.items() if n and (nv.startswith(n) or n in nv)), key=len, reverse=True)
    if containing:
        return containing[0]
    # a compound built from synonyms ("groupby_sum" -> group)
    for syn, canonical in synonyms.items():
        if canonical in options and nv.startswith(syn):
            return canonical
    close = difflib.get_close_matches(nv, list(by_norm), n=1, cutoff=0.75)
    if close:
        return by_norm[close[0]]
    return value


# ----------------------------------------------------------- containers
def coerce_list(value: Any, item_ann: Any, field_name: str, path: str, log: list) -> Any:
    if value is None:
        return None
    item_kind = _kind(item_ann)
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
                log.append(f"{path}: parsed a JSON string as a list")
                return coerce_list(parsed, item_ann, field_name, path, log)
            except ValueError:
                pass
        if item_kind in ("str", "any", "literal", "int", "float"):
            parts = [p.strip(" \t-•*") for p in re.split(r"[\n;,]+", text)] if re.search(r"[\n;,]", text) else [text]
            items = [p for p in parts if p]
            log.append(f"{path}: split a string into {len(items)} item(s)")
        else:
            items = [text]
            log.append(f"{path}: wrapped a string into a list")
    elif isinstance(value, dict):
        items = _dict_to_items(value, item_ann, path, log)
    else:
        items = [value]
        log.append(f"{path}: wrapped a scalar into a list")
    out = []
    position_field = _position_field(item_ann) if item_kind == "model" else None
    for i, item in enumerate(items):
        if position_field and isinstance(item, dict) and position_field not in item:
            item = {**item, position_field: i + 1}
            log.append(f"{path}[{i}].{position_field}: filled positional {i + 1}")
        out.append(coerce_value(item, item_ann, field_name=field_name, path=f"{path}[{i}]", log=log))
    return out


POSITION_FIELDS = ("order", "index", "position")


def _position_field(item_ann: Any) -> Optional[str]:
    """The required int field of a list item that only records its position
    (``order`` on a plan step) — filled from the position when missing."""
    model = _unwrap_optional(item_ann)[0]
    if not _is_model(model):
        return None
    for name in POSITION_FIELDS:
        info = model.model_fields.get(name)
        if info is not None and info.is_required() and _kind(info.annotation) == "int":
            return name
    return None


def _dict_to_items(value: dict, item_ann: Any, path: str, log: list) -> list:
    """A mapping handed where a list is expected."""
    item_kind = _kind(item_ann)
    if item_kind == "model":
        model = _unwrap_optional(item_ann)[0]
        key_field = next((f for f in KEY_FIELD_PREFERENCE if f in model.model_fields), None)
        req = _required_str_fields(model) or _str_fields(model)
        key_field = key_field or (req[0] if req else None)
        items = []
        for k, v in value.items():
            if isinstance(v, dict):
                item = dict(v)
                if key_field and key_field not in item:
                    item[key_field] = k
            elif isinstance(v, str):
                others = [f for f in req if f != key_field]
                item = {key_field: k, others[0]: v} if key_field and others else ({key_field or "value": v} if key_field else {"value": v})
            elif isinstance(v, (list, tuple)):
                list_fields = [n for n, i in model.model_fields.items() if _kind(i.annotation) == "list"]
                item = {key_field: k} if key_field else {}
                if list_fields:
                    item[list_fields[0]] = list(v)
            else:
                item = {key_field: k, "value": v} if key_field else {"value": v}
            items.append(item)
        log.append(f"{path}: turned a mapping into {len(items)} object(s), key -> {key_field!r}")
        return items
    if item_kind == "dict":
        # {"Table_0": ["a", "b"], "Table_1": ["c", "d"]} -> one entry per aligned pair
        if value and all(isinstance(v, (list, tuple)) for v in value.values()):
            n = max(len(v) for v in value.values())
            items = [{k: (v[i] if i < len(v) else None) for k, v in value.items()} for i in range(n)]
            items = [{k: v for k, v in it.items() if v is not None} for it in items]
            log.append(f"{path}: zipped a mapping of lists into {len(items)} entries")
            return items
        log.append(f"{path}: wrapped a mapping into a one-entry list")
        return [value]
    # list of strings: the keys (a {"term": weight} style) unless the values are the strings
    if value and all(isinstance(v, str) for v in value.values()) and not all(isinstance(k, str) and k for k in value):
        log.append(f"{path}: took the values of a mapping as the list")
        return list(value.values())
    log.append(f"{path}: took the keys of a mapping as the list")
    return list(value.keys())


def coerce_dict(value: Any, key_ann: Any, val_ann: Any, field_name: str, path: str, log: list) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            cv = coerce_value(v, val_ann, field_name=field_name, path=f"{path}.{k}", log=log) if val_ann is not Any else v
            out[str(k)] = cv
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        if text[0] == "{":
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    log.append(f"{path}: parsed a JSON string as an object")
                    return coerce_dict(parsed, key_ann, val_ann, field_name, path, log)
            except ValueError:
                pass
        pairs = [re.split(r"\s*[:=]\s*", p, maxsplit=1) for p in re.split(r"[\n;,]+", text) if p.strip()]
        if pairs and all(len(p) == 2 for p in pairs):
            log.append(f"{path}: parsed 'key: value' text into an object")
            return {k.strip(): v.strip() for k, v in pairs}
        log.append(f"{path}: wrapped a string into an object")
        return {"value": text}
    if isinstance(value, (list, tuple)):
        items = list(value)
        if not items:
            return {}
        if all(isinstance(it, dict) for it in items):
            merged: dict = {}
            for it in items:
                keys = {k.lower(): k for k in it}
                kk = next((keys[a] for a in ("key", "name", "table", "table_alias", "alias", "column_name") if a in keys), None)
                vk = next((keys[a] for a in ("value", "column", "columns", "col", "role") if a in keys), None)
                if kk and vk and len(it) <= 3:
                    merged[str(it[kk])] = it[vk]
                else:
                    merged.update(it)
            log.append(f"{path}: merged a list of objects into one object")
            return coerce_dict(merged, key_ann, val_ann, field_name, path, log)
        if all(isinstance(it, (list, tuple)) and len(it) == 2 for it in items):
            log.append(f"{path}: turned pairs into an object")
            return {str(k): v for k, v in items}
        if all(isinstance(it, str) for it in items):
            split = [re.split(r"\s*[:=.]\s*", it, maxsplit=1) for it in items]
            if all(len(s) == 2 for s in split):
                log.append(f"{path}: parsed 'a.b' strings into an object")
                return {k: v for k, v in split}
        log.append(f"{path}: kept a list under 'values' in an object")
        return {"values": items}
    log.append(f"{path}: wrapped a scalar into an object")
    return {"value": value}


# -------------------------------------------------------------- dispatch
def coerce_value(value: Any, ann: Any, *, field_name: str = "", path: str = "$", log: Optional[list] = None) -> Any:
    log = log if log is not None else []
    if ann is Any or ann is None:
        return value
    ann, optional = _unwrap_optional(ann)
    if value is None:
        return None
    kind = _kind(ann)
    before = value
    if kind == "union":
        arms = list(get_args(ann))
        by_kind = {_kind(a): a for a in arms}
        want = "model" if isinstance(value, dict) else "list" if isinstance(value, list) else "str" if isinstance(value, str) else None
        arm = by_kind.get(want) if want else None
        if arm is None:
            for a in arms:
                if _kind(a) == "model" and isinstance(value, dict):
                    arm = a
                    break
        return coerce_value(value, arm, field_name=field_name, path=path, log=log) if arm is not None else value
    if kind == "literal":
        out = coerce_literal(value, get_args(ann), field_name)
    elif kind == "model":
        if isinstance(value, dict):
            return coerce_model(value, ann, path=path, log=log)
        if isinstance(value, str) and value.strip().startswith("{"):
            try:
                parsed = json.loads(value)
            except ValueError:
                return value
            log.append(f"{path}: parsed a JSON string as an object")
            return coerce_model(parsed, ann, path=path, log=log) if isinstance(parsed, dict) else value
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            log.append(f"{path}: unwrapped a one-item list into an object")
            return coerce_model(value[0], ann, path=path, log=log)
        return value
    elif kind == "list":
        return coerce_list(value, _item_ann(ann), field_name, path, log)
    elif kind == "dict":
        k_ann, v_ann = _dict_anns(ann)
        return coerce_dict(value, k_ann, v_ann, field_name, path, log)
    elif kind == "str":
        out = coerce_str(value)
    elif kind == "bool":
        out = coerce_bool(value)
        if out is None:
            return value
    elif kind == "int":
        out = coerce_int(value)
        if out is None:
            return value
    elif kind == "float":
        out = coerce_float(value)
        if out is None:
            return value
    else:
        return value
    if out != before or type(out) is not type(before):
        log.append(f"{path}: {kind} <- {type(before).__name__} {_short(before)} -> {_short(out)}")
    return out


def _short(value: Any, n: int = 60) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return text if len(text) <= n else text[: n - 1] + "…"


def coerce_model(data: Any, model: type[BaseModel], *, path: str = "$", log: Optional[list] = None) -> Any:
    """Coerce one object: keys matched to fields (case, separators,
    singular/plural), each value coerced to its field's annotation, ``null``
    dropped where a default exists, missing required containers filled."""
    log = log if log is not None else []
    if not isinstance(data, dict):
        return data
    out: dict = {}
    for key, value in data.items():
        field = _match_key(str(key), model)
        if field is None:
            out[key] = value  # an extra key: ignored by the model, kept for transparency
            continue
        if field in out:
            continue
        if field != key:
            log.append(f"{path}.{key}: key -> {field!r}")
        info = model.model_fields[field]
        cv = coerce_value(value, info.annotation, field_name=field, path=f"{path}.{field}", log=log)
        if cv is None:
            if not info.is_required():
                log.append(f"{path}.{field}: dropped null (default applies)")
                continue
            filled = _default_for(info.annotation)
            if filled is not None:
                log.append(f"{path}.{field}: null -> {filled!r} (required)")
                cv = filled
        out[field] = cv
    for field, info in model.model_fields.items():
        if field not in out and info.is_required() and _kind(info.annotation) in ("list", "dict"):
            out[field] = _default_for(info.annotation)
            log.append(f"{path}.{field}: missing required container -> {out[field]!r}")
    return out


def coerce_to_model(data: Any, model: type[BaseModel], *, log: Optional[list] = None) -> Any:
    log = log if log is not None else []
    data = unwrap_top(data, model, log)
    return coerce_model(data, model, log=log) if isinstance(data, dict) else data


# ---------------------------------------------------- error-driven pass
def _annotation_at(model: type[BaseModel], loc: tuple) -> Any:
    ann: Any = model
    for part in loc:
        ann, _ = _unwrap_optional(ann)
        kind = _kind(ann)
        if kind == "model":
            info = ann.model_fields.get(part) if isinstance(part, str) else None
            if info is None:
                return None
            ann = info.annotation
        elif kind == "list":
            ann = _item_ann(ann)
        elif kind == "dict":
            ann = _dict_anns(ann)[1]
        elif kind == "union":
            arms = [a for a in get_args(ann) if _is_model(a) and getattr(a, "__name__", "") == part]
            if not arms:
                return None
            ann = arms[0]
        else:
            return None
    return ann


def _container_at(data: Any, loc: tuple) -> tuple[Any, Any]:
    """``(container, key)`` for the error location, or ``(None, None)``."""
    cur = data
    for part in loc[:-1]:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and isinstance(part, int) and 0 <= part < len(cur):
            cur = cur[part]
        else:
            return None, None
    return cur, (loc[-1] if loc else None)


def _expected_options(ctx: dict) -> tuple:
    expected = ctx.get("expected")
    if isinstance(expected, str):
        return tuple(re.findall(r"'([^']*)'", expected))
    return tuple()


def apply_error_fixes(data: Any, errors: list[dict], model: type[BaseModel], log: Optional[list] = None) -> bool:
    """Fix what the validation errors name, in place. Returns True when
    anything changed. Content is never invented: a missing vote, a missing
    sentence or an unknown option stays missing."""
    log = log if log is not None else []
    changed = False
    for err in errors:
        loc = tuple(err.get("loc") or ())
        etype = err.get("type", "")
        if not loc:
            continue
        container, key = _container_at(data, loc)
        if container is None:
            continue
        has = (isinstance(container, dict) and key in container) or (isinstance(container, list) and isinstance(key, int) and key < len(container))
        value = container[key] if has else None
        ann = _annotation_at(model, loc)
        path = "$." + ".".join(str(p) for p in loc)
        new: Any = value
        delete = False
        if etype == "missing":
            # A missing REQUIRED field is content, not shape: a question, a
            # verdict's check text or a step's columns are never filled in
            # here (the repair call or the re-request asks the model for
            # them). The one exception is a positional index (`order`,
            # `slot`) the item's place in its list already states.
            if ann is None:
                continue
            kind = _kind(ann)
            if kind == "int" and key in POSITION_FIELDS and len(loc) >= 2 and isinstance(loc[-2], int):
                new = loc[-2] + 1
            else:
                continue
        elif etype == "literal_error":
            options = _expected_options(err.get("ctx") or {})
            new = coerce_literal(value, options, loc[-1] if isinstance(loc[-1], str) else "")
        elif etype in ("list_type",):
            new = coerce_list(value, _item_ann(ann) if ann is not None else Any, str(key), path, log)
            if not isinstance(new, list):
                new = [value]
        elif etype in ("dict_type",):
            k_ann, v_ann = _dict_anns(ann) if ann is not None else (Any, Any)
            new = coerce_dict(value, k_ann, v_ann, str(key), path, log)
            if not isinstance(new, dict):
                new = {}
        elif etype in ("string_type",):
            new = coerce_str(value)
            if new is None:
                new = ""
        elif etype in ("bool_type", "bool_parsing"):
            b = coerce_bool(value)
            if b is None:
                field_info = None
                parent_ann = _annotation_at(model, loc[:-1])
                if parent_ann is not None and _kind(parent_ann) == "model":
                    field_info = _unwrap_optional(parent_ann)[0].model_fields.get(key)
                if field_info is not None and not field_info.is_required():
                    delete = True
                else:
                    continue
            new = b
        elif etype in ("int_type", "int_parsing", "int_from_float", "finite_number"):
            i = coerce_int(value)
            if i is None:
                continue
            new = i
        elif etype in ("float_type", "float_parsing"):
            f = coerce_float(value)
            if f is None:
                continue
            new = f
        elif etype in ("model_type", "model_attributes_type", "dataclass_type"):
            if isinstance(value, str) and value.strip().startswith("{"):
                try:
                    new = json.loads(value)
                except ValueError:
                    continue
            elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
                new = value[0]
            elif isinstance(value, str) and ann is not None and _kind(ann) == "model":
                req = _required_str_fields(_unwrap_optional(ann)[0])
                if len(req) >= 2 and ":" in value:
                    a, b = value.split(":", 1)
                    new = {req[0]: a.strip(), req[1]: b.strip()}
                elif len(req) == 1:
                    new = {req[0]: value}
                else:
                    continue
            else:
                continue
        elif etype == "extra_forbidden":
            delete = True
        elif etype in ("none_required",):
            continue
        else:
            continue
        if delete:
            if isinstance(container, dict) and key in container:
                del container[key]
                log.append(f"{path}: removed ({etype})")
                changed = True
            continue
        if new is value and has and type(new) is type(value):
            continue
        if isinstance(container, dict):
            container[key] = new
        elif isinstance(container, list) and isinstance(key, int):
            container[key] = new
        else:
            continue
        log.append(f"{path}: {etype} -> {_short(new)}")
        changed = True
    return changed


def validate_leniently(data: Any, model: type[BaseModel], *, max_passes: int = 3) -> tuple[Optional[BaseModel], Any, list[str], list[dict]]:
    """Strict validation, then coercion and error-driven fixes.

    Returns ``(instance or None, data as last coerced, fixes applied,
    remaining errors)``."""
    log: list[str] = []
    strict: Optional[BaseModel] = None
    try:
        strict = model.model_validate(data)
    except ValidationError:
        pass
    if strict is not None and not validated_empty(strict, data, model):
        return strict, data, log, []
    current = coerce_to_model(json.loads(json.dumps(data, default=str)), model, log=log)
    if strict is not None:
        # the strict reading is an empty set: prefer the coerced one only when it recovers items
        try:
            recovered = model.model_validate(current)
        except ValidationError:
            return strict, data, log, []
        single = _single_list_field(model)
        if single and getattr(recovered, single[0], None):
            log.append(f"$: strict validation gave an empty {single[0]!r}; kept the coerced reading")
            return recovered, current, log, []
        return strict, data, log, []
    errors: list[dict] = []
    for _ in range(max_passes):
        try:
            return model.model_validate(current), current, log, []
        except ValidationError as exc:
            errors = exc.errors(include_url=False)
            if not apply_error_fixes(current, errors, model, log):
                return None, current, log, errors
    try:
        return model.model_validate(current), current, log, []
    except ValidationError as exc:
        return None, current, log, exc.errors(include_url=False)


def validated_empty(instance: BaseModel, data: Any, model: type[BaseModel]) -> bool:
    """True when a set model validated only because its one list field
    defaulted to empty while the raw data carried unmatched keys (a wrapped
    or misnamed answer) or was not an object at all."""
    single = _single_list_field(model)
    if not single:
        return False
    if getattr(instance, single[0], None):
        return False
    if not isinstance(data, dict):
        return True
    return not any(_match_key(k, model) for k in data)


def errors_brief(errors: list[dict]) -> list[str]:
    return [f"{' -> '.join(str(x) for x in e.get('loc', ()))}: {e.get('msg')} ({e.get('type')})" for e in errors]


def salvage_items(data: Any, model: type[BaseModel]) -> tuple[Optional[BaseModel], int, int, list[str]]:
    """For a set model (one list field of objects): validate each item on
    its own, keep the valid ones. Returns ``(instance or None, kept,
    dropped, fixes)``."""
    single = _single_list_field(model)
    if not single or not isinstance(data, dict):
        return None, 0, 0, []
    name, item_ann = single
    item_model = _unwrap_optional(item_ann)[0]
    items = data.get(name)
    if not isinstance(items, list) or not _is_model(item_model):
        return None, 0, 0, []
    kept, log = [], []
    for i, item in enumerate(items):
        inst, _, fixes, _ = validate_leniently(item, item_model)
        if inst is not None:
            kept.append(inst.model_dump())
            log.extend(f"{name}[{i}]{f[1:]}" if f.startswith("$") else f for f in fixes)
    if not kept:
        return None, 0, len(items), log
    rest = {k: v for k, v in data.items() if k != name}
    try:
        return model.model_validate({**rest, name: kept}), len(kept), len(items) - len(kept), log
    except ValidationError:
        return None, len(kept), len(items) - len(kept), log
