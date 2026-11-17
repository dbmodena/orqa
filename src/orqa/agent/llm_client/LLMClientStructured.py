"""Structured-output LLM client: Pydantic response models, JSON repair, shape
coercion and a BUDGET-AWARE retry loop shared by every structured call.

What the loop does on every attempt (measured in
``experiments/structured_stability``, 2026-11-02/03, on OCI gpt-oss-120b):

* **An explicit output budget on every call.** The provider's silent default
  is 4096 completion tokens INCLUDING a reasoning model's hidden thinking,
  and OCI reports ``finish_reason: "stop"`` even when the budget ran out
  mid-thought — so the symptom of an exhausted budget is an empty answer, or
  JSON whose tail is missing, never a "length" stop. That was the whole
  population of the "Pydantic errors" of the question and plan stages
  (15/15 non-valid calls over 292, every one at its token cap). Every call
  therefore carries ``max_tokens`` (``default_max_tokens`` when the caller
  passes none), and exhaustion is read from the token count
  (:func:`budget_exhausted`): a cut answer is retried with twice the
  budget, up to ``budget_ceiling``, never by re-sending the same prompt with
  a "fix your JSON" note.
* **Shape coercion before failing** (:mod:`~orqa.agent.llm_client.coercion`):
  the missing tail fields of a cut answer and the shape slips of a weaker
  model validate without a second call; content is never invented.
* **One short repair call** before the full re-prompt: the model is shown
  its own JSON with the exact errors (a few hundred tokens instead of the
  whole prompt again).
* **Native JSON-schema decoding** (``response_format: json_schema``) only
  where a subclass turns it on — short outputs: the question generator and
  the judges. On long outputs (plans, code) the constrained decoder fell
  into repeated-sentence loops running to the cap, so it stays off there,
  and a **repetition guard** (:func:`degenerate_repetition`) rejects such an
  answer on the raw text whatever mode produced it.

Per-client switches are class attributes (see :class:`LLMClientStructured`),
so a subclass declares its policy in one line.
"""
import importlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, ValidationError

from .LLMClient import LLMClient
from ...utils.generation_trace import trace_llm_call
from orqa.agent.llm_client.messages import sanitize_messages
from orqa.agent.llm_client.coercion import errors_brief, salvage_items, validate_leniently, validated_empty

logger = logging.getLogger(__name__)

# How much of a failed response is quoted back in a retry's error message.
# The model needs to SEE what it produced to repair it, but the raw output can
# run to the full completion cap, so it is quoted as a bounded excerpt rather
# than echoed whole as an extra `assistant` turn.
_FAILED_OUTPUT_EXCERPT_CHARS = 1200

# Output budget of a structured call that passes no max_tokens of its own.
# Validity over tokens (the user's call, 2026-11-03): large enough that a
# reasoning model's thinking plus the visible JSON never hits it on the
# payloads seen so far (question generator up to ~5k tokens, judges up to
# ~4k, planner batches up to ~11k, single-plan corrections up to ~17k).
DEFAULT_MAX_TOKENS = 32000
# Where budget escalation stops: the most OCI accepts for gpt-oss-120b
# (probed live).
BUDGET_CEILING = 131072
# completion_tokens at (or within a hair of) max_tokens means the visible
# answer was cut, whatever finish_reason says.
BUDGET_EXHAUSTED_RATIO = 0.98
# The repair call shows the model its own (short) JSON; it still needs room
# to think.
REPAIR_MAX_TOKENS_FLOOR = 4096


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def budget_exhausted(finish_reason: Any, usage: dict, max_tokens: Optional[int], content: str) -> bool:
    """Whether the answer was cut by the output budget.

    ``finish_reason == "length"`` is the honest signal; OCI does not send it
    for reasoning models, so the token count decides: ``completion_tokens``
    at the cap, or an empty answer under a cap (Gemini reports its thinking
    outside ``completion_tokens``: an empty answer with 0 visible tokens is
    the cap swallowed by thought).
    """
    if finish_reason == "length":
        return True
    if not max_tokens:
        return False
    completion = int((usage or {}).get("completion_tokens") or 0)
    if completion >= int(max_tokens) * BUDGET_EXHAUSTED_RATIO:
        return True
    if not (content or "").strip():
        return True
    return False


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\\n+|\n+")
_JSON_LINE = re.compile(r'^\s*[\[\]{},]*\s*"[^"]*"\s*:')


def degenerate_repetition(content: str, min_repeats: int = 6, min_len: int = 12) -> Optional[str]:
    """A sentence or line repeated many times in a row — the looping a
    constrained decoder can fall into and run to the token cap with. Such an
    answer may still parse and validate (the loop sits inside one string
    field, and a batch comes back with one plan instead of three), so it is
    caught on the raw text before validation. Returns the repeated unit, or
    ``None`` for a clean answer (0 false positives on 290 recorded clean
    answers)."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(content or "") if p and p.strip()]
    run, previous = 1, None
    for part in parts:
        if part == previous and len(part) >= min_len:
            run += 1
            if run >= min_repeats:
                return part[:120]
        else:
            run, previous = 1, part
    # two or three sentences alternating (A B A B ...) never repeat consecutively.
    # PROSE sentences only: a JSON `"key": value` line legitimately recurs (one
    # per step), and so does a line of code — a 22-step plan's SQL repeats
    # `SELECT "Expense Type", "Service Area",` once per CTE (a false positive
    # seen live, 2026-11-03), so a unit must end like a sentence and recur
    # far more often than any plan has steps.
    counts: dict[str, int] = {}
    for part in parts:
        if (len(part) >= 20 and len(part.split()) >= 4 and part[-1] in ".!?"
                and not _JSON_LINE.match(part)):
            counts[part] = counts.get(part, 0) + 1
    worst = max(counts.items(), key=lambda kv: kv[1], default=(None, 0))
    if worst[1] >= max(min_repeats, 12) and worst[0] is not None:
        return worst[0][:120]
    return None


def inline_refs(schema: dict) -> dict:
    """The model's JSON schema with every ``$ref`` replaced by the referenced
    definition and ``$defs`` removed. Pydantic emits ``$defs``/``$ref`` for
    nested models; OCI's gpt-oss deployments take them, Gemini's
    ``response_schema`` rejects the payload ("Unknown name $defs"). None of
    the pipeline's models is recursive, so inlining always terminates."""
    defs = schema.get("$defs") or {}

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 50:
            return node
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str) and node["$ref"].startswith("#/$defs/"):
                target = defs.get(node["$ref"].split("/")[-1], {})
                merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
                return resolve(merged, depth + 1)
            return {k: resolve(v, depth + 1) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v, depth + 1) for v in node]
        return node

    return resolve(schema)


def served_by_fallback(response: Any, primary_model: str, primary_id: Optional[str] = None) -> bool:
    """True when the Router answered from a FALLBACK deployment rather than
    the configured primary — a different, weaker model answering a
    structured call without anything else in the pipeline saying so.

    Decided on the Router's deployment id when the response carries one
    (``_hidden_params["model_id"]``, set from the ``model_info.id`` the
    client gives every Router entry), since the model name a provider
    echoes back need not be the configured string — OCI answers a dedicated
    endpoint with the base model's OCID. Without an id, the names are
    compared as before.
    """
    hidden = getattr(response, "_hidden_params", None)
    served_id = hidden.get("model_id") if isinstance(hidden, dict) else None
    if primary_id and served_id:
        return str(served_id) != str(primary_id)
    try:
        served = str(response.get("model") or "")
    except Exception:  # noqa: BLE001 — a response object without .get
        return False
    if not served:
        return False
    primary = primary_model.split("/", 1)[-1]
    return served.split("/", 1)[-1] != primary and primary not in served


def _looks_like_format_rejection(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(s in text for s in (
        "response_format", "responseformat", "json_schema", "jsonschema", "response_schema",
        "structured output", "unsupported param", "$defs", "invalid json payload",
    ))


def _needed_structural_repair(cleaned: str) -> bool:
    """True when the text only parses after the repair pipeline closed
    delimiters or strings for it — the signature of an answer cut by the
    output budget. Such an answer can still validate (the cut field
    defaults, the trailing items are gone), so it must not pass as valid when
    the budget was exhausted."""
    try:
        json.loads(cleaned)
        return False
    except json.JSONDecodeError:
        return True


def _usage_of(response: Any) -> dict:
    usage = response["usage"] or {}
    return {k: int(usage.get(k, 0) or 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}


def _add_usage(total: dict, part: dict) -> None:
    for k in total:
        total[k] += int((part or {}).get(k, 0) or 0)


def _whitespace_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ch in " \n\t\r") / len(text)



def _reasoning_of(message: Any) -> Optional[str]:
    """The reasoning text a provider returned beside the answer
    (``reasoning_content``, as litellm normalizes it, or ``reasoning``), or
    ``None``. Kept in the call record for the generation trace."""
    for key in ("reasoning_content", "reasoning"):
        try:
            value = message.get(key) if hasattr(message, "get") else getattr(message, key, None)
        except Exception:  # noqa: BLE001 — a record field, never a reason to fail
            value = None
        if isinstance(value, str) and value.strip():
            return value
    return None

class LLMClientStructured(LLMClient):
    """
    LiteLLM client with YAML configuration and structured output support.

    Adds on top of LLMClient:
    - Pydantic response-model loading and validation
    - A robust JSON repair pipeline shared by all subclasses
    - The budget-aware retry loop described in the module docstring

    Per-client policy (class attributes, overridden by subclasses):

    ``preamble``
        First line of the system message (the schema follows it).
    ``native_schema``
        Ask the provider to constrain decoding to the response model's JSON
        schema. ON for short outputs (question generator, judges), OFF for
        long ones (plans, code) where the constrained decoder loops. A
        provider that rejects the parameter is detected on the first error
        and the client falls back to prompt-only output for good.
    ``schema_in_prompt``
        Put the schema text in the system message (off when the user message
        already carries it, as the statement corrector's does).
    ``salvage_items``
        A set model keeps its valid items when a few fail (the question
        stage re-asks the missing slots).
    ``passthrough_items``
        A batch whose items could not all be validated is handed on as raw
        dicts (the planner validates each plan on its own and re-requests
        the broken one).
    ``repair_call``
        One short correction call over the model's own JSON before the full
        re-prompt.
    ``default_max_tokens`` / ``budget_ceiling`` / ``lower_effort_on_truncation``
        The output budget of a call that passes none, where escalation
        stops, and whether exhaustion at the ceiling also lowers
        ``reasoning_effort`` (off: validity over tokens).

    ``call_observer`` (instance attribute, ``None`` by default) receives one
    record dict per attempt (outcome, tokens, raw text, errors, fixes) — the
    experiment harness and the pipeline logger hook in there.
    """

    preamble: str = "You are a helpful assistant."
    native_schema: bool = False
    schema_in_prompt: bool = True
    salvage_items: bool = False
    passthrough_items: bool = False
    repair_call: bool = True
    default_max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    budget_ceiling: int = BUDGET_CEILING
    lower_effort_on_truncation: bool = False
    # Label of this client in observer records (None -> the class name).
    client_name: Optional[str] = None

    def __init__(self, config_path: Path, response_model: str = "response_model"):
        super().__init__(config_path)
        self.response_model = self._load_pydantic_response_model(response_model)
        self._native_ok = True
        self.call_observer: Optional[Callable[[dict], None]] = None
        self.last_prompt: Optional[str] = None

    # ------------------------------------------------------------------
    # Pydantic helpers
    # ------------------------------------------------------------------

    def _load_pydantic_response_model(self, model: str) -> Optional[Type[BaseModel]]:
        """Dynamically load a Pydantic model class from the YAML config."""
        response_model_config = self.config.get(model)
        if not response_model_config:
            return None

        module_name = response_model_config.get("module")
        class_name = response_model_config.get("class")
        if not module_name or not class_name:
            raise ValueError("response_model must specify both 'module' and 'class'")

        try:
            # Anchored to the fixed "orqa.agent" package rather than
            # __package__: response-model module paths in the YAML config
            # (e.g. "structured_outputs") are always relative to
            # orqa.agent regardless of which subpackage this class itself
            # lives in, so resolution must not depend on this file's own
            # location.
            module = importlib.import_module(f".{module_name}", package="orqa.agent")
            model_class = getattr(module, class_name)
            if not issubclass(model_class, BaseModel):
                raise TypeError(f"{class_name} is not a Pydantic BaseModel")
            return model_class
        except ImportError as e:
            raise ImportError(f"Could not import module '{module_name}': {e}") from e
        except AttributeError as e:
            raise AttributeError(
                f"Could not find class '{class_name}' in module '{module_name}': {e}"
            ) from e

    def schema_constraint(self) -> str:
        """The static output-format constraint block for the current response model.

        Deterministic for a given response model (``model_json_schema`` is
        stable), so it can safely live in the system message where it extends
        the cacheable static prefix instead of trailing the dynamic content.
        """
        format_str = json.dumps(self.response_model.model_json_schema(), indent=2)
        return (
            "### Output format instructions\n"
            "Return the final answer **strictly in JSON** and **exactly matching** "
            "the Pydantic following schema provided:\n\n"
            f"{format_str}\n\n"
            "All required fields must be present, field names and data types must match exactly, "
            "and all validation constraints (including value ranges, uniqueness, and completeness "
            "rules) must be satisfied. Do not add extra fields, omit required fields, "
            "or include explanatory text outside the JSON."
        )

    def reform_prompt_constraint(self, prompt: str) -> str:
        """Append the Pydantic JSON-schema constraint to a prompt."""
        return f"{prompt}\n{self.schema_constraint()}"

    def response_format(self) -> dict:
        """The provider-side ``response_format`` for native JSON-schema decoding."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.response_model.__name__,
                "schema": inline_refs(self.response_model.model_json_schema()),
                "strict": True,
            },
        }

    # ------------------------------------------------------------------
    # JSON cleaning & repair pipeline
    # ------------------------------------------------------------------

    def _clean_json_response(self, content: str) -> str:
        """
        Strip markdown fences and extract the outermost JSON object or array.
        Returns the trimmed content even if no valid JSON delimiters are found,
        so the repair pipeline can attempt further fixes.
        """
        content = content.strip()

        # Remove markdown code fences
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # Locate the first opening delimiter
        brace_start = content.find("{")
        bracket_start = content.find("[")

        if brace_start == -1 and bracket_start == -1:
            return content

        if brace_start == -1:
            start = bracket_start
        elif bracket_start == -1:
            start = brace_start
        else:
            start = min(brace_start, bracket_start)

        opener = content[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        for i in range(start, len(content)):
            if content[i] == opener:
                depth += 1
            elif content[i] == closer:
                depth -= 1
                if depth == 0:
                    return content[start : i + 1].strip()

        return content.strip()

    def _fix_triple_quotes(self, content: str) -> str:
        """Replace triple-quoted strings with a valid single-quoted JSON string."""
        pattern = r'"""[\s\S]*?"""'
        offset = 0
        result = content
        for match in re.finditer(pattern, content):
            original = match.group(0)
            inner = original[3:-3].strip()
            escaped = (
                inner
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "")
                .replace("\t", " ")
            )
            replacement = f'"{escaped}"'
            start = match.start() + offset
            end = match.end() + offset
            result = result[:start] + replacement + result[end:]
            offset += len(replacement) - len(original)
        return result

    def _escape_literal_controls(self, content: str) -> str:
        """
        Escape literal control characters (newlines, tabs, carriage returns) that
        appear inside JSON string values but were not escaped by the LLM.
        """
        result: list[str] = []
        in_string = False
        i = 0
        while i < len(content):
            ch = content[i]
            if in_string:
                if ch == "\\":
                    result.append(ch)
                    i += 1
                    if i < len(content):
                        result.append(content[i])
                        i += 1
                    continue
                elif ch == '"':
                    in_string = False
                    result.append(ch)
                elif ch == "\n":
                    result.append("\\n")
                elif ch == "\r":
                    result.append("\\r")
                elif ch == "\t":
                    result.append("\\t")
                else:
                    result.append(ch)
            else:
                if ch == '"':
                    in_string = True
                result.append(ch)
            i += 1
        return "".join(result)

    def _fix_structural_issues(self, content: str) -> str:
        """
        Fix common structural JSON problems produced by LLMs:
        - Trailing commas before } or ]
        - Unterminated string at end of document
        - Unclosed braces/brackets (truncated output)
        """
        # 1. Remove trailing commas before closing delimiters
        fixed = re.sub(r",\s*(?=[}\]])", "", content)

        # 2. Close any unterminated string (odd number of unescaped quotes)
        unescaped_quotes = re.findall(r'(?<!\\)"', fixed)
        if len(unescaped_quotes) % 2 != 0:
            fixed = fixed.rstrip() + '"'

        # 3. Close unclosed braces/brackets (handles truncated LLM output)
        closer_map = {"{": "}", "[": "]"}
        stack: list[str] = []
        in_str = False
        i = 0
        while i < len(fixed):
            ch = fixed[i]
            if in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in ("{", "["):
                    stack.append(closer_map[ch])
                elif ch in ("}", "]"):
                    if stack and stack[-1] == ch:
                        stack.pop()
            i += 1

        fixed = fixed.rstrip()
        for closer in reversed(stack):
            fixed += closer

        return fixed

    def _repair_json(self, content: str) -> dict:
        """
        Multi-stage JSON repair pipeline.

        Stage 1 – fast path (valid JSON)
        Stage 2 – fix triple-quoted strings
        Stage 3 – escape literal control characters inside strings
        Stage 4 – structural fixes (trailing commas, unclosed delimiters)

        Raises ``json.JSONDecodeError`` if all stages fail.
        """
        if not isinstance(content, str):
            raise json.JSONDecodeError("content is not a string", str(content), 0)

        # Stage 1
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Stage 2
        patched = self._fix_triple_quotes(content)
        try:
            return json.loads(patched)
        except json.JSONDecodeError:
            pass

        # Stage 3
        escaped = self._escape_literal_controls(patched)
        try:
            return json.loads(escaped)
        except json.JSONDecodeError:
            pass

        # Stage 4 – will raise if still broken, giving the caller a meaningful error
        structural = self._fix_structural_issues(escaped)
        return json.loads(structural)

    def _normalize_response(self, json_data: Any, root_key: str | None = "queries") -> Any:
        """
        Normalize the LLM response for Pydantic validation.

        If ``root_key`` is None, return the parsed JSON unchanged.
        Otherwise, coerce the top-level JSON into ``{root_key: [...]}`` form.
        """
        if root_key is None:
            return json_data
        if isinstance(json_data, list):
            return {root_key: json_data}
        if isinstance(json_data, dict):
            if root_key in json_data:
                return json_data
            if json_data:
                return {root_key: [json_data]}
        return {root_key: []}

    # ------------------------------------------------------------------
    # Error formatting helpers
    # ------------------------------------------------------------------

    # NOTE (both formatters): the response schema is deliberately NOT repeated
    # here. Every call site that renders these already carries the schema in
    # the SAME request — in the system message for the generation/judge clients
    # (see `schema_constraint`, kept there so it extends the static prefix), or
    # in the original user message for the correction/task-proposer clients
    # (via `reform_prompt_constraint`). Interpolating it again cost a second —
    # and, on a third attempt, a third — full copy of a schema that runs to
    # ~19.5k characters for the plan judge, for no added information.

    def _format_json_error(self, content: str, error: Exception) -> str:
        excerpt = (content or "").strip()[:_FAILED_OUTPUT_EXCERPT_CHARS]
        truncated = "\n…(truncated)" if len(content or "") > _FAILED_OUTPUT_EXCERPT_CHARS else ""
        quoted = f"\nYour previous output started:\n{excerpt}{truncated}\n" if excerpt else ""
        return (
            "JSON PARSING ERROR — your previous response was not valid JSON.\n"
            f"Parser error: {error}\n"
            f"{quoted}"
            "Return ONLY a valid JSON object matching the schema already given "
            "above — no prose before or after it. Usual causes: missing quotes, "
            "a trailing comma, an unescaped character, or text wrapped around "
            "the JSON."
        )

    def _format_validation_error(self, error: ValidationError) -> str:
        lines = [
            f"  - {' -> '.join(str(x) for x in err['loc'])}: {err['msg']} ({err['type']})"
            for err in error.errors()
        ]
        return (
            "SCHEMA VALIDATION ERROR — your JSON did not match the required "
            "schema on these fields:\n"
            + "\n".join(lines)
            + "\nFix exactly these fields and return the full JSON object again, "
            "matching the schema already given above."
        )

    def _validation_feedback(self, errors: list[str], content: str) -> str:
        excerpt = (content or "").strip()[:_FAILED_OUTPUT_EXCERPT_CHARS]
        more = "\n…(truncated)" if len(content or "") > _FAILED_OUTPUT_EXCERPT_CHARS else ""
        return (
            "SCHEMA VALIDATION ERROR — your JSON did not match the required schema on these fields:\n"
            + "\n".join(f"  - {e}" for e in errors[:40])
            + f"\nYour previous output started:\n{excerpt}{more}\n"
            "Fix exactly these fields and return the full JSON object again, matching the schema "
            "already given above."
        )

    def _budget_ceiling(self, cap: Optional[int] = None) -> int:
        """Where escalation stops for a call: ``budget_ceiling`` (what the
        provider accepts at most), lowered to the target deployment's own
        ``max_output_tokens`` when it declares one (a dedicated endpoint's
        limit)."""
        ceiling = int(self.budget_ceiling)
        if cap:
            ceiling = min(ceiling, int(cap))
        return ceiling

    def _initial_budget(self, call_kwargs: dict, cap: Optional[int] = None) -> None:
        """Set the call's opening ``max_tokens`` in place: the caller's own
        value, else ``default_max_tokens``, clamped to the deployment's cap
        (a planner asking 65536 of an endpoint that takes 16000 gets 16000,
        logged once per client, rather than a rejected request)."""
        if self.default_max_tokens and not call_kwargs.get("max_tokens"):
            call_kwargs["max_tokens"] = self.default_max_tokens
        requested = call_kwargs.get("max_tokens")
        if cap and requested and int(requested) > int(cap):
            if not getattr(self, "_cap_logged", False):
                logger.info("%s: output budget %s clamped to the deployment's max_output_tokens %s",
                            self._label, requested, cap)
                self._cap_logged = True
            call_kwargs["max_tokens"] = int(cap)

    def _truncation_feedback(self, call_kwargs: dict, cap: Optional[int] = None) -> str:
        """Raise the output budget for the retry (in place, on
        ``call_kwargs``); exhaustion AT the ceiling may also ask the model to
        think less (``reasoning_effort``, dropped by providers whose model
        has no such knob) when ``lower_effort_on_truncation`` is on."""
        current = int(call_kwargs.get("max_tokens") or 4096)
        ceiling = self._budget_ceiling(cap)
        if current >= ceiling and self.lower_effort_on_truncation:
            call_kwargs["reasoning_effort"] = "low"
        call_kwargs["max_tokens"] = min(current * 2, ceiling)
        return (
            "OUTPUT CUT OFF — your previous answer hit the output length limit before the JSON "
            "was complete. Return the complete JSON object again, more concisely: shorter "
            "descriptions and reasons, no repetition, no blank lines, nothing outside the JSON."
        )

    # ------------------------------------------------------------------
    # Observer
    # ------------------------------------------------------------------

    def _observe(self, record: dict) -> None:
        # Every attempt also goes to the generation trace, when one is open
        # (a no-op otherwise); the schema name tells the question judge's
        # votes from the plan judge's, which share a panel name.
        trace_llm_call(record, getattr(getattr(self, "response_model", None), "__name__", None))
        observer = getattr(self, "call_observer", None)
        if observer is None:
            return
        try:
            observer(record)
        except Exception as exc:  # noqa: BLE001 — observers never break a call
            logger.debug("call_observer failed: %s", exc)

    @property
    def _label(self) -> str:
        return self.client_name or type(self).__name__

    # ------------------------------------------------------------------
    # One call, one parse, one validation
    # ------------------------------------------------------------------

    def _system_message(self, system_prompt: Optional[str] = None) -> dict:
        """The system message: ``system_prompt`` (or the client's preamble),
        followed by the schema text unless the user message carries it."""
        head = system_prompt if system_prompt is not None else self.preamble
        if self.schema_in_prompt and self.response_model is not None:
            body = f"{head}\n\n{self.schema_constraint()}"
        elif system_prompt is not None:
            body = head
        else:
            body = (f"{head} Return only a JSON object matching the required "
                    "response schema, with no prose around it.")
        return {"role": "system", "content": body}

    def _call(self, messages: list[dict], native: bool, _record: Optional[dict] = None,
              **kwargs) -> tuple[Any, bool]:
        """One router call. Returns ``(response, native_used)``; a provider
        that rejects ``response_format`` turns native mode off for good.

        ``_record`` (the attempt's call record) receives what is sent, as
        sent, before the request goes out — so a call that times out still
        says what it asked: the system and user messages, the parameters and
        the configured deployment (the generation trace keeps them,
        ``trace.llm_io``)."""
        args = {
            "model": "primary",
            "messages": sanitize_messages(messages),
            **self._default_temperature_kwargs(self.config["model"]),
            **kwargs,
        }
        if _record is not None:
            sent = args["messages"]
            _record["system"] = next((m["content"] for m in sent if m.get("role") == "system"), None)
            _record["prompt"] = sent[-1]["content"] if sent else ""
            _record["params"] = {k: v for k, v in args.items() if k not in ("model", "messages")}
            _record["deployment"] = getattr(self, "primary_label", None)
        if native:
            try:
                return self._completion_with_backoff({**args, "response_format": self.response_format()}), True
            except Exception as exc:  # noqa: BLE001
                if not _looks_like_format_rejection(exc):
                    raise
                logger.warning("%s: provider rejected response_format (%s); native schema mode off.",
                               self._label, str(exc)[:160])
                self._native_ok = False
        return self._completion_with_backoff(args), False

    def _parse(self, content: str) -> tuple[Any, str]:
        cleaned = self._clean_json_response(content)
        return self._repair_json(cleaned), cleaned

    def _validate(self, json_data: Any, record: dict) -> Optional[dict]:
        """Strict, then coerced / salvaged / passed through (per the class
        switches). Fills the record's outcome fields; returns the result dict
        or ``None``."""
        model = self.response_model
        try:
            result = model.model_validate(json_data)
            # A set model "validates" an answer that carries none of its
            # fields, because its one list field defaults to empty — every
            # slot then reads "nothing returned". That is not a valid answer.
            if not validated_empty(result, json_data, model):
                record["outcome"] = "strict"
                return result.model_dump()
            record["strict_errors"] = ["$: none of the schema's fields is present (an empty set by default)"]
        except ValidationError as exc:
            record["strict_errors"] = errors_brief(exc.errors(include_url=False))
        inst, coerced, fixes, remaining = validate_leniently(json_data, model)
        record["fixes"] = fixes
        if inst is not None and not validated_empty(inst, coerced, model):
            record["outcome"] = "coerced"
            logger.info("%s: answer validated after shape coercion (%s)", self._label, "; ".join(fixes[:6]))
            return inst.model_dump()
        record["coerce_errors"] = errors_brief(remaining) or list(record["strict_errors"])
        if self.salvage_items:
            inst, kept, dropped, more = salvage_items(coerced, model)
            if inst is not None:
                record["outcome"] = "salvaged"
                record["salvage"] = {"kept": kept, "dropped": dropped}
                record["fixes"] = fixes + more
                logger.warning("%s: kept %d valid item(s), dropped %d", self._label, kept, dropped)
                return inst.model_dump()
        if self.passthrough_items and isinstance(coerced, dict):
            lists = [k for k, v in coerced.items() if isinstance(v, list) and v and all(isinstance(i, dict) for i in v)]
            if len(lists) == 1:
                record["outcome"] = "passthrough"
                return coerced
        return None

    def _repair(self, json_data: Any, errors: list[str], native: bool,
                max_tokens: Optional[int], system_message: dict) -> tuple[Optional[dict], dict, dict]:
        """One short correction call over the model's own JSON."""
        record: dict = {"client": self._label, "kind": "repair", "native_schema": native}
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        document = json.dumps(json_data, ensure_ascii=False, default=str)
        if len(document) > 60000:
            record["outcome"] = "skipped_too_long"
            return None, usage, record
        user = (
            "The JSON document below does not satisfy the required schema. "
            "Validation errors (field path: problem):\n"
            + "\n".join(f"- {e}" for e in errors[:40])
            + "\n\nReturn the complete corrected JSON document only: keep every value that is "
            "already valid, change only what the errors require (shape, types, allowed values, "
            "missing fields), add no commentary.\n\n```json\n" + document + "\n```"
        )
        messages = [system_message, {"role": "user", "content": user}]
        record["prompt"] = user
        t0 = time.perf_counter()
        try:
            kwargs = {"max_tokens": max(int(max_tokens or 0), REPAIR_MAX_TOKENS_FLOOR)}
            response, native_used = self._call(messages, native and self._native_ok, _record=record, **kwargs)
            record["model"] = response.get("model")
            usage = _usage_of(response)
            choice = response["choices"][0]
            content = choice["message"]["content"] or ""
            reasoning = _reasoning_of(choice["message"])
            if reasoning:
                record["reasoning"] = reasoning
            record.update(native_schema=native_used, finish_reason=choice.get("finish_reason"), usage=usage,
                          elapsed_s=round(time.perf_counter() - t0, 2), raw=content)
            data, _ = self._parse(content)
            result = self._validate(data, record)
            if result is not None:
                record["outcome"] = "repaired:" + record["outcome"]
                return result, usage, record
            record["outcome"] = "repair_failed"
        except Exception as exc:  # noqa: BLE001
            record["outcome"] = "repair_error"
            record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return None, usage, record

    # ------------------------------------------------------------------
    # Generic completion loop
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        root_key: str | None = "queries",
        *,
        system_prompt: Optional[str] = None,
        pre_validate: Optional[Callable[[Any], Any]] = None,
        **kwargs,
    ) -> Any:
        """
        Make a structured completion request with automatic retry and error feedback.

        ``system_prompt`` replaces the client's preamble for this call (a
        judge passes its instructions; the schema still follows them).
        ``pre_validate`` is applied to the parsed, normalized JSON before
        validation (the statement corrector backfills the unchanged fields of
        the source query there, so a lean answer carrying only the fixed
        ``code`` validates). Remaining ``kwargs`` go to the router call
        (``max_tokens`` — defaulted to ``default_max_tokens`` — and the like).

        Returns ``(result_dict, usage_total)`` on success; ``({}, usage_total)``
        on exhausted retries.
        """
        self.last_prompt = prompt
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # The prompt goes in a USER message (with the system preamble + schema
        # ahead of it, static per response model so the provider's prompt
        # cache serves that prefix). Rebuilt (never appended to) on every
        # retry: the feedback goes AFTER the unchanged prompt.
        system_message = self._system_message(system_prompt)

        def build(user_content: str) -> list[dict]:
            return [system_message, {"role": "user", "content": user_content}]

        messages = build(prompt)
        call_kwargs = dict(kwargs)
        cap = self.primary_max_output_tokens
        self._initial_budget(call_kwargs, cap)
        last_error: Optional[str] = None
        native_off_once = False

        for attempt in range(self.max_retries):
            native = self.native_schema and self._native_ok and not native_off_once
            native_off_once = False
            record: dict = {"client": self._label, "kind": "call", "attempt": attempt + 1,
                            "native_schema": native, "max_tokens": call_kwargs.get("max_tokens")}
            t0 = time.perf_counter()
            feedback: Optional[str] = None
            try:
                logger.debug("%s: attempt %d/%d (max_tokens %s)", self._label, attempt + 1,
                             self.max_retries, call_kwargs.get("max_tokens"))
                record["prompt"] = messages[-1]["content"]
                response, native_used = self._call(messages, native, _record=record, **call_kwargs)
                record["native_schema"] = native_used
                record["model"] = response.get("model")
                # Only a configured fallback deployment can answer in the
                # primary's place; with none, the served name is whatever
                # the provider echoes (an OCID on a dedicated OCI endpoint).
                if self.fallback_names and served_by_fallback(response, self.primary_model, "primary"):
                    record["fallback_model"] = True
                    logger.warning("%s: answer served by fallback model %s (primary %s)",
                                   self._label, record["model"], self.primary_label)
                usage = _usage_of(response)
                _add_usage(usage_total, usage)
                choice = response["choices"][0]
                finish = choice.get("finish_reason")
                content = choice["message"]["content"]
                content = "" if content is None else str(content)
                record.update(finish_reason=finish, usage=usage, elapsed_s=round(time.perf_counter() - t0, 2),
                              raw=content, raw_chars=len(content),
                              whitespace_ratio=round(_whitespace_ratio(content), 3))
                reasoning = _reasoning_of(choice["message"])
                if reasoning:
                    record["reasoning"] = reasoning
                truncated = budget_exhausted(finish, usage, call_kwargs.get("max_tokens"), content)
                record["truncated"] = truncated

                loop = degenerate_repetition(content)
                if loop:
                    record["outcome"] = "degenerate_loop"
                    record["loop_unit"] = loop
                    last_error = f"degenerate repetition: {loop!r}"
                    logger.warning("%s: degenerate repetition in the answer (%r); retrying", self._label, loop)
                    native_off_once = True
                    feedback = ("DEGENERATE OUTPUT — your previous answer repeated the same sentence over and over "
                                "until the output limit. Return the complete JSON once, concisely, with no repeated text.")
                    continue

                if not content.strip():
                    if truncated:
                        record["outcome"] = "truncated_empty"
                        last_error = "empty answer: output budget exhausted by reasoning"
                        logger.warning("%s: empty answer at %s tokens; raising the budget",
                                       self._label, call_kwargs.get("max_tokens"))
                        feedback = self._truncation_feedback(call_kwargs, cap)
                        continue
                    raise ValueError("LLM returned empty content.")

                try:
                    json_data, cleaned = self._parse(content)
                    record["json_ok"] = True
                except json.JSONDecodeError as exc:
                    record["json_ok"] = False
                    record["json_error"] = str(exc)[:200]
                    last_error = f"JSON: {exc}"
                    if truncated:
                        record["outcome"] = "truncated"
                        logger.warning("%s: answer cut at %s tokens (unparseable); raising the budget",
                                       self._label, call_kwargs.get("max_tokens"))
                        feedback = self._truncation_feedback(call_kwargs, cap)
                    else:
                        record["outcome"] = "json_error"
                        logger.warning("%s: JSON parsing error on attempt %d: %s", self._label, attempt + 1, exc)
                        feedback = self._format_json_error(cleaned, exc)
                    continue

                record["json_repaired"] = _needed_structural_repair(cleaned)
                if truncated and record["json_repaired"]:
                    # cut mid-document and patched shut: whatever validates now
                    # is a partial answer (a plan's tail, a missing item), so
                    # retry with a larger budget instead of shipping it
                    record["outcome"] = "truncated_repaired"
                    last_error = "answer cut at the output budget (structure repaired)"
                    logger.warning("%s: answer cut at %s tokens (closed by the repair stage); raising the budget",
                                   self._label, call_kwargs.get("max_tokens"))
                    feedback = self._truncation_feedback(call_kwargs, cap)
                    continue
                if json_data in ({}, [], None):
                    # `{}` would normalize to an empty set and "validate"
                    record["outcome"] = "empty_json"
                    last_error = "empty JSON document"
                    logger.warning("%s: empty JSON document on attempt %d", self._label, attempt + 1)
                    feedback = self._validation_feedback(["$: the document is empty — return the full object"], content)
                    continue

                json_data = self._normalize_response(json_data, root_key=root_key)
                if pre_validate is not None:
                    json_data = pre_validate(json_data)
                result = self._validate(json_data, record)
                if result is not None:
                    logger.debug("%s: success on attempt %d (%s)", self._label, attempt + 1, record["outcome"])
                    return result, usage_total
                errors = record.get("coerce_errors") or record.get("strict_errors") or []
                last_error = "; ".join(errors[:5])
                if truncated:
                    record["outcome"] = "truncated"
                    logger.warning("%s: answer cut at %s tokens (invalid); raising the budget",
                                   self._label, call_kwargs.get("max_tokens"))
                    feedback = self._truncation_feedback(call_kwargs, cap)
                    continue
                record["outcome"] = "invalid"
                logger.warning("%s: validation error on attempt %d: %s", self._label, attempt + 1, last_error)
                if self.repair_call:
                    repaired, r_usage, r_record = self._repair(
                        json_data, errors, native, call_kwargs.get("max_tokens"), system_message
                    )
                    _add_usage(usage_total, r_usage)
                    self._observe({**r_record, "attempt": attempt + 1})
                    if repaired is not None:
                        record["outcome"] = "invalid->repaired"
                        logger.info("%s: answer fixed by the repair call", self._label)
                        return repaired, usage_total
                feedback = self._validation_feedback(errors, content)
            except Exception as exc:  # noqa: BLE001
                record["outcome"] = "error"
                record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                last_error = record["error"]
                logger.error("%s: error on attempt %d: %s", self._label, attempt + 1, exc)
                if attempt < self.max_retries - 1:
                    # A throttle that outlived its backoff budget means the
                    # provider is still shedding load, so wait longer than
                    # the ordinary content-retry delay before spending
                    # another attempt on it.
                    delay = (
                        max(self.retry_delay, self.throttle_max_delay)
                        if self.is_throttled(exc) else self.retry_delay
                    )
                    time.sleep(delay)
                continue
            finally:
                self._observe(record)
                if feedback is not None and attempt < self.max_retries - 1:
                    messages = build(f"{prompt}\n\n{feedback}")
                    time.sleep(self.retry_delay)

        logger.error("%s: failed after %d attempts. Last error: %s", self._label, self.max_retries, last_error)
        return {}, usage_total
