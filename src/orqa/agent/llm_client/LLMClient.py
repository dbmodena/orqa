import os
import random
import re
import time
from pathlib import Path
from typing import Any, Optional

import litellm
import yaml

from orqa.agent.llm_client.messages import sanitize_messages

import logging

logger = logging.getLogger(__name__)


# Keys of a ``deployments.<name>`` entry (and of a mapping-form model spec)
# that are consumed HERE rather than handed to LiteLLM as litellm_params.
_SPEC_OWN_KEYS = ("model", "name", "max_output_tokens", "description")
# ``${NAME}`` inside a deployment's ``model`` string: an endpoint OCID kept in
# .env (LiteLLM's own ``os.environ/NAME`` form covers whole values only, and
# the model string also carries the provider prefix).
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# When set, the one model reference (a deployment name or LiteLLM model
# string) any client or judge panel may resolve: a run pinned to one endpoint
# stops at start instead of calling another model.
ONLY_LLM_ENV = "ORQA_ONLY_LLM"
# Seconds a request may take when neither the call nor the target's params
# (provider_params / deployment) set a ``timeout``.
DEFAULT_TIMEOUT = 60
# LiteLLM model strings already announced by this process (see
# LLMClient._register_target).
_ANNOUNCED: set[str] = set()


class DisallowedModelError(RuntimeError):
    """The LLM config names a model other than ``$ORQA_ONLY_LLM``. Not a
    ValueError: a judge panel skips an entry it cannot resolve, and this one
    must stop the run, not shrink the panel."""


class LLMClient:
    """
    LiteLLM client with YAML configuration.

    Model references (``model``, ``judge_profiles.*[]``)
    are resolved by :meth:`_resolve_model_spec`. Besides a plain LiteLLM model
    string, a reference can name an entry of the YAML's ``deployments``
    section — a LiteLLM model string plus the litellm params that pin it to a
    specific serving endpoint. That is how OCI's dedicated AI clusters are
    addressed: the model string (``oci/openai.gpt-oss-20b``) still decides the
    request format and catalog entry, while ``oci_serving_mode: DEDICATED``
    and ``oci_endpoint_id`` route the call to the cluster's endpoint instead
    of the on-demand pool. A deployment may also carry ``max_output_tokens``,
    the most that endpoint accepts in one answer: the structured loop clamps
    its default budget and its escalation ceiling to it.
    """

    def __init__(self, config_path: Path):
        """
        Initialize LLM client with configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        # 1. Load configuration
        self.config = self._load_config(config_path)
        # 2. Set basic attributes
        self.temperature = self.config.get("temperature", 0.2)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        # Provider THROTTLING is retried on its own budget, separate from
        # max_retries: a 429 is not a bad answer from the model, and letting
        # it burn the attempts reserved for correcting a malformed response
        # means one throttled call can fail a request the model never got to
        # answer. See _completion_with_backoff.
        self.throttle_max_retries = self.config.get("throttle_max_retries", 6)
        self.throttle_base_delay = self.config.get("throttle_base_delay", 5.0)
        self.throttle_max_delay = self.config.get("throttle_max_delay", 120.0)
        self.enable_json_mode = self.config.get("enable_json_mode", True)
        self.provider_params = self.config.get("provider_params", {}) or {}
        self.deployments = self._load_deployments()
        # The primary model, resolved once: ``primary_model`` is the LiteLLM
        # model string that is called, ``primary_label`` how the pipeline
        # names it (the deployment name when one is configured).
        self._require_allowed(self.config["model"])
        self.primary = self._resolve_model_spec(self.config["model"])
        self.primary_model: str = self.primary["model"]
        self.primary_label: str = self.primary["label"]
        self.primary_max_output_tokens: Optional[int] = self.primary["max_output_tokens"]
        # 3. Call targets: the name a call passes as ``model`` -> what
        # litellm.completion is called with (see _register_target).
        self._targets: dict[str, dict] = {}
        self._register_target("primary", self.primary)

    @staticmethod
    def is_throttled(exc: BaseException) -> bool:
        """Whether ``exc`` is the provider refusing on rate, not on content.

        Matched on the message rather than the exception type because
        LiteLLM surfaces OCI throttling as a generic ``APIConnectionError``
        wrapping the provider payload (``OciException - {"code": "429",
        "message": "Service request limit is exceeded, request is throttled
        for tenant:..."}``), so ``RateLimitError`` alone never fires.
        """
        if isinstance(exc, getattr(litellm, "RateLimitError", ())):
            return True
        text = str(exc).lower()
        return (
            "429" in text
            or "rate limit" in text
            or "ratelimit" in text
            or "too many requests" in text
            or "request limit is exceeded" in text
            or "throttle" in text
        )

    def _completion_with_backoff(self, completion_args: dict) -> Any:
        """``litellm.completion`` call that absorbs provider throttling by
        backing off.

        ``completion_args["model"]`` names a registered target (``"primary"``
        or a judge panel's entry, :meth:`_register_target`); the request is
        the target's params with the call's own arguments on top, so an
        explicit call-time argument beats the target's default.

        OCI's on-demand limit is applied by DYNAMIC throttling: the ceiling
        is undocumented, moves with overall demand, and is steered partly by
        a tenancy's own recent throughput, so there is no fixed request rate
        a caller can stay under by construction. Oracle's guidance is to
        delay after a rejection, and warns that retrying rapidly instead
        drives further rejections and can get a client temporarily blocked.

        Exponential with FULL JITTER (a uniform draw below the ceiling, not
        the ceiling itself): every worker that gets throttled by the same
        capacity dip would otherwise wake at the same instant and recreate
        the burst that caused it.
        """
        name = str(completion_args.get("model"))
        target = self._targets.get(name)
        if target is None:
            raise ValueError(f"unknown LLM target {name!r}; registered: {', '.join(sorted(self._targets))}")
        if target["missing_env"]:
            raise ValueError(
                f"deployment {target['label']!r} cannot be called: unset environment variable(s) "
                f"{', '.join(target['missing_env'])} in its model string"
            )
        request = {**target["params"], **completion_args, "model": target["model"]}
        for attempt in range(self.throttle_max_retries + 1):
            try:
                return litellm.completion(**request)
            except Exception as exc:
                if not self.is_throttled(exc) or attempt == self.throttle_max_retries:
                    raise
                ceiling = min(
                    self.throttle_base_delay * (2 ** attempt), self.throttle_max_delay
                )
                delay = random.uniform(0, ceiling)
                logger.warning(
                    "Provider throttled the request (attempt %d/%d); backing off %.1fs",
                    attempt + 1,
                    self.throttle_max_retries,
                    delay,
                )
                time.sleep(delay)

    # ------------------------------------------------------------------
    # Model references: plain LiteLLM strings, named deployments, mappings
    # ------------------------------------------------------------------

    @staticmethod
    def _as_optional_int(value: Any, where: str) -> Optional[int]:
        """``None``, an int, or an ``os.environ/NAME`` indirection (unset or
        empty -> ``None``) coerced to int; anything else is a config error."""
        if value is None:
            return None
        if isinstance(value, str) and value.startswith("os.environ/"):
            value = os.environ.get(value.split("/", 1)[1], "").strip()
            if not value:
                return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where} must be an integer, got {value!r}") from exc
        if parsed <= 0:
            raise ValueError(f"{where} must be positive, got {parsed}")
        return parsed

    @staticmethod
    def _expand_env(text: str) -> tuple[str, list[str]]:
        """Expand ``${NAME}`` references from the environment; returns the
        text and the names that were unset (left in place)."""
        missing: list[str] = []

        def sub(match: "re.Match[str]") -> str:
            value = os.environ.get(match.group(1), "").strip()
            if not value:
                missing.append(match.group(1))
                return match.group(0)
            return value

        return _ENV_REF.sub(sub, text), missing

    def _load_deployments(self) -> dict[str, dict]:
        """Validate the YAML's ``deployments`` section into
        ``{name: {"model", "params", "max_output_tokens", "missing_env"}}``.

        A deployment whose ``model`` references an unset ``${NAME}`` is kept
        with the reference unexpanded (a client can be built offline, as the
        tests do); the first live call to it is refused with the variable's
        name (:meth:`_completion_with_backoff`), never sent half-addressed.
        """
        raw = self.config.get("deployments") or {}
        if not isinstance(raw, dict):
            raise ValueError(f"deployments must be a mapping of name -> deployment, got {type(raw).__name__}")
        deployments: dict[str, dict] = {}
        for name, entry in raw.items():
            if not isinstance(entry, dict) or not str(entry.get("model") or "").strip():
                raise ValueError(f"deployments.{name} must be a mapping with a non-empty 'model' key")
            entry = dict(entry)
            model = str(entry.pop("model")).strip()
            if model in raw:
                raise ValueError(
                    f"deployments.{name}.model must be a LiteLLM model string, not another deployment ({model!r})"
                )
            model, missing = self._expand_env(model)
            cap = self._as_optional_int(entry.pop("max_output_tokens", None), f"deployments.{name}.max_output_tokens")
            entry.pop("description", None)
            deployments[str(name)] = {
                "model": model, "params": entry, "max_output_tokens": cap, "missing_env": missing,
            }
        return deployments

    @staticmethod
    def _require_allowed(spec: Any) -> None:
        """Refuse a model reference of the LLM config (primary, judge) other than ``$ORQA_ONLY_LLM``. Checked where the config's
        references become call targets, not in :meth:`_resolve_model_spec`,
        which also looks up params of already-resolved model strings."""
        allowed = os.environ.get(ONLY_LLM_ENV, "").strip()
        name = str((spec.get("model") if isinstance(spec, dict) else spec) or "").strip()
        if allowed and name != allowed:
            raise DisallowedModelError(
                f"{ONLY_LLM_ENV}={allowed}, but the LLM config also names {name!r}: remove it from "
                "conf/llm/litellm.yaml (model, judge_profiles) or unset the variable"
            )

    def _resolve_model_spec(self, spec: Any) -> dict:
        """Resolve one model reference into what a call target needs.

        ``spec`` is a LiteLLM model string, the name of a ``deployments``
        entry, or a mapping with a ``model`` key (either of the two) plus
        per-use litellm overrides (optionally ``name`` for the label and
        ``max_output_tokens`` for the cap). Returns::

            {"label": str, "model": str, "params": dict, "max_output_tokens": Optional[int]}

        ``label`` is how the pipeline names the target in logs, votes and run
        metadata (the deployment name when one was used, else the model
        string); ``model`` is the LiteLLM model string; ``params`` are the
        provider params of that model's prefix, then the deployment's own
        litellm params, then the mapping's overrides — later wins.
        """
        overrides: dict = {}
        name: Optional[str] = None
        cap_override: Any = None
        if isinstance(spec, dict):
            entry = dict(spec)
            inner = entry.pop("model", None)
            name = entry.pop("name", None)
            cap_override = entry.pop("max_output_tokens", None)
            entry.pop("description", None)
            if not isinstance(inner, str) or not inner.strip():
                raise ValueError(f"a model spec mapping needs a non-empty 'model' key: {spec!r}")
            overrides = entry
            spec = inner
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError(f"invalid model spec: {spec!r}")
        spec = spec.strip()
        deployment = self.deployments.get(spec)
        if deployment is not None:
            if deployment["missing_env"]:
                logger.warning(
                    "deployment %r references unset environment variable(s) %s; calls to it will be refused",
                    spec, ", ".join(deployment["missing_env"]),
                )
            litellm_model = deployment["model"]
            params = {**self._get_provider_specific_params(litellm_model), **deployment["params"], **overrides}
            cap = deployment["max_output_tokens"]
        else:
            litellm_model = spec
            params = {**self._get_provider_specific_params(litellm_model), **overrides}
            cap = None
        if cap_override is not None:
            cap = self._as_optional_int(cap_override, f"{spec}.max_output_tokens")
        return {
            "label": str(name).strip() if name else spec,
            "model": litellm_model,
            "params": params,
            "max_output_tokens": cap,
            "missing_env": list(deployment["missing_env"]) if deployment is not None else [],
        }

    def _register_target(self, name: str, spec: dict, default_timeout: float = DEFAULT_TIMEOUT) -> None:
        """Make a resolved model spec callable as ``model=name``
        (:meth:`_completion_with_backoff`).

        The target's params are the spec's, with LiteLLM's whole-value
        ``os.environ/NAME`` indirections read from the environment (unset:
        the param is left out), over two defaults: ``timeout`` and
        ``max_retries: 0`` — retries are this client's own (throttle backoff
        and the correction loop), never the provider SDK's hidden ones.
        """
        params: dict[str, Any] = {"timeout": default_timeout, "max_retries": 0}
        for key, value in spec["params"].items():
            if isinstance(value, str) and value.startswith("os.environ/"):
                value = os.environ.get(value.split("/", 1)[1])
            if value is not None:
                params[key] = value
        self._targets[name] = {
            "label": spec["label"],
            "model": spec["model"],
            "params": params,
            "missing_env": spec.get("missing_env") or [],
        }
        # One console line per model and process: the run monitor reads it
        # to show which endpoints a run addresses (run_monitor._REGISTERED).
        if spec["model"] not in _ANNOUNCED:
            _ANNOUNCED.add(spec["model"])
            logger.warning("LLM target: model=%s (%s)", spec["model"], spec["label"])

    def _get_provider_from_model(self, model: str) -> str:
        """Extract provider name from model string."""
        if "/" in model:
            return model.split("/")[0].replace("_chat", "")
        return "unknown"

    def _get_provider_specific_params(self, model: str) -> dict[str, Any]:
        """
        Get provider-specific parameters for a given model.

        Args:
            model: Model string (e.g., "ollama_chat/llama3.3:latest")

        Returns:
            Dictionary of provider-specific parameters
        """
        provider = self._get_provider_from_model(model)

        # Defensive programming: handle None and missing keys
        if not self.provider_params:
            return {}

        provider_config = self.provider_params.get(provider, {})

        # Handle case where provider_config might be None
        if provider_config is None:
            return {}
        return provider_config.copy()

    def _default_temperature_kwargs(
        self, model: str, extra_overrides: dict[str, Any] | None = None
    ) -> dict[str, float]:
        """``{"temperature": self.temperature}`` unless the target model already
        pins its own via provider_params or a per-call override — e.g. a
        reasoning model that requires a fixed temperature while thinking
        (OCI's gpt-oss/gemini-2.5-flash, Anthropic's extended thinking).

        An explicit call-time kwarg always beats a target's own params
        (:meth:`_completion_with_backoff`), so unconditionally
        passing ``temperature=self.temperature`` on every call — the
        previous behavior — silently clobbers any such override regardless
        of how it was configured. Returning `{}` here lets the deployment's
        own default flow through untouched instead.
        """
        overrides = self._resolve_model_spec(model)["params"]
        if extra_overrides:
            overrides = {**overrides, **extra_overrides}
        if "temperature" in overrides:
            return {}
        return {"temperature": self.temperature}

    def _load_config(self, config_path) -> dict[str, Any]:
        """Load configuration from YAML file"""
        logger.debug("Loading LLM config from %s", config_path)
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def complete(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        """
        Make a completion request with optional structured output.
        :param prompt: The prompt to send to the model
        :param **kwargs: Additional arguments to pass to litellm.completion

        :return: If response_model is provided, instance of the Pydantic model,
                otherwise a raw string response
        """
        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        # The prompt goes in a USER message (with a minimal system preamble),
        # not a lone system message — some providers (e.g. Cohere via OCI)
        # reject requests with no user message at all.
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        completion_args = {
            "model": "primary",
            "messages": messages,
            **self._default_temperature_kwargs(self.config["model"]),
            **kwargs,
        }
        if self.primary_max_output_tokens and completion_args.get("max_tokens"):
            completion_args["max_tokens"] = min(
                int(completion_args["max_tokens"]), self.primary_max_output_tokens
            )
        last_content = None
        last_error = None
        for attempt in range(self.max_retries):
            try:
                #print(f"Attempt {attempt + 1}/{self.max_retries}...")
                completion_args["messages"] = sanitize_messages(messages)
                response = self._completion_with_backoff(completion_args)
                usage = response["usage"]
                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"] += usage.get("total_tokens", 0)
                content = response["choices"][0]["message"]["content"]
                last_content = content
                #print(f"✓ Success on attempt {attempt + 1}\n")
                return last_content, usage_total
            except Exception as e:
                last_error = e
                logger.error("Error on attempt %d: %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    # Only a CONTENT failure is worth showing the model; a
                    # throttle that outlived its own backoff budget says
                    # nothing about the answer, and appending it would
                    # re-send a LARGER prompt into a provider already
                    # refusing on load.
                    if not self.is_throttled(e):
                        messages.append({"role": "user", "content": str(e)})
                    logger.debug("Retrying in %s seconds...", self.retry_delay)
                    time.sleep(self.retry_delay)
        # All retries exhausted
        #print(f"\n❌ Failed after {self.max_retries} attempts")
        #print(f"Last error: {last_error}")
        #if last_content and not self.raw:
        #    print(f"\nLast response preview:\n{last_content[:300]}...\n")

        return {}, usage_total
