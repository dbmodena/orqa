import random
import time
from pathlib import Path
from typing import Any

import litellm
import yaml
from litellm import Router

from ..utility.message_builder import sanitize_messages

import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LiteLLM client with YAML configuration.
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
        # 3. Setup router LAST (depends on provider_params being set)
        self.router = self._setup_router()

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
        """Router call that absorbs provider throttling by backing off.

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
        for attempt in range(self.throttle_max_retries + 1):
            try:
                return self.router.completion(**completion_args)
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

        litellm's Router always lets an explicit call-time kwarg beat a
        deployment's own ``litellm_params`` default, so unconditionally
        passing ``temperature=self.temperature`` on every call — the
        previous behavior — silently clobbers any such override regardless
        of how it was configured. Returning `{}` here lets the deployment's
        own default flow through untouched instead.
        """
        overrides = self._get_provider_specific_params(model)
        if extra_overrides:
            overrides = {**overrides, **extra_overrides}
        if "temperature" in overrides:
            return {}
        return {"temperature": self.temperature}

    def _setup_router(self) -> Router:
        """Setup LiteLLM Router with primary and fallback models."""
        model_list = []

        # Primary model
        primary_model = self.config["model"]
        primary_params = self._get_provider_specific_params(primary_model)
        model_list.append(
            {
                "model_name": "primary",
                "litellm_params": {"model": primary_model, **primary_params},
            }
        )

        # Fallback models
        fallback_names = []
        for idx, fallback_model in enumerate(self.config.get("fallback_models", [])):
            fallback_name = f"fallback_{idx}"
            fallback_names.append(fallback_name)
            fallback_params = self._get_provider_specific_params(fallback_model)
            model_list.append(
                {
                    "model_name": fallback_name,
                    "litellm_params": {"model": fallback_model, **fallback_params},
                }
            )

        # Setup fallback chain: primary -> fallback_0 -> fallback_1 -> ...
        fallbacks = [{"primary": fallback_names}] if fallback_names else []

        return Router(
            model_list=model_list,
            fallbacks=fallbacks,
            num_retries=0,  # Router handles retries per model
            timeout=60,
            set_verbose=False,  # Shows which model is being used
        )

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
