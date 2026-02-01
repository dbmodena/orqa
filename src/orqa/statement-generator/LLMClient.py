import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Type

import yaml
from litellm import completion, Router
from pydantic import BaseModel, ValidationError



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
        self.enable_json_mode = self.config.get("enable_json_mode", True)
        self.provider_params = self.config.get("provider_params", {}) or {}
        # 3. Setup router LAST (depends on provider_params being set)
        self.router = self._setup_router()



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


    def _setup_router(self) -> Router:
        """Setup LiteLLM Router with primary and fallback models."""
        model_list = []
        
        # Primary model
        primary_model = self.config["model"]
        primary_params = self._get_provider_specific_params(primary_model)
        model_list.append({
            "model_name": "primary",
            "litellm_params": {
                "model": primary_model,
                **primary_params
            }
        })
        
        # Fallback models
        fallback_names = []
        for idx, fallback_model in enumerate(self.config.get("fallback_models", [])):
            fallback_name = f"fallback_{idx}"
            fallback_names.append(fallback_name)
            fallback_params = self._get_provider_specific_params(fallback_model)
            model_list.append({
                "model_name": fallback_name,
                "litellm_params": {
                    "model": fallback_model,
                    **fallback_params
                }
            })
        
        # Setup fallback chain: primary -> fallback_0 -> fallback_1 -> ...
        fallbacks = [{"primary": fallback_names}] if fallback_names else []
        
        return Router(
            model_list=model_list,
            fallbacks=fallbacks,
            num_retries=1,  # Router handles retries per model
            timeout=60,
            set_verbose=True  # Shows which model is being used
        )


    def _load_config(self, config_path) -> dict[str, Any]:
        """Load configuration from YAML file"""
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
        usage_total = {"prompt_tokens": 0,"completion_tokens": 0,"total_tokens": 0,}
        messages = [{"role": "system", "content": prompt}]
        completion_args = { "model": "primary", "messages": messages,"temperature": self.temp,**kwargs,}
        last_content = None
        last_error = None
        for attempt in range(self.max_retries):
            try:
                print(f"Attempt {attempt + 1}/{self.max_retries}...")
                response = self.router.completion(**completion_args)
                usage = response["usage"]
                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"] += usage.get("total_tokens", 0)
                content = response["choices"][0]["message"]["content"]
                last_content = content
                print(f"✓ Success on attempt {attempt + 1}\n")
                return result, usage_total
            except Exception as e:
                last_error = e
                print(f"✗ Error on attempt {attempt + 1}: {e}")
                # Wait before retry
                if attempt < self.max_retries - 1:
                    message.append({"role":"user","content":e})
                    print(f"Retrying in {self.retry_delay} seconds...\n")
                    time.sleep(self.retry_delay)
        # All retries exhausted
        print(f"\n❌ Failed after {self.max_retries} attempts")
        print(f"Last error: {last_error}")
        if last_content and not self.raw:
            print(f"\nLast response preview:\n{last_content[:300]}...\n")

        return {}, usage_total
