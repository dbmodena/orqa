"""LiteLLM embedding client, YAML-configured like :class:`LLMClient`."""

import logging
import os
import time
from pathlib import Path

import yaml
from litellm import embedding as litellm_embedding

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Thin LiteLLM embedding wrapper configured from litellm.yaml."""

    def __init__(self, config_path: Path, batch_size: int = 64):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.model: str = config["embedding_model"]
        self.batch_size = batch_size
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 10.0)

        provider = self.model.split("/")[0] if "/" in self.model else ""
        provider_params = (config.get("provider_params") or {}).get(provider) or {}
        # Router-only keys like drop_params are fine to pass to litellm.embedding;
        # resolve os.environ/ indirections the same way litellm does for chat.
        self.provider_params = {
            k: (
                os.environ.get(v.split("/", 1)[1], "")
                if isinstance(v, str) and v.startswith("os.environ/")
                else v
            )
            for k, v in provider_params.items()
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    response = litellm_embedding(
                        model=self.model, input=batch, **self.provider_params
                    )
                    data = sorted(response["data"], key=lambda d: d["index"])
                    vectors.extend(d["embedding"] for d in data)
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Embedding batch %d failed (attempt %d/%d): %s",
                        start // self.batch_size,
                        attempt + 1,
                        self.max_retries,
                        exc,
                    )
                    time.sleep(self.retry_delay)
            else:
                raise RuntimeError(
                    f"Embedding batch starting at {start} failed after "
                    f"{self.max_retries} attempts"
                ) from last_error
        return vectors
