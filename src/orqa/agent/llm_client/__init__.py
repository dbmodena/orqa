"""LiteLLM client base classes for the statement-generation agent.

- ``LLMClient`` — YAML-configured LiteLLM/Router client with retry handling.
- ``LLMClientStructured`` — adds Pydantic response-model loading, JSON repair,
  and structured-error retry loops on top of ``LLMClient``.
- ``EmbeddingClient`` — YAML-configured LiteLLM embedding wrapper (batching,
  retries, provider-param resolution), used by the candidates-discovery
  embedding pipeline.
"""
