"""Search-keywords agent — the LLM keyword extraction behind the
retrievability gate's ``llm_keywords`` retriever
(orqa.benchmark.retrieval_panel.make_llm_keyword_extractor).

One single-shot call: given a question ALONE (never the table it came
from), extract the keywords a portal user would search with, in the
question's own language.
"""
from pathlib import Path

from ..llm_client.LLMClientStructured import LLMClientStructured
from ..prompting import BenchmarkSearchKeywordsPrompt


class _SearchKeywordsClient(LLMClientStructured):
    def __init__(self, config_path: Path):
        super().__init__(config_path, response_model="benchmark_search_keywords")


class SearchKeywordsAgent:
    """Question -> retrieval keywords, pinned to the
    ``benchmark_search_keywords`` response_model config key in litellm.yaml.
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._keywords_client = _SearchKeywordsClient(config_path)
        self._keywords_prompt = BenchmarkSearchKeywordsPrompt()

    def generate_keywords(self, question: str) -> tuple[dict, dict]:
        """question -> {detected_language, keywords}. (result, usage)."""
        prompt = self._keywords_prompt.update(question=question)
        result, usage = self._keywords_client.complete(prompt, root_key=None)
        return result or {}, usage
