import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import os
import yaml


def _coerce_bool(value, field_name: str) -> bool:
    """Coerce a yaml-sourced value (bool, truthy/falsy string, or 0/1) to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"{field_name} must be a bool, got string '{value}'")
    if isinstance(value, (int, float)):
        return bool(value)
    raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")


@dataclass
class PolarsOpts:
    read: dict = field(
        default_factory=lambda: {
            "csv": {"ignore_errors": True},  # "encoding": "latin-1"},
            "parquet": {},
            "json": {},
        }
    )
    write: dict = field(default_factory=lambda: {"csv": {}, "parquet": {}, "json": {}})
    scan: dict = field(
        default_factory=lambda: {"csv": {"ignore_errors": True}, "parquet": {}}
    )


@dataclass
class PandasOpts:
    read: dict = field(
        default_factory=lambda: {
            "csv": {
                "sep": None,
                # "encoding": "latin-1",
                "encoding": "utf-8",  # avoid BOM in many files
                "encoding_errors": "ignore",
                "on_bad_lines": "skip",
                "engine": "python",
                "nrows": 1_000,
            },
            "parquet": {},
            "json": {"nrows": 1_000},
        }
    )

    write: dict = field(
        default_factory=lambda: {"csv": {"index": False}, "parquet": {}, "json": {}}
    )


@dataclass
class Crawling:
    """
    Configuration class for the Open Data datasets
    crawling stage.
    """

    max_datasets: int
    from_dataset_index: int
    batch_fetch_metadata: int
    accept_zip: bool
    download_format: Literal["csv", "parquet", "json"]
    max_resource_size: int  # CKAN-only
    max_rows_per_dataset: int  # Socrata-only
    batch_rows_per_dataset: int  # Socrata-only
    max_workers: int
    engine: Literal["pandas", "polars"] = field(default="pandas")
    search_filters: dict = field(default_factory=lambda: {})
    verbose: bool = field(default=True)


@dataclass
class Indexing:
    """
    Configuration class for the Indexing stage (classical BLEND pipeline).
    """

    index_folder_path: Path
    index_database_path: Path
    max_process_workers: int

    verbose: bool


@dataclass
class BLENDOpts:
    """
    Configuration class for BLEND (classical pipeline).
    """

    clean_args: dict
    xash_size: Literal[64, 128, 256, 512]
    max_cell_length: int


@dataclass
class CandidatesDiscovery:
    """
    Configuration class for the Candidates Discovery stage.

    Two alternative pipelines share this config:

    - classical (step ``candidates-discovery``): an agent proposes tasks per
      dataset, the BLEND index searches candidates, SLOTH + Valentine verify
      them after the fact.
    - semantic (step ``candidates-discovery-semantic``): datasets are embedded
      from their metadata; an HNSW index nominates neighbor pairs, Valentine
      gates each pair before an agent selects the tasks, and join-correlation
      tasks are verified with an actual join + correlation computation.
    """

    # We store also the seed datasets initially sampled
    seeds_datasets_path: Path

    # Where the discovered candidates are stored as
    # possible executable tasks
    proposed_tasks_path: Path

    # Where verified candidate tasks are stored (jsonlines)
    tasks_results_path: Path

    # The final candidates for the generation
    candidates_path: Path

    # How many datasets we will randomly sample as
    # seeds for the discovery task
    n_random_dataset_seeds: int

    # Many datasets have just 1 or very few rows
    # and usually are not very interesting
    min_dataset_height: int

    # Column-count ceiling for a table to be usable at all. A table with
    # MORE columns than this is dropped during indexing/embedding (same
    # treatment as a table with no rows or no columns) and never reaches
    # discovery or statement generation — the pipeline only ever works with
    # whole, integral tables, never a vertical slice. Tables at or under the
    # ceiling are shown to the agent in full.
    limit_to_n_columns: int

    # The number of randomly sampled rows we'll pass to the LLM
    # as snapshot of a dataset
    sample_size: int

    # Where to store the matches graph generated from the executed tasks
    matches_graph_path: Path

    # The number of distinct paths to generate for each seed datasets
    n_paths_for_dataset: int

    # The maximum path length for each path
    max_path_length: int

    # NOTE: pairs entering the graph already passed the schema gate with
    # these same thresholds, so the random-walk predicates only bite when
    # the yaml raises them above the gate values.
    sm_macro_avg_threshold: float

    sm_micro_avg_threshold: float

    # Which discovery pipeline this workflow uses: "blend" (classical
    # BLEND+SLOTH) or "semantic" (embeddings+HNSW+Valentine). Decides both
    # which implementation the candidates-discovery step runs and which
    # artifact lineage every step reads/writes (semantic artifacts carry a
    # _semantic suffix so the two methods never share state).
    method: Literal["blend", "semantic"] = "blend"

    # ── Classical (BLEND) pipeline ─────────────────────────────────────
    # Candidates per task, i.e. how many results we'll fetch with BLEND
    top_k_results_per_task: int = 10

    # Hash size for the QCR schema used by BLEND
    qcr_hash_size: int = 128

    # SLOTH overlap gate for the random walks (classical edges only; edges
    # from the semantic pipeline carry no overlap_ratio and pass the clause)
    overlap_ratio_threshold: float = 50.0

    # ── Semantic (embeddings) pipeline ─────────────────────────────────
    # Derived paths (set by load_config)
    embeddings_cache_path: Optional[Path] = None

    # Where the cluster assignments + 2D projection are persisted (see
    # embedding_discovery.clustering.compute_cluster_projection) — the
    # dataset-id -> cluster-id mapping and scatter-plot coordinates the
    # query browser's Stats page reads to render the cluster map.
    clusters_path: Optional[Path] = None

    # Neighbor nomination: datasets are clustered by metadata embedding
    # (cosine KMeans); a dataset's candidate pool is its cluster-mates.
    # n_clusters is derived from target_cluster_size so it scales with the
    # portal size instead of needing per-city tuning.
    target_cluster_size: int = 40

    # Soft overlap at cluster boundaries (cosine-similarity units): a point
    # also joins any OTHER cluster whose centroid it's within this margin
    # of, so cross-cluster joins/unions/correlations aren't cut off by a
    # hard partition.
    cluster_overlap_margin: float = 0.1

    # Safety cap so no single (possibly overlap-inflated) cluster explodes
    # the number of pairs a dataset must be checked against; oversized
    # clusters are randomly subsampled (seeded) down to this size.
    max_cluster_size: int = 60

    # Per-dataset fan-out: at most this many cluster-mates (highest cosine
    # similarity first, above the threshold) are tried per BFS visit.
    top_k_neighbors: int = 10
    cosine_similarity_threshold: float = 0.35

    # Join-correlation verification
    correlation_threshold: float = 0.5
    correlation_method: str = "pearson"
    min_joined_rows_for_correlation: int = 10

    # Embedding computation. Each table's metadata text is fitted to the
    # embedding model's own per-input token limit (see
    # embedding_discovery.embeddings.build_embedding_text).
    embedding_batch_size: int = 64
    # The model the metadata texts are embedded with. None: litellm.yaml's
    # embedding_model. Otherwise the Solr cores' rule (search.solr.embedding_model):
    # cohere.* / ocid1.* through OCI, anything else (bge-m3) through Ollama at
    # embedding_url (None: OrQA's embedding server, src/solr/embedder.py).
    embedding_model: Optional[str] = None
    embedding_url: Optional[str] = None

    # Valentine matcher used for the pair schema gate
    sm_matcher: str = "coma"
    sm_matcher_kwargs: dict = field(
        default_factory=lambda: {"use_instances": False}
    )

    # Discovery budgets (formerly hardcoded in the pipeline)
    tokens_budget: int = 1_000_000
    max_datasets_to_process: int = 100

    verbose: bool = field(default=False)



# Default trace folder, relative to write_path: beside the cleaning and
# indexing logs (OrQAConfig.logging_path = <write_path>/log).
DEFAULT_TRACE_DIR = "log/generation"
_TRACE_KEYS = ("enabled", "dir", "llm_io")
_TRACE_LLM_IO = ("all", "failed", "off")


@dataclass
class GenerationTraceConfig:
    """Where the generate-statements step records its trace (see
    ``orqa.utils.generation_trace``): one folder per run holding every
    question attempt with its gate and judge verdicts, every plan and
    plan-judge round, the generated code and its verdicts, every LLM call's
    outcome (the raw answer when it failed), Python logging warnings and
    errors, and the console output.

    From ``tasks.query_generation.trace``: ``enabled`` (default true) and
    ``dir`` — relative to ``write_path`` (default ``log/generation``) or
    absolute. ``path`` is that folder, resolved by ``load_config``; the trace
    writes a ``.gitignore`` into it, so it is never committed. ``llm_io``
    (default ``all``) picks the LLM calls written in full — system prompt,
    user message, parameters, reasoning, raw answer — to ``llm/<group>.txt``:
    ``all``, ``failed`` (the attempts that gave no usable answer) or ``off``.
    """

    enabled: bool = True
    path: Optional[Path] = None
    llm_io: str = "all"


def _load_trace_config(raw, write_path: Path) -> GenerationTraceConfig:
    """``tasks.query_generation.trace`` -> :class:`GenerationTraceConfig`."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            "tasks.query_generation.trace must be a mapping with "
            f"{', '.join(_TRACE_KEYS)}, got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - set(_TRACE_KEYS))
    if unknown:
        raise ValueError(
            f"tasks.query_generation.trace: unknown key(s) {unknown}; "
            f"expected {list(_TRACE_KEYS)}"
        )
    directory = Path(str(raw.get("dir") or DEFAULT_TRACE_DIR)).expanduser()
    if not directory.is_absolute():
        directory = write_path / directory
    llm_io = str(raw.get("llm_io") or "all").strip().lower()
    if llm_io not in _TRACE_LLM_IO:
        raise ValueError(
            f"tasks.query_generation.trace.llm_io must be one of {list(_TRACE_LLM_IO)}, "
            f"got {raw.get('llm_io')!r}"
        )
    return GenerationTraceConfig(
        enabled=_coerce_bool(raw.get("enabled", True), "tasks.query_generation.trace.enabled"),
        path=directory,
        llm_io=llm_io,
    )


# Run log defaults: <write_path>/log/orqa.log, 3 files of 25 MB in rotation.
DEFAULT_RUN_LOG_DIR = "log"
DEFAULT_RUN_LOG_FILE = "orqa.log"
DEFAULT_RUN_LOG_MAX_SIZE = "25MB"
DEFAULT_RUN_LOG_FILES = 3
_RUN_LOG_KEYS = ("enabled", "dir", "file", "max_size", "files", "level")
_RUN_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


@dataclass
class RunLogConfig:
    """The run log every ``src/main.py`` run appends to (see
    ``orqa.utils.run_log``): what the run printed plus its logging records,
    timestamped, in ``files`` files of at most ``max_bytes`` each, the oldest
    overwritten once they are all full.

    From the top-level ``logging`` block of a workflow yaml: ``enabled``
    (default true), ``dir`` — relative to ``write_path`` (default ``log``) or
    absolute — ``file`` (default ``orqa.log``), ``max_size`` per file
    (default ``25MB``), ``files`` in rotation (default 3) and ``level`` of
    OrQA's own logging records (default ``INFO``; libraries' records are
    kept from WARNING up). ``path`` is the live file, resolved by
    ``load_config``.
    """

    enabled: bool = True
    path: Optional[Path] = None
    max_bytes: int = 25 * 2**20
    files: int = DEFAULT_RUN_LOG_FILES
    level: str = "INFO"


def _load_run_log_config(raw, write_path: Path) -> RunLogConfig:
    """Top-level ``logging`` -> :class:`RunLogConfig`."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"logging must be a mapping with {', '.join(_RUN_LOG_KEYS)}, "
            f"got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - set(_RUN_LOG_KEYS))
    if unknown:
        raise ValueError(
            f"logging: unknown key(s) {unknown}; expected {list(_RUN_LOG_KEYS)}"
        )

    directory = Path(str(raw.get("dir") or DEFAULT_RUN_LOG_DIR)).expanduser()
    if not directory.is_absolute():
        directory = write_path / directory
    name = str(raw.get("file") or DEFAULT_RUN_LOG_FILE)
    if Path(name).name != name:
        raise ValueError(
            f"logging.file must be a file name, got {name!r}; put the folder in logging.dir"
        )

    size = raw.get("max_size", DEFAULT_RUN_LOG_MAX_SIZE)
    if isinstance(size, int) and not isinstance(size, bool):
        max_bytes = size
    else:
        text = str(size).strip().upper().replace(" ", "")
        if not re.fullmatch(r"\d+(K|M|G)B", text):
            raise ValueError(
                f"logging.max_size must be a size like 25MB (KB, MB or GB) or a "
                f"number of bytes, got {size!r}"
            )
        max_bytes = to_bytes(text)
    if max_bytes <= 0:
        raise ValueError(f"logging.max_size must be positive, got {size!r}")

    files = raw.get("files", DEFAULT_RUN_LOG_FILES)
    if isinstance(files, bool) or not isinstance(files, int) or files < 1:
        raise ValueError(f"logging.files must be a whole number >= 1, got {files!r}")

    level = str(raw.get("level") or "INFO").strip().upper()
    if level not in _RUN_LOG_LEVELS:
        raise ValueError(
            f"logging.level must be one of {list(_RUN_LOG_LEVELS)}, got {raw.get('level')!r}"
        )

    return RunLogConfig(
        enabled=_coerce_bool(raw.get("enabled", True), "logging.enabled"),
        path=directory / name,
        max_bytes=max_bytes,
        files=files,
        level=level,
    )


@dataclass
class StatementGeneration:
    """
    Configuration for the Statement Generation pipeline.

    An agent takes a set of datasets involved in a candidate task
    (join, union, or correlation) and generates executable statements
    (Pandas or SQL) to perform the operation, based on dataset metadata
    and a sample of rows.
    """

    # The kind of statements to generate: "PANDAS" or "SQL"
    kind: str
    # Path where the matches files reside
    query_candidates_path: Path
    # Path where the generated queries will reside
    queries_path: Path

    # Per-portal missing-value sentinel literals (e.g. "n/a", "not available",
    # "(null)") derived from blend.clean_args.bad_tokens. No longer applied
    # automatically at load time — used only to report the `missing tokens`
    # each column holds (see ColumnStatistics.compute) to the query planner,
    # which decides for itself whether/how to clean a column via a `clean`
    # plan step (a token can be a real category, e.g. a "-" sign flag).
    bad_tokens: list

    # max number of tokens per natural languange response
    max_response_tokens: int

    # detected languages
    detected_languages:list 

    # Enable single-table query generation alongside cross-table generation
    enable_single_table: bool = False
    # Number of queries to generate per single table (None = fall back to cross-table count)
    single_table_query_count: Optional[int] = None
    # Cap on how many cross-table candidate matches get processed for
    # multi-table query generation (None = process every match in
    # query_candidates(.json), i.e. no cap — the pre-existing behavior).
    # When set, a seeded random sample of this many matches is used, same
    # sampling idiom as single_table_query_count's dataset sampling, so
    # runs (and sibling workflows sharing the yaml seed) are reproducible.
    multi_table_query_count: Optional[int] = None

    # Question generation is split from planning: an agent first writes each
    # question FROM the verified retrieval keywords, and a gate (retriever
    # panel -> question judges) approves it BEFORE any plan exists; the
    # planner then plans the frozen, approved question (see
    # orqa.agent.question). `max_question_corrections` is how many
    # times a rejected question is rewritten before its slot is dropped
    # (default 5, see QuestionStage.MAX_QUESTION_CORRECTIONS).
    max_question_corrections: int = 5

    # The run's generation trace (tasks.query_generation.trace) — see
    # GenerationTraceConfig.
    trace: GenerationTraceConfig = field(default_factory=GenerationTraceConfig)

    @property
    def target_language(self) -> str:
        """
        The language of the open data portal (questions, keywords, metadata),
        e.g. "Spanish" for valencia, "Italian" for bologna, "English" for
        nyc/uk. First entry of tasks.query_generation.languages.
        """
        return (self.detected_languages or ["English"])[0]

    def __post_init__(self):
        self.enable_single_table = _coerce_bool(
            self.enable_single_table, "enable_single_table"
        )
        if (
            not isinstance(self.max_question_corrections, int)
            or isinstance(self.max_question_corrections, bool)
            or self.max_question_corrections < 0
        ):
            raise ValueError(
                "max_question_corrections must be a non-negative int, got "
                f"{self.max_question_corrections!r}"
            )

        # Validate single_table_query_count is a positive int or None
        if self.single_table_query_count is not None:
            if not isinstance(self.single_table_query_count, int) or isinstance(self.single_table_query_count, bool):
                raise TypeError(
                    f"single_table_query_count must be a positive int or None, "
                    f"got {type(self.single_table_query_count).__name__}"
                )
            if self.single_table_query_count <= 0:
                raise ValueError(
                    f"single_table_query_count must be a positive int, got {self.single_table_query_count}"
                )

        # Validate multi_table_query_count is a positive int or None
        if self.multi_table_query_count is not None:
            if not isinstance(self.multi_table_query_count, int) or isinstance(self.multi_table_query_count, bool):
                raise TypeError(
                    f"multi_table_query_count must be a positive int or None, "
                    f"got {type(self.multi_table_query_count).__name__}"
                )
            if self.multi_table_query_count <= 0:
                raise ValueError(
                    f"multi_table_query_count must be a positive int, got {self.multi_table_query_count}"
                )


# Where the yaml keys of the retired Elasticsearch backend (and the old
# keyword_backend/solr_core detour that silently fell back to it) are explained,
# with the steps to bring that backend back.
LEGACY_ELASTICSEARCH_README = "legacy/elasticsearch/README.md"

# The model of the cores' vectors: bge-m3 on OrQA's embedding server since
# 2026-11-06 (LakeGen's cores carried cohere.embed-v4.0).
SOLR_EMBEDDING_MODEL = "bge-m3"
# LakeGen's KNN filter queries: only the documents embedded from its
# metadata-v1 representation with that model are vector candidates (every
# document of the vector cores carries both fields).
LAKEGEN_KNN_FILTERS: tuple[str, ...] = (
    'representation_version:"metadata-v1"',
    f'embedding_model:"{SOLR_EMBEDDING_MODEL}"',
)


@dataclass
class RetrievalContractConfig:
    """The retrievability contract (see
    ``orqa.retrieval.gate``): a question is retrievable
    when a panel of retrievers — lexical gate, semantic, hybrid RRF —
    majority-agrees that it finds every gold table within the top K.
    """

    enabled: bool = True
    # Which retrievers vote. "llm_keywords" is silently skipped when
    # no LLM config is available to the caller building the panel — either
    # way min_agreement degrades to "all available".
    retrievers: tuple[str, ...] = ("lexical_question", "dense_question", "llm_keywords")
    min_agreement: int = 2
    top_k_per_table: int = 10
    # The retrieval pool a downstream retrieval agent (e.g. LakeGen) is
    # actually scored against — K never grows past this even for a
    # many-table plan.
    max_top_k: int = 20
    # The target for a SINGLE-table plan (exactly one plan table) —
    # literal rank 1 by default, deterministic rather than a window: with
    # only one target there's no reason to settle for less (see
    # ``orqa.retrieval.gate.build_contract``). Multi-table
    # plans are unaffected — they always use the top_k_per_table/max_top_k
    # window above.
    single_table_top_k: int = 1
    # How the gate and the keyword-anchor search match keywords. "all": every
    # keyword must match — Solr ``q.op=AND``, what LakeGen sends. "any"
    # (default): ``q.op=OR``, where a keyword the table lacks just adds
    # nothing.
    keyword_match: str = "any"
    # `hybrid_rrf` (add it to `retrievers`): RRF constant and how many results
    # each side contributes before fusing — LakeGen: k=60, and each branch
    # fetches top_k 20 x candidate_multiplier 5 = 100.
    hybrid_rrf_k: int = 60
    hybrid_rrf_depth: int = 100

    def __post_init__(self):
        from orqa.retrieval.panel import RETRIEVER_NAMES

        self.enabled = _coerce_bool(self.enabled, "retrieval_contract.enabled")
        if isinstance(self.retrievers, str):
            self.retrievers = (self.retrievers,)
        else:
            self.retrievers = tuple(self.retrievers)
        self.min_agreement = int(self.min_agreement)
        self.top_k_per_table = int(self.top_k_per_table)
        self.max_top_k = int(self.max_top_k)
        self.single_table_top_k = int(self.single_table_top_k)
        unknown = set(self.retrievers) - set(RETRIEVER_NAMES)
        if unknown:
            raise ValueError(
                f"retrieval_contract.retrievers has unknown name(s) {sorted(unknown)}; "
                f"valid: {list(RETRIEVER_NAMES)}"
            )
        if self.keyword_match not in ("any", "all"):
            raise ValueError(
                f"retrieval_contract.keyword_match must be 'any' or 'all', got {self.keyword_match!r}"
            )
        self.hybrid_rrf_k = int(self.hybrid_rrf_k)
        self.hybrid_rrf_depth = int(self.hybrid_rrf_depth)
        if self.hybrid_rrf_k < 1 or self.hybrid_rrf_depth < 1:
            raise ValueError("retrieval_contract.hybrid_rrf_k and hybrid_rrf_depth must be >= 1")
        if self.min_agreement < 1:
            raise ValueError("retrieval_contract.min_agreement must be >= 1")
        if self.top_k_per_table < 1:
            raise ValueError("retrieval_contract.top_k_per_table must be >= 1")
        if self.max_top_k < 1:
            raise ValueError("retrieval_contract.max_top_k must be >= 1")
        if self.single_table_top_k < 1:
            raise ValueError("retrieval_contract.single_table_top_k must be >= 1")


@dataclass
class SolrConfig:
    """The portal's Solr core (``tasks.search.solr``) on OrQA's own Solr
    (``solr.instance``) — the documents LakeGen queries. Retrieval searches it
    read-only (``solr.index``); the ``build-solr-core`` step builds it
    (``solr.build``).
    """

    # The core's name on OrQA's Solr: the portal's target id.
    core: str = ""
    # None: OrQA's own Solr, http://localhost:$ORQA_SOLR_PORT/solr (8984).
    url: Optional[str] = None
    # The DenseVectorField holding the 1024-d table embeddings LakeGen searches.
    vector_field: str = "table_embedding"
    # Filter queries of every KNN query, as LakeGen sends them.
    knn_filters: tuple[str, ...] = LAKEGEN_KNN_FILTERS
    # The model the core's vectors and the questions are embedded with, by
    # LakeGen's rule: cohere.* / ocid1.* through OCI (litellm.yaml's
    # embedding_model), anything else (bge-m3) through Ollama at
    # embedding_url. It is also the vectors' embedding_model label, so the
    # knn_filters' embedding_model must name it.
    embedding_model: str = SOLR_EMBEDDING_MODEL
    # Ollama URL of a non-OCI model; None: OrQA's own embedding server
    # (src/solr/embedder.py, http://127.0.0.1:$ORQA_OLLAMA_PORT).
    embedding_url: Optional[str] = None
    # The stored vectors' size: the question must be embedded at the same
    # size (bge-m3's 1024; cohere.embed-v4.0's output_dimensions) — OrQA's local 1536-d
    # embedding caches cannot query this field.
    embedding_dimensions: int = 1024
    # Seconds per request, and attempts on a connection error or a 5xx.
    timeout: float = 30.0
    retries: int = 3

    def __post_init__(self):
        self.core = str(self.core or "").strip()
        self.url = str(self.url or "").strip() or None
        self.vector_field = str(self.vector_field or "").strip()
        # The field name is interpolated into the {!knn f=...} local params.
        if not self.vector_field.replace("_", "").isalnum():
            raise ValueError(
                f"search.solr.vector_field must be a plain Solr field name, got {self.vector_field!r}"
            )
        if isinstance(self.knn_filters, str):
            self.knn_filters = (self.knn_filters,)
        else:
            self.knn_filters = tuple(str(f) for f in (self.knn_filters or ()))
        self.embedding_model = str(self.embedding_model or "").strip()
        self.embedding_url = str(self.embedding_url or "").strip() or None
        for spec in self.knn_filters:
            field_name, _, value = spec.partition(":")
            if field_name.strip() == "embedding_model" and value.strip().strip('"') != self.embedding_model:
                raise ValueError(
                    f"search.solr.knn_filters keep {value.strip()} vectors, but "
                    f"search.solr.embedding_model is {self.embedding_model!r}: a KNN query "
                    "would match none of the vectors it is compared with"
                )
        self.embedding_dimensions = int(self.embedding_dimensions)
        self.timeout = float(self.timeout)
        self.retries = int(self.retries)
        if self.embedding_dimensions < 1:
            raise ValueError("search.solr.embedding_dimensions must be >= 1")
        if self.timeout <= 0:
            raise ValueError("search.solr.timeout must be > 0")
        if self.retries < 1:
            raise ValueError("search.solr.retries must be >= 1")


@dataclass
class SearchConfig:
    """
    The reverse index shared by the pre-planning keyword-anchor search
    (orqa.retrieval.anchors), the retrievability gate
    (orqa.retrieval.gate) and the question stage's
    retrieval panel.

    The portal's core on OrQA's Solr (``solr`` below), the documents LakeGen
    queries — edismax keyword search and KNN on the core's own vectors
    (``solr.index``). It must answer when the run starts, or the run stops:
    there is no fallback index.
    """

    # The Solr core.
    solr: SolrConfig = field(default_factory=SolrConfig)

    # LRU cache of question embeddings for the KNN branch.
    query_embedding_cache_size: int = 2048

    # Adaptive K for the legacy keyword-searchability check that used to
    # gate the plan judge — superseded by retrieval_contract's own
    # top_k_per_table/max_top_k below. Kept only for
    # suggest_retrievable_keywords' pre-planning vocabulary HINT (see
    # orqa.retrieval.anchors): each hint search's own K =
    # round(len(plan_tables) * this coefficient).
    keyword_search_top_k_coefficient: float = 5.0

    # Whether the PRE-planning retrievability check (see
    # orqa.retrieval.anchors.suggest_retrievable_keywords,
    # run once per table group before any LLM call) is allowed to ABORT a
    # group as "unretrievable_group" when its bounded metadata-vocabulary
    # search finds no combination that surfaces every table within top-K.
    # False (the default) still runs the check and still hands the planner a verified keyword anchor when one
    # is found — only the abort-on-failure half is disabled, so a group
    # that can't be resolved just proceeds to planning without an anchor
    # (the old behavior) instead of being skipped outright.
    gate_unretrievable_groups: bool = False

    # Master on/off switch for the retrieval gate as a whole — BOTH the
    # PRE-planning keyword combination/suggestion check
    # (suggest_retrievable_keywords, above) AND the plan judge's reactive
    # retrievability layer (check_question_retrievability, applied per
    # plan in StatementOrchestrator._judge_plans). True (the default)
    # preserves today's behavior: whenever a reverse index is configured,
    # both mechanisms run. False disables both outright, regardless of
    # whether an index is otherwise available — StatementOrchestrator
    # implements this by treating its search index as unset (both
    # mechanisms already no-op to an automatic pass when the index is
    # None, so no other code path needs to change). Independent of
    # gate_unretrievable_groups above, which only controls whether a
    # PRE-planning miss aborts the group — that flag is meaningless once
    # this one is False, since the check it gates never runs at all.
    retrieval_gate_enabled: bool = True

    # The retrievability contract (retriever-panel majority vote) — see
    # RetrievalContractConfig and
    # orqa.retrieval.gate. Independent of
    # retrieval_gate_enabled above: that flag is the master kill switch for
    # BOTH mechanisms; this one only configures the contract's own knobs
    # once the gate is on.
    retrieval_contract: RetrievalContractConfig = field(
        default_factory=RetrievalContractConfig
    )

    def __post_init__(self):
        if not self.solr.core:
            raise ValueError("tasks.search.solr.core (the portal's core on OrQA's Solr) is required")
        self.gate_unretrievable_groups = _coerce_bool(
            self.gate_unretrievable_groups, "gate_unretrievable_groups"
        )
        self.retrieval_gate_enabled = _coerce_bool(
            self.retrieval_gate_enabled, "retrieval_gate_enabled"
        )
        self.query_embedding_cache_size = int(self.query_embedding_cache_size)
        if self.query_embedding_cache_size < 1:
            raise ValueError("query_embedding_cache_size must be >= 1")


# How many judges each panel mode resolves to — the only two values a
# JudgePanel's judge_count is ever constructed with. "trio" caps at 3 even
# if judge_profiles.<panel> lists more (e.g. future failover profiles); a
# panel configured with fewer than the requested count simply uses however
# many it has (JudgePanel does no padding).
JUDGE_MODE_COUNTS: dict[str, int] = {"mono": 1, "trio": 3}


@dataclass
class Judges:
    """How many judges vote in each panel (see
    ``orqa.agent.judges``) — independent of ``judge_profiles.plan``/
    ``judge_profiles.code`` in the LLM yaml, which lists the CANDIDATE judge
    models; this only caps how many of them are actually used per run.
    ``"mono"`` uses just the first configured judge (a single verdict, no
    voting); ``"trio"`` uses the first three (majority vote). Aggregation
    itself is generic over N (see JudgePanel._aggregate), so either mode
    works mechanically — "mono" is a deliberate quality/cost trade-off, not
    a degraded fallback.
    """

    plan_mode: Literal["mono", "trio"] = "trio"
    code_mode: Literal["mono", "trio"] = "trio"

    def __post_init__(self):
        for field_name in ("plan_mode", "code_mode"):
            value = getattr(self, field_name)
            if value not in JUDGE_MODE_COUNTS:
                raise ValueError(
                    f"tasks.judges.{field_name} must be one of "
                    f"{sorted(JUDGE_MODE_COUNTS)}, got {value!r}"
                )


@dataclass
class OrQAConfig:
    source: Literal["ckan", "socrata", "ods"]
    seed: int
    crawling: Crawling
    candidates_discovery: CandidatesDiscovery
    statement_generation: StatementGeneration
    search: SearchConfig
    judges: Judges

    # Classical (BLEND) pipeline configuration; None when the workflow yaml
    # omits the blend/indexing blocks (semantic-only setups).
    indexing: Optional[Indexing]
    blend_opts: Optional[BLENDOpts]

    # Dataframe tools configurations for read/write ops
    polars_opts: PolarsOpts = field(init=False)
    pandas_opts: PandasOpts = field(init=False)

    # The base path where OrQA data is READ from — raw crawled datasets,
    # cleaned datasets, and the original crawl metadata are always found
    # here, and are treated as a pre-existing, possibly read-only input.
    data_path: Path

    # Where everything this run WRITES lives instead: normalized metadata,
    # the blend/embeddings index, candidates-discovery artifacts, logs,
    # and the search index (statistics_path stays on
    # data_path — see __post_init__, it's produced by `clean`, not later
    # stages). Defaults to data_path (today's single-shared-root behavior)
    # when a workflow yaml doesn't set `write_path`; only needs to differ
    # when DATADIR is a read-only mount of already-crawled-and-cleaned data.
    write_path: Path

    # Where the crawler lands freshly downloaded datasets. The download
    # layout is fixed by the ulod bulk downloaders (they always write
    # <destination>/datasets/<format>), so this one ignores table_dir; the
    # `clean` step moves it aside into crawled_datasets_path.
    downloaded_datasets_path: Path = field(init=False)

    # Where all the datasets are stored once downloaded
    crawled_datasets_path: Path = field(init=False)

    # Where all the datasets are stored after cleaning
    datasets_path: Path = field(init=False)

    # Where the datasets metadata are stored once downloaded
    metadata_path: Path = field(init=False)

    # THe original and the normalized matadata filepaths
    original_metadata_filepath: Path = field(init=False)
    normalized_metadata_filepath: Path = field(init=False)

    # Where all the default prompts are stored
    # This path is not under the main data_path of OrQA,
    # but in the configuration directory already present
    # in the repository
    prompts_path: Path = field(init=False)

    # Where the LiteLLM and other LLM-related config things
    # are kept
    llm_config_path: Path = field(init=False)

    logging_path: Path = field(init=False)

    # The format with which datasets are stored locally
    datasets_format: str

    # Name of the folder under data_path holding the tables every stage
    # after `clean` reads (indexing, discovery, generation) —
    # the format subfolder is still appended, so "datasets" resolves to
    # <data_path>/datasets/<download_format>. Overridable per workflow yaml
    # (top-level `table_dir:`) when the tables to run on live in a
    # differently-named folder, e.g. a separately prepared
    # <data_path>/cleaned_datasets/<download_format>. Only the READ side
    # moves: the crawler always lands its downloads in the literal
    # "datasets" folder (see downloaded_datasets_path).
    table_dir: str = "datasets"

    filter_filenames_patterns: tuple[str, ...] = ()
    filter_column_patterns: tuple[str, ...] = ()
    # Default probing set for _select_best_separator (cleaning.py) — handles
    # the delimiters actually seen across portals (comma, EU semicolon, tab,
    # pipe) without requiring every workflow yaml to opt in explicitly.
    try_separators: tuple[str, ...] = (",", ";", "\t", "|")

    # The run log (top-level `logging` block, see RunLogConfig): one rotating
    # file under logging_path by default, written by every main.py run.
    run_log: RunLogConfig = field(default_factory=RunLogConfig)

    # The clean step's per-dataset column statistics — read-side, stays on
    # data_path (see write_statistics_path below for the write-side one).
    statistics_path: Path = field(init=False)

    # Candidates-discovery's OWN run statistics (generation_time_stats*.csv)
    # — a write-time artifact of the discovery run itself, not the clean
    # step's output, so it belongs on write_path despite the similar name.
    write_statistics_path: Path = field(init=False)

    def __post_init__(self):
        self.downloaded_datasets_path = (
            self.data_path / "datasets" / self.crawling.download_format
        )
        self.crawled_datasets_path = (
            self.data_path / "datasets" / "crawling" / self.crawling.download_format
        )

        # Read-side: raw crawl + clean output, always on data_path.
        self.datasets_path = (
            self.data_path / self.table_dir / self.crawling.download_format
        )
        self.metadata_path = self.data_path / "metadata"
        self.original_metadata_filepath = self.metadata_path / "metadata.json"
        self.statistics_path = self.data_path / "statistics"

        # Write-side: everything normalize-metadata onward, on write_path
        # (== data_path unless a workflow yaml overrides it).
        self.normalized_metadata_filepath = (
            self.write_path / "metadata" / "normalized_metadata.json"
        )
        self.write_statistics_path = self.write_path / "statistics"

        self.logging_path = self.write_path / "log"
        self.prompts_path = Path(os.environ["ORQA_CONF"]) / "prompts"
        self.llm_config_path =  Path(os.environ["ORQA_CONF"]) / "llm"
        assert self.prompts_path.exists()
        self.pandas_opts = PandasOpts()
        self.polars_opts = PolarsOpts()


def to_bytes(b: str) -> int:
    m = re.match(r"^(\d+)((K|M|G)B)$", b)
    assert m, f"Format of bytes invalid: {b}"

    n, e = m.group(1), m.group(2)

    match e:
        case "KB":
            return int(n) * (2**10)
        case "MB":
            return int(n) * (2**20)
        case "GB":
            return int(n) * (2**30)
        case _:
            raise ValueError(f"Invalid value: {b}")


# Search keys that no longer exist, and where their job went. Each raises
# rather than being ignored: a yaml still setting one expects a behaviour the
# run would silently not have.
_RETIRED_SEARCH_KEYS = {
    "elasticsearch_url": "the Elasticsearch backend was removed",
    "elasticsearch_managed": "the Elasticsearch backend was removed",
    "backend": "every portal is searched on its Solr core (search.solr); the builtin index was removed",
    "hybrid_search_enabled": "the builtin hybrid index was removed; the dense retriever is the core's KNN",
    "lexical_weight": "the builtin hybrid index was removed",
    "semantic_weight": "the builtin hybrid index was removed",
    "fusion_method": "the builtin hybrid index was removed; hybrid_rrf is retrieval_contract.hybrid_rrf_k",
    "rrf_k": "the builtin hybrid index was removed; hybrid_rrf is retrieval_contract.hybrid_rrf_k",
    "query_embedding_input_type": "the builtin hybrid index was removed",
}
_RETIRED_CONTRACT_KEYS = {
    # keyword_backend: solr fell back silently to the Elasticsearch imitation
    # whenever the backend was elasticsearch — which every Solr portal's was.
    "keyword_backend": "the gate searches the portal's Solr core (search.solr)",
    "solr_core": "the core is tasks.search.solr.core",
    "solr_url": "the Solr URL is tasks.search.solr.url",
    "search_fields": "the builtin index was removed; the core's own qf decides which fields match",
}


def _reject_retired_search_keys(search_task: dict) -> None:
    """Raise on a ``tasks.search`` key of a retired backend (Elasticsearch,
    the builtin index and its hybrid fusion) or of the old
    ``retrieval_contract`` Solr detour, naming where it went."""
    contract = search_task.get("retrieval_contract") or {}
    retired = [
        f"tasks.search.{key} ({why})"
        for key, why in _RETIRED_SEARCH_KEYS.items() if key in search_task
    ] + [
        f"tasks.search.retrieval_contract.{key} ({why})"
        for key, why in _RETIRED_CONTRACT_KEYS.items() if key in contract
    ]
    if retired:
        raise ValueError(
            f"Removed search key(s): {'; '.join(retired)}. Retrieval runs on the "
            "portal's Solr core (tasks.search.solr); delete the key(s). For the "
            f"retired Elasticsearch backend see {LEGACY_ELASTICSEARCH_README}."
        )


def load_config(yaml_path: Path, data_path: Path) -> OrQAConfig:
    with open(yaml_path, "r") as file:
        parsed = yaml.safe_load(file)

    source = parsed["source"]

    # Optional top-level `write_path` — where this run's outputs (normalized
    # metadata, blend/embeddings index, candidates-discovery, logs,
    # search index) get written, instead of data_path.
    # Defaults to data_path (today's single-shared-root behavior) when
    # absent — only needed when DATADIR is a read-only mount of
    # already-crawled-and-cleaned data (see OrQAConfig.write_path).
    write_path_raw = parsed.get("write_path")
    write_path = Path(write_path_raw).expanduser() if write_path_raw else data_path
    write_path.mkdir(parents=True, exist_ok=True)

    # Optional top-level `table_dir` — the folder under data_path holding
    # the tables to run on (the <download_format> subfolder is still
    # appended). Defaults to "datasets", i.e. the folder the `clean` step
    # writes; set it when the run should read a differently-named,
    # separately prepared table folder such as "cleaned_datasets".
    table_dir = str(parsed.get("table_dir") or "datasets")

    # we will use a unique seed for random operations
    seed = int(parsed["seed"])

    # gets the type of queries we want to generate
    kind = parsed["tasks"]["query_generation"]["kind"]

    # max number of natural language response tokens
    max_response_tokens = parsed["tasks"]["query_generation"]["max_response_tokens"]

    # setup the Crawling step
    crawling_task = parsed["tasks"]["crawling"]

    # candidates discovery thresholds
    # overlap_ratio_threshold = parsed["tasks"]["candidates_discovery"]["overlap_ratio_threshold"]
    # sm_macro_avg_threshold = parsed["tasks"]["candidates_discovery"]["sm_macro_avg_threshold"]
    # sm_micro_avg_threshold = parsed["tasks"]["candidates_discovery"]["sm_micro_avg_threshold"]

    # For the Crawling stage, set to default values to deal with CKAN/Socrata differences
    # CKAN-only parameters
    crawling_task["max_resource_size"] = crawling_task.get("max_resource_size", "1MB")
    crawling_task["max_resource_size"] = to_bytes(crawling_task["max_resource_size"])
    crawling_task["batch_fetch_metadata"] = crawling_task.get(
        "batch_fetch_metadata", 100
    )

    # Socrata-only parameters
    crawling_task["max_rows_per_dataset"] = crawling_task.get(
        "max_rows_per_dataset", 100_000
    )
    crawling_task["batch_rows_per_dataset"] = crawling_task.get(
        "batch_rows_per_dataset", 10_000
    )

    crawling = Crawling(**crawling_task)

    # setup the classical (BLEND) pipeline pieces, when configured
    indexing = None
    if "indexing" in parsed["tasks"]:
        index_folder_path = write_path / "blend"
        indexing = Indexing(
            **parsed["tasks"]["indexing"],
            index_folder_path=index_folder_path,
            index_database_path=index_folder_path / "index.db",
        )

    blend_opts = None
    if "blend" in parsed:
        parsed["blend"]["clean_args"]["bad_tokens"] = tuple(
            parsed["blend"]["clean_args"].get("bad_tokens", [])
        )
        blend_opts = BLENDOpts(**parsed["blend"])

    # Derive the missing-value sentinel list shown to the query planner from
    # BLEND's cleaning tokens, excluding the tail of low-signal-but-VALID
    # values BLEND also filters for indexing purposes only (real data, not
    # missing-value indicators — e.g. "0"/"yes"/"no" are legitimate answers).
    _BLEND_LOW_SIGNAL_TOKENS = {"0", "0.0", "1", "2", "no", "yes", "ny"}
    blend_bad_tokens = list(
        parsed.get("blend", {}).get("clean_args", {}).get("bad_tokens", [])
    )
    bad_tokens = [
        t for t in blend_bad_tokens
        if str(t).strip().lower() not in _BLEND_LOW_SIGNAL_TOKENS
    ]

    # setup the Candidates Discovery step
    candidates_discovery_task = parsed["tasks"]["candidates_discovery"]
    cand_disc_directory = write_path / "candidates_discovery"
    cand_disc_directory.mkdir(exist_ok=True)

    # The workflow yaml decides the discovery method; the semantic method
    # keeps a fully separate artifact lineage (…_semantic files) so the two
    # methods never resume from or overwrite each other's state.
    discovery_method = candidates_discovery_task.get("method", "blend")
    if discovery_method not in ("blend", "semantic"):
        raise ValueError(
            f"tasks.candidates_discovery.method must be 'blend' or "
            f"'semantic', got {discovery_method!r}"
        )
    sfx = "_semantic" if discovery_method == "semantic" else ""
    seeds_datasets_path = cand_disc_directory / f"seeds_datasets{sfx}.json"
    proposed_tasks_path = cand_disc_directory / f"proposed_tasks{sfx}.json"
    tasks_results_path = cand_disc_directory / f"tasks_results{sfx}.json"
    matches_graph_path = cand_disc_directory / f"matches_graph{sfx}.gml"
    final_candidates_path = (
        cand_disc_directory / f"final_generation_candidates{sfx}.json"
    )

    # queries candidates path
    query_candidates_path = cand_disc_directory / f"query_candidates{sfx}.json"
    # generated queries path
    queries_path = cand_disc_directory / f"generated_queries{sfx}.json"

    candidates_discovery = CandidatesDiscovery(
        seeds_datasets_path,
        proposed_tasks_path,
        tasks_results_path,
        final_candidates_path,
        matches_graph_path=matches_graph_path,
        embeddings_cache_path=cand_disc_directory / "embeddings.npz",
        clusters_path=cand_disc_directory / "clusters.json",
        **candidates_discovery_task,
    )

    # Read optional single-table generation parameters from config
    enable_single_table = parsed["tasks"]["query_generation"].get(
        "enable_single_table", False
    )
    single_table_query_count = parsed["tasks"]["query_generation"].get(
        "single_table_query_count", None
    )
    multi_table_query_count = parsed["tasks"]["query_generation"].get(
        "multi_table_query_count", None
    )
    detected_languages  = parsed["tasks"]["query_generation"].get(
        "languages", ["English"]
    )
    query_generation_task = parsed["tasks"]["query_generation"]
    # The old flow (the planner writes question and steps together) is gone; a
    # yaml that still switches the stage off would otherwise be silently
    # ignored while its author expects the old flow.
    if "question_stage" in query_generation_task and not _coerce_bool(
        query_generation_task["question_stage"], "question_stage"
    ):
        raise ValueError(
            "tasks.query_generation.question_stage was removed: the question "
            "stage is always on — delete the key"
        )
    max_question_corrections = parsed["tasks"]["query_generation"].get(
        "max_question_corrections", 5
    )
    statement_generation = StatementGeneration(
        kind=kind,
        query_candidates_path=query_candidates_path,
        queries_path=queries_path,
        bad_tokens=bad_tokens,
        max_response_tokens=max_response_tokens,
        detected_languages=detected_languages,
        enable_single_table=enable_single_table,
        single_table_query_count=single_table_query_count,
        multi_table_query_count=multi_table_query_count,
        max_question_corrections=max_question_corrections,
        trace=_load_trace_config(query_generation_task.get("trace"), write_path),
    )

    # The portal's Solr core and the retrievability gate.
    search_task = parsed["tasks"].get("search") or {}
    _reject_retired_search_keys(search_task)
    search = SearchConfig(
        solr=SolrConfig(**(search_task.get("solr") or {})),
        query_embedding_cache_size=int(
            search_task.get("query_embedding_cache_size", 2048)
        ),
        keyword_search_top_k_coefficient=float(
            search_task.get("keyword_search_top_k_coefficient", 5.0)
        ),
        gate_unretrievable_groups=search_task.get(
            "gate_unretrievable_groups", False
        ),
        retrieval_gate_enabled=search_task.get(
            "retrieval_gate_enabled", True
        ),
        retrieval_contract=RetrievalContractConfig(
            **{
                key: value
                for key, value in (search_task.get("retrieval_contract") or {}).items()
                # retired with the sibling checks — older yamls may still set it
                if key != "max_residual_siblings"
            }
        ),
    )

    # How many judges vote on each plan/code panel (see JudgePanel) — the
    # judge_profiles.plan/code lists in the LLM yaml stay the pool of
    # CANDIDATE models; this only caps how many of them are used per run.
    judges_task = parsed["tasks"].get("judges") or {}
    judges = Judges(
        plan_mode=judges_task.get("plan_mode", "trio"),
        code_mode=judges_task.get("code_mode", "trio"),
    )

    cleaning_task = parsed["tasks"].get("cleaning", {})
    filter_filenames_patterns = cleaning_task.get("filter_filenames_patterns", ())
    if filter_filenames_patterns in (None, "..."):
        filter_filenames_patterns = ()
    elif isinstance(filter_filenames_patterns, str):
        filter_filenames_patterns = (filter_filenames_patterns,)
    else:
        filter_filenames_patterns = tuple(filter_filenames_patterns)

    filter_column_patterns = cleaning_task.get("filter_column_patterns", ())
    if filter_column_patterns in (None, "..."):
        filter_column_patterns = ()
    elif isinstance(filter_column_patterns, str):
        filter_column_patterns = (filter_column_patterns,)
    else:
        filter_column_patterns = tuple(filter_column_patterns)

    # Sentinel so a yaml key genuinely absent (-> broad default probing set)
    # is distinguishable from one explicitly set empty/null (-> opt out of
    # probing entirely) — `.get(key, default)` alone can't tell those apart.
    _unset = object()
    try_separators = cleaning_task.get("try_separators", _unset)
    if try_separators is _unset:
        try_separators = OrQAConfig.__dataclass_fields__["try_separators"].default
    elif try_separators in (None, "..."):
        try_separators = ()
    elif isinstance(try_separators, str):
        try_separators = (try_separators,)
    else:
        try_separators = tuple(try_separators)

    orqa_cfg = OrQAConfig(
        source=source,
        seed=seed,
        crawling=crawling,
        indexing=indexing,
        blend_opts=blend_opts,
        candidates_discovery=candidates_discovery,
        statement_generation=statement_generation,
        search=search,
        judges=judges,
        filter_filenames_patterns=filter_filenames_patterns,
        filter_column_patterns=filter_column_patterns,
        try_separators=try_separators,
        data_path=data_path,
        write_path=write_path,
        datasets_format=crawling.download_format,
        table_dir=table_dir,
        run_log=_load_run_log_config(parsed.get("logging"), write_path),
    )

    for engine in ["pandas", "polars"]:
        if engine not in parsed:
            continue
        engine_cfg = parsed[engine]
        for op in ["read", "write"]:
            if op not in engine_cfg:
                continue
            for format in ["csv", "parquet", "json"]:
                match (engine, op):
                    case ("pandas", "read"):
                        orqa_cfg.pandas_opts.read[format] = orqa_cfg.pandas_opts.read[
                            format
                        ] | engine_cfg[op].pop(format, {})
                    case ("pandas", "write"):
                        orqa_cfg.pandas_opts.write[format] = orqa_cfg.pandas_opts.write[
                            format
                        ] | engine_cfg[op].pop(format, {})
                    case ("polars", "read"):
                        orqa_cfg.polars_opts.read[format] = orqa_cfg.polars_opts.read[
                            format
                        ] | engine_cfg[op].pop(format, {})
                    case ("pandas", "read"):
                        orqa_cfg.polars_opts.read[format] = orqa_cfg.polars_opts.read[
                            format
                        ] | engine_cfg[op].pop(format, {})

    return orqa_cfg
