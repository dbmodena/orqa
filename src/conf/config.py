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

    # Some datasets have a lot of columns. We might
    # limit our tasks to a smaller subset
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

    # Embedding computation
    embedding_batch_size: int = 64
    embedding_text_max_chars: int = 4000

    # Valentine matcher used for the pair schema gate
    sm_matcher: str = "coma"
    sm_matcher_kwargs: dict = field(
        default_factory=lambda: {"use_instances": False}
    )

    # Discovery budgets (formerly hardcoded in the pipeline)
    tokens_budget: int = 1_000_000
    max_datasets_to_process: int = 100

    verbose: bool = field(default=False)



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
    # automatically at load time — used only to compute per-column bad-token
    # counts (see ColumnStatistics.compute) shown to the query planner, which
    # decides for itself whether/how to clean a column via a `clean` plan step.
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


@dataclass
class MCPSearch:
    """
    Configuration for the MCP dataset-search server.

    The server exposes a keyword reverse index built from the normalized
    metadata, so that an agent can find the CSVs needed to answer a
    question. With the "elasticsearch" backend the reverse index lives in
    an Elasticsearch index (created at server startup when missing); with
    the "builtin" backend it is materialized under <data_path>/index/.
    """

    # Port used when the server runs in "port" mode (streamable HTTP).
    port: int

    # Interface to bind in "port" mode.
    host: str

    # Which reverse index implementation to use.
    backend: Literal["elasticsearch", "builtin"]

    # Elasticsearch endpoint (elasticsearch backend only). The
    # ELASTICSEARCH_URL env variable, when set, takes precedence.
    elasticsearch_url: str

    # Adaptive K for the plan judge's keyword-searchability check (see
    # orqa.agent.utility.keyword_searchability.check_keyword_searchability):
    # each plan's own K = round(len(plan_tables) * this coefficient), so a
    # plan joining more tables is held to a wider top-K net than a
    # single-table one, rather than every plan sharing one fixed K.
    keyword_search_top_k_coefficient: float = 5.0

    # Whether the PRE-planning retrievability check (see
    # orqa.agent.utility.keyword_suggestion.suggest_retrievable_keywords,
    # run once per table group before any LLM call) is allowed to ABORT a
    # group as "unretrievable_group" when no keyword combination can
    # surface every table within top-K. False (the default) still runs the
    # check and still hands the planner a verified keyword anchor when one
    # is found — only the abort-on-failure half is disabled, so a group
    # that can't be resolved just proceeds to planning without an anchor
    # (the old behavior) instead of being skipped outright.
    gate_unretrievable_groups: bool = False

    # Master on/off switch for the retrieval gate as a whole — BOTH the
    # PRE-planning keyword combination/suggestion check
    # (suggest_retrievable_keywords, above) AND the plan judge's reactive
    # keyword-searchability layer (check_keyword_searchability, applied per
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

    # Name of the per-city Elasticsearch index, derived from the data path
    es_index_name: str = field(init=False)

    # Where the materialized index is stored (builtin backend only)
    index_path: Path = field(init=False)
    index_filepath: Path = field(init=False)

    def __post_init__(self):
        self.gate_unretrievable_groups = _coerce_bool(
            self.gate_unretrievable_groups, "gate_unretrievable_groups"
        )
        self.retrieval_gate_enabled = _coerce_bool(
            self.retrieval_gate_enabled, "retrieval_gate_enabled"
        )


# How many judges each panel mode resolves to — the only two values a
# JudgePanel's judge_count is ever constructed with. "trio" caps at 3 even
# if judge_profiles.<panel> lists more (e.g. future failover profiles); a
# panel configured with fewer than the requested count simply uses however
# many it has (JudgePanel does no padding).
JUDGE_MODE_COUNTS: dict[str, int] = {"mono": 1, "trio": 3}


@dataclass
class Judges:
    """How many judges vote in each panel (see
    ``orqa.agent.agents.JudgePanel``) — independent of ``judge_profiles.plan``/
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
    mcp_search: MCPSearch
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
    # benchmark results, and the search index (statistics_path stays on
    # data_path — see __post_init__, it's produced by `clean`, not later
    # stages). Defaults to data_path (today's single-shared-root behavior)
    # when a workflow yaml doesn't set `write_path`; only needs to differ
    # when DATADIR is a read-only mount of already-crawled-and-cleaned data.
    write_path: Path

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

    filter_filenames_patterns: tuple[str, ...] = ()
    filter_column_patterns: tuple[str, ...] = ()
    # Default probing set for _select_best_separator (cleaning.py) — handles
    # the delimiters actually seen across portals (comma, EU semicolon, tab,
    # pipe) without requiring every workflow yaml to opt in explicitly.
    try_separators: tuple[str, ...] = (",", ";", "\t", "|")

    statistics_path: Path = field(init=False)

    # Benchmark folder, adjacent to candidates_discovery/ and metadata/.
    # Every artifact of a benchmark run (questions todo list, per-question
    # results) lives in the subfolder named after the programming language
    # kind selected in the workflow yaml (benchmark/pandas, benchmark/sql).
    benchmark_path: Path = field(init=False)
    benchmark_results_path: Path = field(init=False)
    questions_todo_filepath: Path = field(init=False)

    def __post_init__(self):
        self.crawled_datasets_path = (
            self.data_path / "datasets" / "crawling" / self.crawling.download_format
        )

        # Read-side: raw crawl + clean output, always on data_path.
        self.datasets_path = self.data_path / "datasets" / self.crawling.download_format
        self.metadata_path = self.data_path / "metadata"
        self.original_metadata_filepath = self.metadata_path / "metadata.json"
        self.statistics_path = self.data_path / "statistics"

        # Write-side: everything normalize-metadata onward, on write_path
        # (== data_path unless a workflow yaml overrides it).
        self.normalized_metadata_filepath = (
            self.write_path / "metadata" / "normalized_metadata.json"
        )

        self.mcp_search.index_path = self.write_path / "index"
        self.mcp_search.index_filepath = (
            self.mcp_search.index_path / "metadata_index.json"
        )
        # e.g. data/orqa/socrata/nyc -> "orqa-socrata-nyc" — always derived
        # from data_path (the target's identity), regardless of where
        # write_path points.
        self.mcp_search.es_index_name = "orqa-{}-{}".format(
            self.data_path.parent.name, self.data_path.name
        ).lower()

        self.benchmark_path = self.write_path / "benchmark"
        self.benchmark_results_path = (
            self.benchmark_path / self.statement_generation.kind.lower()
        )
        self.questions_todo_filepath = (
            self.benchmark_results_path / "questions_todo.json"
        )

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


def load_config(yaml_path: Path, data_path: Path) -> OrQAConfig:
    with open(yaml_path, "r") as file:
        parsed = yaml.safe_load(file)

    source = parsed["source"]

    # Optional top-level `write_path` — where this run's outputs (normalized
    # metadata, blend/embeddings index, candidates-discovery, logs,
    # benchmark results, search index) get written, instead of data_path.
    # Defaults to data_path (today's single-shared-root behavior) when
    # absent — only needed when DATADIR is a read-only mount of
    # already-crawled-and-cleaned data (see OrQAConfig.write_path).
    write_path_raw = parsed.get("write_path")
    write_path = Path(write_path_raw).expanduser() if write_path_raw else data_path
    write_path.mkdir(parents=True, exist_ok=True)

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
    )

    # setup the MCP dataset-search server (port comes from the
    # city's workflow yaml; defaults keep older yamls working)
    mcp_search_task = parsed["tasks"].get("mcp_search") or {}
    backend = mcp_search_task.get("backend", "elasticsearch")
    if backend not in ("elasticsearch", "builtin"):
        raise ValueError(
            f"tasks.mcp_search.backend must be 'elasticsearch' or "
            f"'builtin', got {backend!r}"
        )
    mcp_search = MCPSearch(
        port=int(mcp_search_task.get("port", 8765)),
        host=str(mcp_search_task.get("host", "127.0.0.1")),
        backend=backend,
        elasticsearch_url=str(
            mcp_search_task.get("elasticsearch_url", "http://localhost:9200")
        ),
        keyword_search_top_k_coefficient=float(
            mcp_search_task.get("keyword_search_top_k_coefficient", 5.0)
        ),
        gate_unretrievable_groups=mcp_search_task.get(
            "gate_unretrievable_groups", False
        ),
        retrieval_gate_enabled=mcp_search_task.get(
            "retrieval_gate_enabled", True
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
        mcp_search=mcp_search,
        judges=judges,
        filter_filenames_patterns=filter_filenames_patterns,
        filter_column_patterns=filter_column_patterns,
        try_separators=try_separators,
        data_path=data_path,
        write_path=write_path,
        datasets_format=crawling.download_format,
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
