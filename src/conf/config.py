import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import os
import yaml


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

    # list of bad values in order to prefilter on query generation
    bad_tokens: list

    # max number of tokens per natural languange response
    max_response_tokens: int

    # detected languages
    detected_languages:list 

    # Enable single-table query generation alongside cross-table generation
    enable_single_table: bool = False
    # Number of queries to generate per single table (None = fall back to cross-table count)
    single_table_query_count: Optional[int] = None

    @property
    def target_language(self) -> str:
        """
        The language of the open data portal (questions, keywords, metadata),
        e.g. "Spanish" for valencia, "Italian" for bologna, "English" for
        nyc/uk. First entry of tasks.query_generation.languages.
        """
        return (self.detected_languages or ["English"])[0]

    def __post_init__(self):
        # Validate enable_single_table is a bool, coerce common truthy/falsy values
        if not isinstance(self.enable_single_table, bool):
            if isinstance(self.enable_single_table, str):
                if self.enable_single_table.lower() in ("true", "1", "yes"):
                    self.enable_single_table = True
                elif self.enable_single_table.lower() in ("false", "0", "no"):
                    self.enable_single_table = False
                else:
                    raise ValueError(
                        f"enable_single_table must be a bool, got string '{self.enable_single_table}'"
                    )
            elif isinstance(self.enable_single_table, (int, float)):
                self.enable_single_table = bool(self.enable_single_table)
            else:
                raise TypeError(
                    f"enable_single_table must be a bool, got {type(self.enable_single_table).__name__}"
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

    # Name of the per-city Elasticsearch index, derived from the data path
    es_index_name: str = field(init=False)

    # Where the materialized index is stored (builtin backend only)
    index_path: Path = field(init=False)
    index_filepath: Path = field(init=False)


@dataclass
class OrQAConfig:
    source: Literal["ckan", "socrata", "ods"]
    seed: int
    crawling: Crawling
    candidates_discovery: CandidatesDiscovery
    statement_generation: StatementGeneration
    mcp_search: MCPSearch

    # Classical (BLEND) pipeline configuration; None when the workflow yaml
    # omits the blend/indexing blocks (semantic-only setups).
    indexing: Optional[Indexing]
    blend_opts: Optional[BLENDOpts]

    # Dataframe tools configurations for read/write ops
    polars_opts: PolarsOpts = field(init=False)
    pandas_opts: PandasOpts = field(init=False)

    # The base path where OrQA data is stored
    data_path: Path

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

    # Where the per-task-type ML skill markdowns are stored
    # (classification.md, regression.md, ...). Same as prompts_path:
    # not under data_path, but in the repository's configuration directory.
    skills_path: Path = field(init=False)

    # Where the LiteLLM and other LLM-related config things
    # are kept
    llm_config_path: Path = field(init=False)

    logging_path: Path = field(init=False)

    # The format with which datasets are stored locally
    datasets_format: str

    filter_filenames_patterns: tuple[str, ...] = ()
    filter_column_patterns: tuple[str, ...] = ()
    try_separators: tuple[str, ...] = ()

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

        self.datasets_path = self.data_path / "datasets" / self.crawling.download_format
        self.metadata_path = self.data_path / "metadata"
        self.original_metadata_filepath = self.metadata_path / "metadata.json"
        self.normalized_metadata_filepath = self.metadata_path / "normalized_metadata.json"

        self.mcp_search.index_path = self.data_path / "index"
        self.mcp_search.index_filepath = (
            self.mcp_search.index_path / "metadata_index.json"
        )
        # e.g. data/orqa/socrata/nyc -> "orqa-socrata-nyc"
        self.mcp_search.es_index_name = "orqa-{}-{}".format(
            self.data_path.parent.name, self.data_path.name
        ).lower()

        self.benchmark_path = self.data_path / "benchmark"
        self.benchmark_results_path = (
            self.benchmark_path / self.statement_generation.kind.lower()
        )
        self.questions_todo_filepath = (
            self.benchmark_results_path / "questions_todo.json"
        )

        self.logging_path = self.data_path / "log"
        self.prompts_path = Path(os.environ["ORQA_CONF"]) / "prompts"
        self.skills_path = Path(os.environ["ORQA_CONF"]) / "skills"
        self.llm_config_path =  Path(os.environ["ORQA_CONF"]) / "llm"
        self.statistics_path = self.data_path / "statistics"
        assert self.prompts_path.exists()
        assert self.skills_path.exists()
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

    # we will use a unique seed for random operations
    seed = int(parsed["seed"])

    # gets the type of queries we want to generate
    kind = parsed["tasks"]["query_generation"]["kind"]

    # gets the list of bad tokens
    bad_tokens = parsed["tasks"]["query_generation"]["bad_tokens"]

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
        index_folder_path = data_path / "blend"
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

    # setup the Candidates Discovery step
    candidates_discovery_task = parsed["tasks"]["candidates_discovery"]
    cand_disc_directory = data_path / "candidates_discovery"
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

    try_separators = cleaning_task.get("try_separators", ())
    if try_separators in (None, "..."):
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
        filter_filenames_patterns=filter_filenames_patterns,
        filter_column_patterns=filter_column_patterns,
        try_separators=try_separators,
        data_path=data_path,
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
