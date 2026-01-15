from blend import BLEND
from blend.indexing import index_tables

from conf import OrQAConfig


def create_blend_index(cfg: OrQAConfig):
    cfg.indexing.index_folder_path.mkdir(parents=True, exist_ok=True)

    index = BLEND(
        cfg.indexing.index_database_path,
        clean_args=cfg.indexing.clean_args,
        xash_size=cfg.indexing.xash_size,
    )

    logging_path = cfg.logging_path.joinpath("indexing", "blend.log")
    logging_path.parent.mkdir(parents=True, exist_ok=True)

    cfg.tmp_path.mkdir(parents=True, exist_ok=True)

    index_tables(
        index,
        cfg.datasets_path,
        logfile_path=logging_path,
        log_stdout=cfg.indexing.verbose,
        load_opts=cfg.polars_opts.scan[cfg.crawling.download_format],
        max_workers=cfg.indexing.max_process_workers,
        max_queue_size=1000,
        # tmp_path=cfg.tmp_path,
    )
