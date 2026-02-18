from blend import BLEND
from blend.indexing import index_tables

from conf import OrQAConfig


def create_blend_index(cfg: OrQAConfig):
    cfg.indexing.index_folder_path.mkdir(parents=True, exist_ok=True)

    index = BLEND(
        cfg.indexing.index_database_path,
        clean_args=cfg.blend_opts.clean_args,
        xash_size=cfg.blend_opts.xash_size,
        max_cell_length=cfg.blend_opts.max_cell_length,
    )

    logging_path = cfg.logging_path.joinpath("indexing", "blend.log")
    logging_path.parent.mkdir(parents=True, exist_ok=True)

    cfg.tmp_path.mkdir(parents=True, exist_ok=True)

    print(" INDEXING TABLES WITH BLEND ".center(100, "="))
    index_tables(
        index,
        cfg.datasets_path,
        logfile_path=logging_path,
        log_stdout=cfg.indexing.verbose,
        load_opts=cfg.polars_opts.scan[cfg.crawling.download_format],
        max_workers=cfg.indexing.max_process_workers,
        max_queue_size=1000,
        # tmp_path=cfg.tmp_path
    )
    print(" INDEXING COMPLETED ".center(100, "="))
