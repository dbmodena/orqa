from blend.blend import BLEND

from conf import OrQAConfig


def create_blend_index(cfg: OrQAConfig):
    cfg.indexing.index_folder_path.mkdir(parents=True, exist_ok=True)

    index = BLEND(
        cfg.indexing.index_database_path,
        clean_function_args=cfg.indexing.clean_func_args,
        xash_size=cfg.indexing.xash_size,
    )

    index.create_index(
        cfg.datasets_path,
        scan_table_opts=cfg.polars_opts.scan[cfg.crawling.download_format],
        max_workers=cfg.indexing.max_process_workers,
        verbose=cfg.indexing.verbose,
    )
