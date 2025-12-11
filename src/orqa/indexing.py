from blend.blend import BLEND
import blend
from conf import OrQAConfig


def create_blend_index(cfg: OrQAConfig):
    cfg.indexing.index_folder_path.mkdir(parents=True, exist_ok=True)

    blend = BLEND(
        cfg.indexing.index_database_path,
        clean_function_args=cfg.indexing.clean_func_args,
        xash_size=cfg.indexing.xash_size,
    )

    blend.create_index(
        cfg.datasets_path,
        scan_table_opts=cfg.polars_opts.scan[cfg.crawling.download_format],
        max_workers=cfg.indexing.max_process_workers,
        verbose=cfg.indexing.verbose,
    )

    top_k = cfg.indexing.top_k
