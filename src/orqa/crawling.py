import os
import re
from pathlib import Path
from typing import Any

from fake_useragent import UserAgent
from ulod.bulk.ckan import CKANDownloadConfig, ckan_download_datasets
from ulod.ckan import UKCKAN, CanadaCKAN

from conf import OrQAConfig

ua = UserAgent()
headers = {"User-Agent": ua.firefox}


def _canada_filter_resource_metadata(metadata: dict[str, Any]) -> bool:
    if metadata["format"].lower() not in ["csv"]:
        return False

    if "language" in metadata and "en" not in metadata["language"]:
        return False

    if re.search(r"\(CSV.+\)", metadata["name"], re.DOTALL) is not None:
        return False

    return True


def _canada_create_links_for_unzipped_folder(datasets_path: Path, cfg: OrQAConfig):
    for dataset_file in os.listdir(datasets_path):
        dataset_path = datasets_path.joinpath(dataset_file)
        if dataset_path.is_dir():
            subfiles = os.listdir(dataset_path)
            if len(subfiles) != 2:
                continue

            fd, fmd = subfiles
            fd, fmd = (fd, fmd) if "_MetaData" in fmd else (fmd, fd)

            # TODO: now for simplicity we assume to have only CSV files,
            # but for large canada collections it would be better to support
            # also parquet option
            #
            # create symlink
            symlink_path = datasets_path.joinpath(fd)

            if not symlink_path.exists():
                symlink_path.symlink_to(dataset_path.joinpath(fd))


def crawl_canada(cfg: OrQAConfig):
    download_destination = cfg.data_path
    download_destination.mkdir(parents=True, exist_ok=True)

    canada = CanadaCKAN(headers=headers)

    download_cfg = CKANDownloadConfig(
        download_destination,
        max_datasets=cfg.crawling.max_datasets,
        from_dataset_index=cfg.crawling.from_dataset_index,
        batch_fetch_metadata=cfg.crawling.batch_fetch_metadata,
        filter_resource_metadata=_canada_filter_resource_metadata,
        download_format=cfg.crawling.download_format,
        http_headers=headers,
        read_dataset_kwargs=cfg.pandas_opts.read[cfg.crawling.download_format],
        save_dataset_kwargs=cfg.pandas_opts.write[cfg.crawling.download_format],
        accept_zip=cfg.crawling.accept_zip,
        engine=cfg.crawling.engine,
        max_resource_size=cfg.crawling.max_resource_size,
        max_process_workers=cfg.crawling.max_process_workers,
        max_thread_workers=cfg.crawling.max_thread_workers,
        verbose=cfg.crawling.verbose,
    )

    ckan_download_datasets(download_cfg, canada)

    _canada_create_links_for_unzipped_folder(
        download_destination.joinpath("datasets", cfg.crawling.download_format), cfg
    )


def crawl_uk(cfg: OrQAConfig):
    download_destination = cfg.data_path
    download_destination.mkdir(parents=True, exist_ok=True)

    uk = UKCKAN(headers=headers)

    download_cfg = CKANDownloadConfig(
        download_destination,
        max_datasets=cfg.crawling.max_datasets,
        from_dataset_index=cfg.crawling.from_dataset_index,
        batch_fetch_metadata=cfg.crawling.batch_fetch_metadata,
        download_format=cfg.crawling.download_format,
        http_headers=headers,
        read_dataset_kwargs=cfg.pandas_opts.read[cfg.crawling.download_format],
        save_dataset_kwargs=cfg.pandas_opts.write[cfg.crawling.download_format],
        accept_zip=cfg.crawling.accept_zip,
        engine=cfg.crawling.engine,
        max_resource_size=cfg.crawling.max_resource_size,
        max_process_workers=cfg.crawling.max_process_workers,
        max_thread_workers=cfg.crawling.max_thread_workers,
        verbose=cfg.crawling.verbose,
    )

    ckan_download_datasets(download_cfg, uk)
