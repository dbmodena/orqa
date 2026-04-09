import os
import re
from pathlib import Path
from typing import Any

from fake_useragent import UserAgent
from ulod.bulk.ckan import CKANDownloadConfig, ckan_download_datasets
from ulod.ckan import CanadaCKAN
from ulod.ckan.uk import UKCKAN
from ulod.ckan.italy import ModenaCKAN

from ulod.socrata import NYCSocrata
from ulod.bulk.socrata import SocrataDownloadConfig, socrata_download_datasets

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


def _uk_filter_resource_metadata(metadata: dict[str, Any]) -> bool:
    if metadata["format"].lower() not in ["csv"]:
        return False
    # TODO: UK tarif datasets have many many many different
    # versions for the same data, thus is not easy to work
    # on them for OrQA aim. For now, we skip them. In future,
    # we might be interested into more fine-grained tasks
    # about selecting some specific version of a dataset.
    if metadata["name"] and re.match(r"v\d+", metadata["name"]):
        return False

    # NOTE: UK Contracts Finder datasets have a very bad formatting,
    # something that have maybe taken from XML files to CSV without a
    # proper handling. We can't work on them, since their informative
    # content is not easy to catch.
    if metadata["name"] and re.match(r"Contracts Finder", metadata["name"]):
        return False

    # related to the tarif datasets
    # if "ODS" in metadata["name"]:
    #     return False
    return True


def rename_crawled_datasets_folder(cfg: OrQAConfig):
    import shutil

    shutil.move(cfg.datasets_path, cfg.crawled_datasets_path)


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
        max_resource_size=cfg.crawling.max_resource_size,
        max_workers=cfg.crawling.max_workers,
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
        # search_filters=cfg.crawling.search_filters,
        filter_resource_metadata=_uk_filter_resource_metadata,
        download_format=cfg.crawling.download_format,
        http_headers=headers,
        max_resource_size=cfg.crawling.max_resource_size,
        max_workers=cfg.crawling.max_workers,
        verbose=cfg.crawling.verbose,
    )

    # ckan_download_datasets(download_cfg, uk)
    rename_crawled_datasets_folder(cfg)


def crawl_modena(cfg: OrQAConfig):
    download_destination = cfg.data_path
    download_destination.mkdir(parents=True, exist_ok=True)

    client = ModenaCKAN(headers=headers)

    download_cfg = CKANDownloadConfig(
        download_destination,
        max_datasets=cfg.crawling.max_datasets,
        from_dataset_index=cfg.crawling.from_dataset_index,
        batch_fetch_metadata=cfg.crawling.batch_fetch_metadata,
        # search_filters=cfg.crawling.search_filters,
        filter_resource_metadata=_uk_filter_resource_metadata,
        download_format=cfg.crawling.download_format,
        http_headers=headers,
        max_resource_size=cfg.crawling.max_resource_size,
        max_workers=cfg.crawling.max_workers,
        verbose=cfg.crawling.verbose,
    )

    ckan_download_datasets(download_cfg, client)


def crawl_nyc(cfg: OrQAConfig):
    download_destination = cfg.data_path
    download_destination.mkdir(parents=True, exist_ok=True)

    nyc = NYCSocrata(os.environ["SOCRATA_NYC_APP_TOKEN"])

    download_cfg = SocrataDownloadConfig(
        download_destination,
        max_datasets=cfg.crawling.max_datasets,
        from_dataset_index=cfg.crawling.from_dataset_index,
        download_format=cfg.crawling.download_format,
        save_metadata=True,
        engine=cfg.crawling.engine,
        # cast_datatypes=True,
        max_rows_per_dataset=cfg.crawling.max_rows_per_dataset,
        batch_rows_per_dataset=cfg.crawling.batch_rows_per_dataset,
        max_workers=cfg.crawling.max_workers,
        verbose=cfg.crawling.verbose,
    )

    socrata_download_datasets(download_cfg, nyc)



from ulod.ckan.italy import ItalyCKAN, FerraraCKAN, MilanoCKAN
from ulod.ckan.spain import MadridCKAN
from ulod.bulk.ods import ODSDownloadConfig, ods_download_datasets
from ulod.ods.italy import BolognaODS
from ulod.ods.france import ParisODS


connection_pool_kw = {"redirect": True, "timeout": 5}


def _csv_only_filter_resource_metadata(metadata: dict[str, Any]) -> bool:
    if metadata["format"].lower() not in ["csv"]:
        return False
    return True


def crawl_paris(cfg: OrQAConfig):
    """opendata.paris.fr — French, OpenDataSoft backend."""
    download_destination = cfg.data_path
    download_destination.mkdir(parents=True, exist_ok=True)

    client = ParisODS(headers=headers, connection_kw=connection_pool_kw)
    print(cfg.crawling.max_datasets)
    download_cfg = ODSDownloadConfig(
        download_destination,
        max_datasets=cfg.crawling.max_datasets,
        from_dataset_index=cfg.crawling.from_dataset_index,
        batch_fetch_metadata=cfg.crawling.batch_fetch_metadata,
        download_format=cfg.crawling.download_format,
        http_headers=headers,
        save_with_resource_name=True,
        connection_pool_kw=connection_pool_kw,
        max_workers=cfg.crawling.max_workers,
        verbose=cfg.crawling.verbose,
    )

    ods_download_datasets(download_cfg, client)


def crawl_bologna(cfg: OrQAConfig):
    """opendata.comune.bologna.it — Italian, OpenDataSoft backend."""
    download_destination = cfg.data_path
    download_destination.mkdir(parents=True, exist_ok=True)

    client = BolognaODS(headers=headers, connection_kw=connection_pool_kw)

    download_cfg = ODSDownloadConfig(
        download_destination,
        max_datasets=cfg.crawling.max_datasets,
        from_dataset_index=cfg.crawling.from_dataset_index,
        batch_fetch_metadata=cfg.crawling.batch_fetch_metadata,
        download_format=cfg.crawling.download_format,
        http_headers=headers,
        save_with_resource_name=True,
        connection_pool_kw=connection_pool_kw,
        max_workers=cfg.crawling.max_workers,
        verbose=cfg.crawling.verbose,
    )

    ods_download_datasets(download_cfg, client)


def crawl_madrid(cfg: OrQAConfig):
    """datos.madrid.es — Spanish, CKAN backend."""
    download_destination = cfg.data_path
    download_destination.mkdir(parents=True, exist_ok=True)

    client = MadridCKAN(headers=headers, connection_kw=connection_pool_kw)

    download_cfg = CKANDownloadConfig(
        download_destination,
        max_datasets=cfg.crawling.max_datasets,
        from_dataset_index=cfg.crawling.from_dataset_index,
        batch_fetch_metadata=cfg.crawling.batch_fetch_metadata,
        filter_resource_metadata=_csv_only_filter_resource_metadata,
        download_format=cfg.crawling.download_format,
        http_headers=headers,
        save_with_resource_name=True,
        connection_pool_kw=connection_pool_kw,
        max_resource_size=cfg.crawling.max_resource_size,
        max_workers=cfg.crawling.max_workers,
        verbose=cfg.crawling.verbose,
    )

    ckan_download_datasets(download_cfg, client)