import re
import shutil
import duckdb
from pathlib import Path

_SAFE_DATASET_ID = re.compile(r'^[A-Za-z_][A-Za-z0-9_ -]*$')

_ID_COLUMN_CANDIDATES = ("dataset_name", "dataset_id", "source", "table_name", "file")


def has_unsafe_id(dataset_id: str) -> bool:
    if not dataset_id or not dataset_id.strip():
        return True
    return not _SAFE_DATASET_ID.match(dataset_id)


def _get_backup_path(db_path: Path) -> Path:
    return db_path.with_name(db_path.stem + "_original" + db_path.suffix)


def _get_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    return [row[0] for row in conn.execute("SHOW TABLES").fetchall()]


def _find_id_columns(conn: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    """
    Return all columns to scan for illegal IDs.
    Priority: known candidate names first, then all VARCHAR columns as fallback.
    """
    rows = conn.execute(f'DESCRIBE "{table}"').fetchall()
    # rows: (column_name, column_type, ...)
    columns = [(row[0], row[1].upper()) for row in rows]
    column_names = [name for name, _ in columns]

    known = [c for c in _ID_COLUMN_CANDIDATES if c in column_names]
    if known:
        return known

    # fallback: any string-like column could hold dataset IDs
    return [name for name, dtype in columns if "VARCHAR" in dtype or "TEXT" in dtype]


def purge_illegal_datasets(db_path: Path, dry_run: bool = False) -> int:
    """
    Remove rows whose dataset ID fails the whitelist check.

    On first run: backs up the original DB, then purges.
    On subsequent runs: backup already exists → skips purge entirely.

    Returns the total number of purged entries (0 if skipped or dry run).
    """
    backup_path = _get_backup_path(db_path)

    if backup_path.exists():
        print(
            f"  [SKIP] Backup '{backup_path.name}' already exists — "
            "index was previously cleaned, skipping purge."
        )
        return 0

    if not dry_run:
        shutil.copy2(db_path, backup_path)
        print(f"  Backup created: '{backup_path.name}'")

    total_purged = 0
    conn = duckdb.connect(str(db_path))

    try:
        for table in _get_tables(conn):
            id_cols = _find_id_columns(conn, table)
            if not id_cols:
                print(f"  [WARN] No scannable columns in '{table}' — skipping.")
                continue

            for id_col in id_cols:
                bad_ids = [
                    row[0]
                    for row in conn.execute(f'SELECT DISTINCT "{id_col}" FROM "{table}"').fetchall()
                    if has_unsafe_id(str(row[0]))
                ]

                if not bad_ids:
                    continue

                print(
                    f"  {'[DRY RUN] ' if dry_run else ''}Purging {len(bad_ids)} "
                    f"illegal ID(s) from '{table}' (column: '{id_col}'):"
                )
                for name in sorted(bad_ids):
                    print(f"    - {repr(name)}")

                if not dry_run:
                    for name in bad_ids:
                        conn.execute(
                            f'DELETE FROM "{table}" WHERE "{id_col}" = ?', [name]
                        )
                    total_purged += len(bad_ids)

        if not dry_run and total_purged:
            print(f"  Purge complete — {total_purged} row(s) removed.")
        elif not dry_run:
            print("  No illegal IDs found — index was already clean.")

    finally:
        conn.close()

    return total_purged