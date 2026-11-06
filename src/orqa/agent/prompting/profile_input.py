"""Per-column inputs of the column statistics: the frequency table every fact
is computed from, built from a parquet file through Arrow (no pandas frame),
or from a pandas frame (CSV, or a table already loaded) exactly as
``ColumnStatistics.compute`` builds it today.

For every selected column the result holds what ``compute`` derives from
``pd.factorize`` on the frame ``utils.prepare_dataset`` loads
(``pd.read_parquet`` -> ``clean_columns`` -> ``select_columns``):

- ``codes``: one int32 per row, the row's index into the distinct values,
  -1 for null (NaN, None, NaT);
- ``uniques``: the distinct non-null values in first-seen order, as the
  ``pd.Index`` ``pd.factorize`` returns for that pandas column (an object
  Index of str, an int64 / float64 / bool Index, a DatetimeIndex, ...), so
  ``uniques.tolist()`` gives the Python objects the frame would hold (str,
  int, float, bool, Timestamp, Decimal, date);
- ``counts``: rows per distinct value, ``nan_count``, and the pandas dtype
  of the frame's column (``int64`` with nulls loads as ``float64``, ``bool``
  with nulls as ``object``, ``large_string`` as ``object``); the type itself
  is decided by ``compute`` from that dtype and the distinct values.

The Arrow route (``profile_parquet``) reads string columns dictionary-encoded
(``read_dictionary``: the parquet dictionary pages become the codes, 1.7 s
and 7.7 GB for a 113M x 7 table instead of 17 s and 25.7 GB for the frame)
and the other columns as plain arrays. ``pyarrow.compute.dictionary_encode``
turns both into first-seen codes plus distinct values without a Python
object per row (a dictionary column's chunk dictionaries are unified first,
then its index stream is encoded: the parquet dictionary order is the
writer's, not the rows' first-seen order that ``pd.factorize`` gives and the
facts rely on for ties). The distinct values alone are then put through
pandas' own ``factorize``, so that pandas' equivalences apply exactly: NaN is
null, -0.0 equals 0.0, an integer column with nulls is float64 (values above
2**53 may collide), the int64 minimum of a timestamp is NaT. Whatever the
Arrow route cannot do exactly (list / struct / map cells, which pandas
cannot hash, categorical and extension columns, a failed kernel) is done per
column through ``to_pandas()`` on that column alone, which is the pandas
route itself.

The pandas route (``profile_dataframe``) is ``compute``'s own loop:
``pd.factorize(series, use_na_sentinel=True)``, the text form of unhashable
cells on a TypeError, ``np.bincount`` on the codes.

``ColumnStatistics.compute(df, ..., source=path)`` takes the Arrow route
for a parquet ``path`` and checks it against the frame (row count, dtypes,
sampled cells), keeping the pandas route for any column that differs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Codes are int32 below this many distinct values, as in statistics.py
# (``_INT32_CODES_MAX_CARD``): Arrow's dictionary indices are int32 too.
_INT32_CODES_MAX_CARD = 2**31
# Rows per ``np.bincount`` call: bincount copies its input to int64, so one
# call over 113M codes would allocate 0.9 GB on top of the mask and the
# compressed copy; 16M-row blocks keep that under 200 MB.
_COUNT_BLOCK = 1 << 24


@dataclass
class ColumnInput:
    """One column's frequency table and type, as ``ColumnStatistics.compute``
    builds them from ``pd.factorize`` (see the module docstring)."""

    name: str  # the column label, stringified like utils.clean_columns
    dtype: str  # str(series.dtype) of the pandas frame's column
    pandas_dtype: Any  # that dtype object (np.dtype or pandas ExtensionDtype)
    codes: np.ndarray  # int32 per row, -1 for null
    uniques: pd.Index  # distinct non-null values, first-seen order, pandas' representation
    counts: np.ndarray  # int64 rows per distinct value
    nan_count: int
    # False when pandas cannot hash the cells (list / struct columns): the
    # codes come from their text form, as statistics._factorize does, and
    # ``compute`` keeps the counts only.
    hashable: bool = True

    @property
    def cardinality(self) -> int:
        return int(len(self.uniques))

    @property
    def num_rows(self) -> int:
        return int(len(self.codes))

    @property
    def null_ratio(self) -> float:
        return self.nan_count / self.num_rows if self.num_rows else 0.0



@dataclass
class ProfileInput:
    """The selected columns of one table, in the selected order (a column
    the Arrow route leaves to the frame is absent)."""

    num_rows: int
    columns: List[ColumnInput] = field(default_factory=list)
    source: str = "pandas"  # "arrow" | "pandas"

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.columns]

    def __getitem__(self, name: str) -> ColumnInput:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Column selection from the parquet schema (what prepare_dataset selects)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ParquetColumn:
    label: str  # the pandas column label, stringified like clean_columns
    field_index: int  # the Arrow field it is read from
    field_name: str
    arrow_type: pa.DataType
    base_dtype: Any  # the pandas dtype of the column when it holds no null


def _parquet_columns(schema: pa.Schema) -> List[_ParquetColumn]:
    """Every column of the frame ``pd.read_parquet`` gives for ``schema``, in
    frame order: the Arrow field it comes from (an index pandas stored with
    the table is no column), its label stringified like ``clean_columns``
    (pandas restores int labels from its metadata; their field name is the
    same ``str``) and its no-null pandas dtype.

    The labels and dtypes come from pyarrow's own conversion of the empty
    table, so metadata-driven conversions (categoricals, time zones, an
    index column) are the frame's. Only the null-driven ones are missing
    there (``_pandas_dtype``)."""
    index_fields = set()
    meta = schema.pandas_metadata
    if meta:
        index_fields = {entry for entry in meta.get("index_columns", []) if isinstance(entry, str)}
    fields = [(i, f) for i, f in enumerate(schema) if f.name not in index_fields]
    labels: Optional[List[Any]] = None
    dtypes: List[Any] = []
    try:
        empty = schema.empty_table().to_pandas()
        if len(empty.columns) == len(fields) and all(
            str(label) == f.name for label, (_, f) in zip(empty.columns, fields)
        ):
            labels = list(empty.columns)
            dtypes = list(empty.dtypes)
    except Exception:  # pragma: no cover - exotic metadata; the per-field conversion below is the fallback
        logger.debug("profile_input: pandas conversion of the empty schema failed", exc_info=True)
    if labels is None:
        labels = [f.name for _, f in fields]
        dtypes = [pa.array([], type=f.type).to_pandas().dtype for _, f in fields]
    return [
        _ParquetColumn(label=str(label), field_index=i, field_name=f.name, arrow_type=f.type, base_dtype=dtype)
        for label, dtype, (i, f) in zip(labels, dtypes, fields)
    ]


def _usable(label: str) -> bool:
    """``utils._is_usable_column_label``: blank labels are dropped."""
    return label.strip() != ""


def parquet_column_labels(path: Union[str, Path]) -> List[str]:
    """The column labels ``prepare_dataset`` sees after ``clean_columns``
    (stringified, blank ones dropped), in file order, from the schema only."""
    return [c.label for c in _parquet_columns(pq.read_schema(path)) if _usable(c.label)]


def select_parquet_columns(
    path: Union[str, Path],
    limit_to_n_columns: int = 20,
    involved_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    """The ordered column names ``prepare_dataset`` selects for ``path``
    (``clean_columns`` then ``utils.select_columns``), without loading it."""
    from orqa.utils import select_columns  # the very function prepare_dataset uses

    return select_columns(parquet_column_labels(path), limit_to_n_columns, list(involved_cols or []))


# ---------------------------------------------------------------------------
# The Arrow route
# ---------------------------------------------------------------------------


def profile_parquet(
    path: Union[str, Path],
    columns: Sequence[str],
    *,
    limit_to_n_rows: Optional[int] = None,
    use_threads: bool = True,
) -> ProfileInput:
    """The frequency tables of ``columns`` of the parquet file ``path``,
    equal to those ``ColumnStatistics.compute`` builds on the frame
    ``prepare_dataset`` would load, without building that frame.

    Args:
        path: The parquet file.
        columns: The ordered column names the pipeline selected, as
            ``select_parquet_columns`` (or ``prepare_dataset``'s frame)
            names them: stringified labels. Each must exist in the file.
        limit_to_n_rows: ``prepare_dataset``'s row cap (``df.head``), None
            for every row (production).
        use_threads: Passed to the parquet reader.

    Raises:
        KeyError: A requested column is not in the file.
        ValueError: The file holds several fields with one requested name.
            pandas cannot load such a file at all (``pd.read_parquet``
            fails on duplicate field names), so no frame exists to equal.

    Each column is read on its own (one Arrow column in memory at a time,
    plus the codes already built) and dropped once its codes exist.
    """
    path = Path(path)
    schema = pq.read_schema(path)
    available = [c for c in _parquet_columns(schema) if _usable(c.label)]
    by_label: Dict[str, List[_ParquetColumn]] = {}
    for column in available:
        by_label.setdefault(column.label, []).append(column)
    wanted = [str(c) for c in columns]
    missing = [c for c in wanted if c not in by_label]
    if missing:
        raise KeyError(f"{path.name}: columns {missing} are not in the file (available: {list(by_label)})")
    duplicated = sorted({c for c in wanted if len(by_label[c]) > 1})
    if duplicated:
        raise ValueError(
            f"{path.name}: several fields are named {duplicated}; pandas cannot load a parquet file "
            "with duplicate field names, so there is no frame whose statistics this could equal"
        )
    selected = [by_label[c][0] for c in wanted]
    dictionary_fields = [c.field_name for c in selected if _read_as_dictionary(c.arrow_type)]

    parquet_file = pq.ParquetFile(path, read_dictionary=dictionary_fields or None)
    try:
        num_rows = int(parquet_file.metadata.num_rows)
        if limit_to_n_rows is not None:
            num_rows = min(num_rows, max(0, int(limit_to_n_rows)))
        out: List[ColumnInput] = []
        for column in selected:
            chunked = parquet_file.read(columns=[column.field_name], use_threads=use_threads).column(0)
            # pandas types the whole column before ``head``: an int column
            # with a null past the cap is float64 in the capped frame too.
            has_nulls = chunked.null_count > 0
            if len(chunked) > num_rows:
                chunked = chunked.slice(0, num_rows)
            given = _column_from_arrow(column, chunked, num_rows, has_nulls)
            if given is not None:
                out.append(given)
            del chunked
    finally:
        parquet_file.close()
    return ProfileInput(num_rows=num_rows, columns=out, source="arrow")


def _read_as_dictionary(arrow_type: pa.DataType) -> bool:
    """Columns read through the parquet dictionary pages: BYTE_ARRAY
    storage, the only one ``read_dictionary`` supports."""
    return (
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_binary(arrow_type)
        or pa.types.is_large_binary(arrow_type)
    )


def _arrow_exact(arrow_type: pa.DataType) -> bool:
    """Types whose distinct values the Arrow route derives exactly (hash
    kernels exist and pandas' equivalences are applied on the distinct
    values afterwards). Everything else takes the pandas route per column."""
    t = arrow_type
    return (
        _read_as_dictionary(t)
        or pa.types.is_fixed_size_binary(t)
        or pa.types.is_integer(t)
        or pa.types.is_floating(t)
        or pa.types.is_boolean(t)
        or pa.types.is_timestamp(t)
        or pa.types.is_date(t)
        or pa.types.is_time(t)
        or pa.types.is_duration(t)
        or pa.types.is_decimal(t)
    )


def _column_from_arrow(
    column: _ParquetColumn, chunked: pa.ChunkedArray, num_rows: int, has_nulls: bool
) -> ColumnInput:
    """One column's ColumnInput from its Arrow data (``has_nulls``: the
    whole column of the file holds a null, which decides the pandas dtype).
    None for a column pandas loads as an extension dtype other than a zoned
    timestamp (``Int64``, ``boolean``, ``category``, ``string``... restored
    from the file's pandas metadata): only the frame has its exact values."""
    dtype = _pandas_dtype(column.base_dtype, has_nulls)
    if not isinstance(dtype, (np.dtype, pd.DatetimeTZDtype)):
        return None
    if len(chunked) == 0:
        # Nothing to encode.
        return _column_from_series(column.label, pd.Series([], dtype=dtype), num_rows)
    if not _arrow_exact(column.arrow_type):
        return _column_from_series(column.label, _to_pandas(chunked, column), num_rows)
    try:
        codes, values = _first_seen(chunked)
        uniques, remap = _pandas_uniques(values, dtype)
        del values
        if remap is not None:
            codes = _remap(codes, remap)
    except Exception:
        # The pandas route is exact by construction; a kernel or conversion
        # this version of Arrow / pandas does not offer only costs memory.
        logger.warning(
            "profile_input: column %r (%s): Arrow route failed, using the pandas route",
            column.label, column.arrow_type, exc_info=True,
        )
        return _column_from_series(column.label, _to_pandas(chunked, column), num_rows)
    counts = _counts(codes, len(uniques))
    nan_count = num_rows - int(counts.sum())
    return ColumnInput(
        name=column.label, dtype=str(dtype), pandas_dtype=dtype,
        codes=codes, uniques=uniques, counts=counts, nan_count=nan_count, hashable=True,
    )


def _to_pandas(chunked: pa.ChunkedArray, column: _ParquetColumn) -> pd.Series:
    """The column as the frame would hold it: a column read through the
    dictionary pages is cast back to its plain type first (a dictionary
    converts to a Categorical, the frame holds object str)."""
    if pa.types.is_dictionary(chunked.type) and not pa.types.is_dictionary(column.arrow_type):
        chunked = chunked.cast(column.arrow_type)
    return chunked.to_pandas()


def _first_seen(chunked: pa.ChunkedArray) -> Tuple[np.ndarray, pa.Array]:
    """``(codes, values)``: int32 first-seen codes per row (-1 null) and the
    distinct values in that order, as one Arrow array."""
    if pa.types.is_dictionary(chunked.type):
        # The parquet dictionaries (one per row group, PLAIN pages appended
        # by the reader) are unified into one shared dictionary, and the
        # index stream is encoded again: the encoding's dictionary lists the
        # shared dictionary's codes in the rows' first-seen order.
        unified = chunked.unify_dictionaries()
        shared = unified.chunk(0).dictionary
        indices = pa.chunked_array([chunk.indices for chunk in unified.chunks])
        del unified
        encoded = pc.dictionary_encode(indices)
        del indices
        values = shared.take(_dictionary_of(encoded))
    else:
        encoded = pc.dictionary_encode(chunked)
        values = _dictionary_of(encoded)
    return _codes_of(encoded), values


def _dictionary_of(encoded: pa.ChunkedArray) -> pa.Array:
    """The dictionary of a ``dictionary_encode`` result: one memo table over
    every chunk, so each chunk carries the complete dictionary (``_counts``
    checks that no code falls outside it)."""
    return encoded.chunk(encoded.num_chunks - 1).dictionary


def _codes_of(encoded: pa.ChunkedArray) -> np.ndarray:
    indices = pa.chunked_array([chunk.indices for chunk in encoded.chunks])
    if indices.null_count:
        indices = pc.fill_null(indices, pa.scalar(-1, type=indices.type))
    return indices.to_numpy().astype(np.int32, copy=False)


def _pandas_dtype(base_dtype: Any, has_nulls: bool) -> Any:
    """The frame's dtype of a column: pyarrow gives an integer column with
    nulls as float64 and a bool column with nulls as object; every other
    dtype does not depend on nulls."""
    if has_nulls and isinstance(base_dtype, np.dtype):
        if base_dtype.kind in "iu":
            return np.dtype("float64")
        if base_dtype.kind == "b":
            return np.dtype("O")
    return base_dtype


def _pandas_uniques(values: pa.Array, dtype: Any) -> Tuple[pd.Index, Optional[np.ndarray]]:
    """The distinct values in pandas' representation, with the remapping of
    the Arrow codes when pandas' equivalences merge or null some of them.

    Returns ``(uniques, remap)``: ``uniques`` is what ``pd.factorize`` would
    return on the frame's column (its Index class and dtype), ``remap[code]``
    the final code of Arrow code ``code`` (-1 for a value pandas treats as
    null), or None when the Arrow codes already are the final ones."""
    arrow_type = values.type
    if _read_as_dictionary(arrow_type) or pa.types.is_fixed_size_binary(arrow_type):
        # str / bytes: Arrow's byte equality is pandas' object equality.
        array = values.to_numpy(zero_copy_only=False)
        if values.null_count == 0:
            return pd.Index(array, dtype=object), None
        return _refactorize(pd.Series(array, dtype=object))
    if pa.types.is_boolean(arrow_type):
        array = values.to_numpy(zero_copy_only=False)
        if dtype == object:  # bool with nulls: object column of Python bools
            array = array.astype(object)
        if values.null_count == 0:
            return pd.Index(array), None
        return _refactorize(pd.Series(array))
    if pa.types.is_integer(arrow_type):
        array = values.to_numpy(zero_copy_only=False)
        if isinstance(dtype, np.dtype) and dtype.kind == "f":
            # int with nulls -> float64: values above 2**53 may collide.
            return _refactorize(pd.Series(array.astype(dtype)))
        if values.null_count == 0:
            return pd.Index(array), None
        return _refactorize(pd.Series(array))
    if pa.types.is_floating(arrow_type):
        # NaN is null to pandas (Arrow keeps it as a value) and -0.0 equals 0.0.
        return _refactorize(pd.Series(values.to_numpy(zero_copy_only=False)))
    # Timestamps (NaT at the int64 minimum), durations, dates, times,
    # decimals: pandas' own conversion of the distinct values.
    series = values.to_pandas()
    if str(series.dtype) != str(dtype):
        raise ValueError(f"distinct values convert to {series.dtype}, the frame holds {dtype}")
    return _refactorize(series)


def _refactorize(series: pd.Series) -> Tuple[pd.Index, np.ndarray]:
    remap, uniques = pd.factorize(series, use_na_sentinel=True)
    return uniques, remap.astype(np.int32, copy=False)


def _remap(codes: np.ndarray, remap: np.ndarray) -> np.ndarray:
    """``remap[codes]`` with -1 kept (-1 indexes the appended slot)."""
    if np.array_equal(remap, np.arange(len(remap), dtype=remap.dtype)):
        return codes
    lookup = np.append(remap.astype(np.int32, copy=False), np.int32(-1))
    return lookup[codes]


def _counts(codes: np.ndarray, cardinality: int) -> np.ndarray:
    """``np.bincount(codes[codes >= 0], minlength=cardinality)`` in blocks."""
    counts = np.zeros(cardinality, dtype=np.int64)
    for start in range(0, len(codes), _COUNT_BLOCK):
        block = codes[start:start + _COUNT_BLOCK]
        block = block[block >= 0]
        if block.size:
            partial = np.bincount(block, minlength=cardinality)
            if len(partial) > cardinality:
                raise RuntimeError("a code falls outside the distinct values")
            counts += partial
    return counts


# ---------------------------------------------------------------------------
# The pandas route (what compute does today)
# ---------------------------------------------------------------------------


def profile_dataframe(df: pd.DataFrame) -> ProfileInput:
    """The frequency tables of every column of ``df``, as
    ``ColumnStatistics.compute`` builds them (columns by position, labels
    stringified)."""
    num_rows = int(len(df))
    columns = [
        _column_from_series(str(label), df.iloc[:, i], num_rows) for i, label in enumerate(df.columns)
    ]
    return ProfileInput(num_rows=num_rows, columns=columns, source="pandas")


def _column_from_series(label: str, series: pd.Series, num_rows: int) -> ColumnInput:
    """``statistics._factorize`` + bincount + the type decision on one column."""
    try:
        codes, uniques = pd.factorize(series, use_na_sentinel=True)
        hashable = True
    except TypeError:
        # A parquet list or struct column loads as arrays or dicts: their
        # text form still gives the column its counts and its partition.
        logger.warning("profile_input: column %r holds unhashable values; codes from their text form", label)
        codes, uniques = pd.factorize(series.astype(str).where(series.notna()), use_na_sentinel=True)
        hashable = False
    codes = codes.astype(np.int32 if len(uniques) < _INT32_CODES_MAX_CARD else np.int64, copy=False)
    counts = _counts(codes, len(uniques))
    nan_count = num_rows - int(counts.sum())
    return ColumnInput(
        name=label, dtype=str(series.dtype), pandas_dtype=series.dtype,
        codes=codes, uniques=uniques, counts=counts, nan_count=nan_count, hashable=hashable,
    )
