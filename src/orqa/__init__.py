import pandas as pd

# Text columns load as pandas 3's string dtype ("str": Arrow-backed, NaN for a
# missing value) instead of one Python object per cell. LakeGen re-executes
# every reference_code on pandas 3, whose plain pd.read_parquet gives exactly
# these frames (400/400 nyc and uk tables identical, 2026-11-09), so OrQA's
# answers are computed on the frames LakeGen scores them on. It also keeps the
# largest tables in memory: the 169M-row nyc taxi table is 45 GB this way and
# went past 115 GB as object strings. Set on import, before any frame exists;
# a forked code sandbox inherits it.
pd.options.future.infer_string = True
