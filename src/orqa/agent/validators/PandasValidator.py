import pandas as pd
import polars as pl
from io import StringIO
import sys
from typing import List, Dict, Tuple, Any
from .QueryValidator import QueryValidator
import re

UNAUTHORIZED_COMMANDS = [
    'read_csv', 'read_excel', 'read_json', 'read_parquet', 'print',
    'read_sql', 'read_table', 'read_html', 'read_pickle',
    'to_csv', 'to_excel', 'to_json', 'to_parquet', 'to_pickle',
    'open(', 'os.', 'sys.', 'exec(', 'eval('
]

class PandasValidator(QueryValidator):
    """Validator for pandas/polars queries."""

    def _preprocess_query(self, query: str) -> str:
        return self.clean_pandas(query)

    def _execute_query(self, query: str) -> None:
        safe_builtins = {
            'abs': abs, 'min': min, 'max': max, 'sum': sum,
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'True': True, 'False': False, 'None': None,
        }

        namespace = {
            'pd': pd,
            'pl': pl,
            '__builtins__': safe_builtins
        }

        for df, name in zip(self.dataframes, self.table_names):
            namespace[name] = df

        local_namespace = namespace.copy()

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            compiled_code = compile(query, '<string>', 'exec')
            exec(compiled_code, local_namespace)
        except KeyError as e:
            raise KeyError(
                f'Column {str(e)} not found at runtime.\n\n'
                f'Possible causes:\n'
                f'  1. Column does not exist in any of the original tables — check spelling and case\n'
                f'  2. Column was incorrectly suffixed — a suffix is ONLY valid if the column name exists in BOTH tables\n'
                f'  3. Column was incorrectly left unsuffixed — if a column name exists in BOTH tables it MUST be suffixed\n\n'
                f'Strictly follow the suffix constraints provided — do not infer, assume, or apply suffixes beyond what is defined there.\n'
            )
        except SyntaxError as e:
            raise SyntaxError(
                f'Invalid Python syntax in query.\n'
                f'Error: {str(e)}\n'
                f'Tip: Ensure multiple statements are separated by newlines (\\n)'
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _get_language_specific_rules(self) -> List[str]:
        return [
            '- Column names must exactly match the DataFrame schema.',
            '- Use correct pandas/polars syntax (e.g., df[\'column\'] for pandas, df[\'column\'] for polars).',
            '- For numeric operations on string columns, convert with .astype(float) (pandas) or .cast(pl.Float64) (polars).',
            f'- IMPORTANT: Available DataFrames are named EXACTLY: {", ".join(self.table_names)}',
            '- DO NOT add "df_" prefix or any other modification to these names.',
        ]

    def _get_language_name(self) -> str:
        return 'Python'

    def clean_pandas(self, query_code: str) -> str:
        """Clean pandas query code."""
        if not query_code:
            return ''

        query = query_code.replace('`', "'")

        # FIX (Bug 3): Protect .query("...") / .query('...') calls BEFORE the
        # blanket double-quote normalisation below.
        #
        # The old code converted ALL " to ' first, which turned
        #   df.query("borough == 'Manhattan'")
        # into
        #   df.query('borough == 'Manhattan'')   <- SyntaxError
        # The follow-up fix_query_quotes regex then never fired because its
        # [^']* pattern can't match across the inner single quotes.
        #
        # Fix: stash every .query(…) argument verbatim, normalise the rest of
        # the code, then restore the originals untouched.
        _query_calls: list = []

        def _stash(m: re.Match) -> str:
            _query_calls.append(m.group(0))
            return f'__QUERYCALL_{len(_query_calls) - 1}__'

        # Match .query("...") or .query('...') with either outer quote style.
        query = re.sub(r'\.query\((["\']).*?\1\)', _stash, query, flags=re.DOTALL)

        # Safe to normalise: replace remaining double quotes with single quotes.
        query = query.replace('"', "'")

        # Restore .query() calls exactly as originally written.
        def _restore(m: re.Match) -> str:
            return _query_calls[int(m.group(1))]

        query = re.sub(r'__QUERYCALL_(\d+)__', _restore, query)

        # Fix backslash continuations with trailing spaces ("\ " instead of "\")
        query = re.sub(r'\\\s+\n', '\\\n', query)
        # Remove backslash continuations entirely and join lines
        query = re.sub(r'\\\n\s*', ' ', query)

        lines = query.split(';')
        cleaned = [line.strip() for line in lines if 'import' not in line.lower()]
        # Remove comments from each line
        cleaned = [re.sub(r'#.*$', '', line).strip() for line in cleaned]
        # Remove unauthorized commands
        cleaned = [line for line in cleaned if not any(cmd in line for cmd in UNAUTHORIZED_COMMANDS)]
        # Filter out empty lines
        cleaned = [line for line in cleaned if line]
        return '; '.join(cleaned).strip()