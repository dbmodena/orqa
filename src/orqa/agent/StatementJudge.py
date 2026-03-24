import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Type

import yaml
import litellm
from litellm import completion, Router
from pydantic import BaseModel, ValidationError


import pandas as pd
from .prompting import DatasetDescription, _load_prompt
from pathlib import Path
from .structured_outputs import QuerySet, Query
import duckdb


import pandas as pd
import polars as pl
from typing import List, Dict, Tuple, Union, Any

import sys
from io import StringIO

import pandas as pd
from pathlib import Path
import duckdb
from .LLMClientStructured import LLMClientStructured
from .validators.SQLValidator import SQLValidator
from .validators.PandasValidator import PandasValidator
import re




class LLMStatementJudge(LLMClientStructured):
    """
    LiteLLM client with YAML configuration and structured output support.
    """

    def __init__(self, config_path: Path):
        super().__init__(config_path, "statement_judge")

    