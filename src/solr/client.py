"""Read-only HTTP client of the Solr cores OrQA retrieves from.

It serves OrQA's own Solr (``solr.instance``) and only reads:
``select`` (GET ``/<core>/select``), ``query_json`` (POST ``/<core>/query``,
a JSON *search* request) and ``schema`` (GET ``/<core>/schema``). Nothing
here updates, deletes, commits or reloads; writes go through
``solr.instance.SolrInstance``, which only writes to OrQA's own server.

Two transport choices are measured, not stylistic:

* KNN goes through ``query_json``: a 1024-d vector in a GET query string made
  an 11.9 KB URL and HTTP 414, and through ``/select`` (whose core default is
  ``defType=edismax``) ``{!knn}`` silently returned 0 hits. ``/query`` has no
  defType default, so the KNN parser is honoured — LakeGen posts there too.
* One pooled ``requests.Session`` is shared by every call and thread: the
  anchor search sends thousands of ~1 ms queries, and a fresh connection per
  query (the old ``requests.get``) costs more than the query.

Connection errors, timeouts and HTTP 5xx are retried with backoff; anything
still failing raises :class:`SolrError` naming the core and URL, so a dead
core is never mistaken for "no hits".
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter

from solr.instance import default_url

logger = logging.getLogger(__name__)


class SolrError(RuntimeError):
    """A Solr request that failed for good (after retries, or a 4xx)."""


class LocalSolrClient:
    """Read-only client of one Solr service.

    Args:
        core: Default core of ``select`` (an index passes its own ``core=``
            per call, so one client and its connection pool can serve
            several cores).
        base_url: The Solr service URL (``.../solr``); None: OrQA's own.
        timeout: Seconds per request.
        retries: Attempts per request on connection errors, timeouts and
            HTTP 5xx; a 4xx is the caller's mistake and is not retried.
        backoff: First pause between attempts, doubled after each.
        pool_size: Pooled connections, enough for ``search_many``'s threads.
    """

    _COLUMN_DEFAULTS = {"description": None}

    def __init__(
        self,
        core: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        retries: int = 3,
        backoff: float = 0.5,
        pool_size: int = 16,
    ) -> None:
        self.core = core.strip("/") if core else None
        self.base_url = (base_url or default_url()).rstrip("/")
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.backoff = backoff
        # One pool shared across threads: urllib3's pool is thread-safe, and
        # Solr sets no cookies, so the Session's only unsafe state is unused.
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=pool_size)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _core(self, core: Optional[str]) -> str:
        name = (core or self.core or "").strip("/")
        if not name:
            raise ValueError("no Solr core given (pass core= or construct the client with one)")
        return name

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        return [value]

    @classmethod
    def _restore_columns(cls, doc: dict[str, Any]) -> dict[str, Any]:
        restored_doc: dict[str, Any] = {}
        column_values: dict[str, list[Any]] = {}

        for key, value in doc.items():
            if key.startswith("columns."):
                column_values[key.removeprefix("columns.")] = cls._as_list(value)
            else:
                restored_doc[key] = value

        if not column_values:
            return restored_doc

        column_count = max(len(values) for values in column_values.values())
        columns: list[dict[str, Any]] = []
        for index in range(column_count):
            column = {
                field_name: values[index]
                for field_name, values in column_values.items()
                if index < len(values)
            }
            for field_name, default in cls._COLUMN_DEFAULTS.items():
                column.setdefault(field_name, default)
            columns.append(column)

        restored_doc["columns"] = columns
        return restored_doc

    @classmethod
    def _restore_response_docs(cls, result: dict[str, Any]) -> dict[str, Any]:
        response = result.get("response")
        if not isinstance(response, dict):
            return result

        docs = response.get("docs")
        if not isinstance(docs, list):
            return result

        response["docs"] = [
            cls._restore_columns(doc) if isinstance(doc, dict) else doc
            for doc in docs
        ]
        return result

    @staticmethod
    def _solr_message(response: requests.Response) -> str:
        """Solr's own error text (``error.msg``) when the body carries one."""
        try:
            return str(response.json()["error"]["msg"])
        except Exception:
            return response.text[:300]

    def _request(self, method: str, url: str, core: str, **kwargs: Any) -> dict[str, Any]:
        last: str = ""
        for attempt in range(self.retries):
            if attempt:
                time.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                response = self._session.request(method, url, timeout=self.timeout, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = f"{type(exc).__name__}: {exc}"
                logger.debug("Solr %s %s attempt %d failed: %s", method, url, attempt + 1, last)
                continue
            if response.status_code >= 500:
                last = f"HTTP {response.status_code}: {self._solr_message(response)}"
                logger.debug("Solr %s %s attempt %d failed: %s", method, url, attempt + 1, last)
                continue
            if response.status_code >= 400:
                raise SolrError(
                    f"Solr core {core!r} rejected {method} {url}: "
                    f"HTTP {response.status_code}: {self._solr_message(response)}"
                )
            return response.json()
        raise SolrError(
            f"Solr core {core!r} at {url} failed after {self.retries} attempts: {last}"
        )

    def select(
        self,
        tokens: Sequence[str],
        *,
        core: Optional[str] = None,
        def_type: str = "edismax",
        q_op: str = "AND",
        indent: bool = True,
        restore_columns: bool = True,
        **params: Any,
    ) -> dict[str, Any]:
        """GET ``/<core>/select`` with ``q`` = the tokens joined by spaces —
        LakeGen's lexical request. ``qf`` is left to the core's handler
        defaults unless passed. A list value (``fq=[...]``) repeats the
        parameter. ``restore_columns=False`` leaves the flat ``columns.*``
        arrays as stored (an export)."""
        if isinstance(tokens, str):
            raise TypeError("tokens must be a sequence of strings, not a string")
        name = self._core(core)
        result = self._request(
            "GET",
            f"{self.base_url}/{name}/select",
            name,
            params={
                "defType": def_type,
                "indent": str(indent).lower(),
                "q.op": q_op,
                "q": " ".join(tokens),
                "wt": "json",
                **params,
            },
        )
        return self._restore_response_docs(result) if restore_columns else result

    def schema(self, core: Optional[str] = None) -> dict[str, Any]:
        """GET ``/<core>/schema``: the core's whole schema."""
        name = self._core(core)
        return self._request(
            "GET", f"{self.base_url}/{name}/schema", name, params={"wt": "json"}
        )["schema"]

    def query_json(self, core: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON Request API search to ``/<core>/query`` (``query``,
        ``filter``, ``fields``, ``limit``, ``params``). Read-only: the
        ``/query`` handler only searches."""
        name = self._core(core)
        result = self._request(
            "POST", f"{self.base_url}/{name}/query", name, json=body
        )
        return self._restore_response_docs(result)
