"""Retrieval as the question stage uses it: the retriever panel behind the
retrievability gate (``orqa.retrieval.panel``) and the text helpers shared
with the keyword-anchor search (``orqa.retrieval.text``). The portal's index
itself is its Solr core (``solr.index``)."""

from orqa.retrieval.text import record_field_texts, tokenize

__all__ = ["record_field_texts", "tokenize"]
