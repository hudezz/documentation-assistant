"""Indexed documentation sites, one Pinecone namespace per site.

The namespaces in the index are the source list; the built-in LangChain docs
live in the default namespace.
"""

import os
import re
from functools import lru_cache
from typing import Dict
from urllib.parse import urlparse

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

BUILTIN = ""      # default namespace: the LangChain docs from ingestion.py
MAX_SOURCES = 12  # user-added sites; new crawls are refused beyond this


@lru_cache(maxsize=1)
def _index():
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(os.environ["INDEX_NAME"])


def validate_url(raw: str) -> str:
    """Return a normalised https URL, or raise ValueError."""
    url = raw.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    host = (urlparse(url).hostname or "").lower()
    if "." not in host or host.endswith(".local") or re.fullmatch(r"[\d.]+", host):
        raise ValueError("Enter a public website address, e.g. https://docs.pydantic.dev")
    return url.split("#")[0].rstrip("/")


def namespace_for(url: str) -> str:
    """docs.pydantic.dev/latest -> docs-pydantic-dev-latest"""
    p = urlparse(url)
    return re.sub(r"[^a-z0-9]+", "-", f"{p.hostname}{p.path}".lower()).strip("-")[:60]


def list_sources() -> Dict[str, int]:
    """{namespace: chunk count}, built-in first."""
    counts = {ns: s.vector_count for ns, s in _index().describe_index_stats().namespaces.items()}
    return {BUILTIN: counts.pop(BUILTIN, 0), **dict(sorted(counts.items()))}


def label(namespace: str) -> str:
    return "LangChain docs (built-in)" if namespace == BUILTIN else namespace


def remove(namespace: str) -> None:
    if namespace != BUILTIN:
        _index().delete(delete_all=True, namespace=namespace)
