"""Ingestion pipeline: crawl a documentation site and index it in Pinecone.

Crawl (Tavily) -> Documents -> chunks -> embeddings written to Pinecone in
concurrent batches, into one namespace per site. The CLI indexes the built-in
LangChain docs; the UI calls `ingest_site` for user-added sites.

    uv run ingestion.py
"""

import asyncio
import os
import ssl
from typing import Any, Callable, Dict, List, Optional

import certifi
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Use certifi's CA bundle so HTTPS calls verify on machines with stale system certs.
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

LANGCHAIN_DOCS = "https://python.langchain.com/"
CHUNK_SIZE, CHUNK_OVERLAP = 4000, 200

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)

# Progress callback: on_event(stage, **details), where stage is one of
# "crawl", "documents", "chunks", "store", "batch", "done".
Event = Callable[..., None]


def _noop(stage: str, **details: Any) -> None:
    pass


@traceable(name="crawl", run_type="tool")
def crawl_site(url: str, instructions: Optional[str] = None, max_depth: int = 3,
               limit: Optional[int] = None, extract_depth: str = "advanced"):
    """Crawl `url`. Returns the Tavily response, or None on error."""
    log_info(f"🔍 TavilyCrawl: Starting to crawl documentation from {url}", Colors.PURPLE)

    params: Dict[str, Any] = {"url": url, "extract_depth": extract_depth, "max_depth": max_depth}
    if instructions:
        params["instructions"] = instructions
    if limit:
        params["limit"] = limit

    tavily_crawl_results = TavilyCrawl().invoke(input=params)
    if tavily_crawl_results.get("error"):
        log_error(f"TavilyCrawl: {tavily_crawl_results['error']}")
        return None
    log_success(f"TavilyCrawl: Successfully crawled {len(tavily_crawl_results.get('results', []))} pages")
    return tavily_crawl_results


def make_documents(tavily_crawl_results) -> List[Document]:
    """Convert crawled pages to Documents, skipping pages with no content."""
    all_docs = []
    for item in tavily_crawl_results["results"]:
        if not item.get("raw_content"):
            continue
        all_docs.append(Document(page_content=item["raw_content"], metadata={"source": item["url"]}))
    return all_docs


@traceable(name="split")
def split_documents(all_docs: List[Document]) -> List[Document]:
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with {CHUNK_SIZE} chunk size and {CHUNK_OVERLAP} overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents")
    return splitted_docs


@traceable(name="store")
async def index_documents_async(splitted_docs: List[Document], namespace: str = "",
                                batch_size: int = 50, on_event: Event = _noop) -> int:
    """Embed and upsert chunks into `namespace`, all batches concurrently.

    Returns the number of batches that succeeded.
    """
    log_header("VECTOR STORAGE PHASE")
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings, namespace=namespace or None
    )
    batches = [splitted_docs[i : i + batch_size] for i in range(0, len(splitted_docs), batch_size)]
    log_info(f"📦 VectorStore Indexing: {len(batches)} batches of up to {batch_size} documents")

    async def add_batch(batch: List[Document], batch_num: int) -> bool:
        try:
            await vectorstore.aadd_documents(batch)
            log_success(f"VectorStore Indexing: added batch {batch_num}/{len(batches)} ({len(batch)} documents)")
            on_event("batch", done=batch_num, total=len(batches))
            return True
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False

    results = await asyncio.gather(*(add_batch(b, i + 1) for i, b in enumerate(batches)), return_exceptions=True)
    successful = sum(1 for r in results if r is True)
    if successful == len(batches):
        log_success(f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})")
    else:
        log_warning(f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully")
    return successful


def print_summary(pages: int, all_docs, splitted_docs) -> None:
    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Pages crawled: {pages}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


# Grouped into one LangSmith trace per site when LANGSMITH_TRACING=true (no-op otherwise).
@traceable(name="ingest_site")
def ingest_site(url: str, namespace: str = "", *, instructions: Optional[str] = None,
                max_depth: int = 3, limit: Optional[int] = None, extract_depth: str = "advanced",
                batch_size: int = 500, on_event: Event = _noop) -> Dict[str, int]:
    """Run the full pipeline for one site. Returns {"pages", "documents", "chunks"}.

    Raises RuntimeError if the crawl fails or nothing could be indexed.
    """
    log_header("DOCUMENTATION INGESTION PIPELINE")
    on_event("crawl", url=url)
    results = crawl_site(url, instructions, max_depth, limit, extract_depth)
    if results is None or not results.get("results"):
        raise RuntimeError("The crawl returned no pages. Check the URL, or try the site's docs root.")

    pages = len(results["results"])
    all_docs = make_documents(results)
    on_event("documents", pages=pages, documents=len(all_docs))
    if not all_docs:
        raise RuntimeError("The crawled pages had no readable text.")

    splitted_docs = split_documents(all_docs)
    on_event("chunks", chunks=len(splitted_docs))

    on_event("store", chunks=len(splitted_docs))
    if asyncio.run(index_documents_async(splitted_docs, namespace, batch_size, on_event)) == 0:
        raise RuntimeError("Saving to Pinecone failed.")

    print_summary(pages, all_docs, splitted_docs)
    stats = {"pages": pages, "documents": len(all_docs), "chunks": len(splitted_docs)}
    on_event("done", **stats)
    return stats


if __name__ == "__main__":
    ingest_site(LANGCHAIN_DOCS, instructions="Documentation relevant to AI agents", max_depth=3)
