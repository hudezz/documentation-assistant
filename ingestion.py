"""Ingestion pipeline: crawl the LangChain docs and index them in Pinecone.

Crawl (Tavily) -> Documents -> chunks -> embeddings written to Pinecone in
concurrent batches. Run once, or whenever the docs need refreshing:

    uv run ingestion.py
"""

import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Use certifi's CA bundle so HTTPS calls verify on machines with stale system certs.
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
tavily_crawl = TavilyCrawl()


def crawl_site():
    """Crawl the docs site. Returns the Tavily response, or None on error."""
    log_info(
        "🔍 TavilyCrawl: Starting to crawl documentation from https://python.langchain.com/",
        Colors.PURPLE,
    )

    tavily_crawl_results = tavily_crawl.invoke(
        input={
            "url": "https://python.langchain.com/",
            "extract_depth": "advanced",
            "instructions": "Documentatin relevant to ai agents",
            "max_depth": 3,
        }
    )
    if tavily_crawl_results.get("error"):
        log_error(f"TavilyCrawl: {tavily_crawl_results['error']}")
        return None
    else:
        log_success(
            f"TavilyCrawl: Successfully crawled {len(tavily_crawl_results)} URLs from documentation site"
        )

    return tavily_crawl_results


def make_documents(tavily_crawl_results):
    """Convert crawled pages to Documents, skipping pages with no content."""
    all_docs = []
    for tavily_crawl_result_item in tavily_crawl_results["results"]:
        if not tavily_crawl_result_item.get("raw_content"):
            continue
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    return all_docs


def split_documents(all_docs):
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    return splitted_docs


async def index_documents_async(splitted_docs: List[Document], batch_size: int = 50):
    """Embed and upsert chunks to Pinecone, all batches concurrently."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(splitted_docs)} documents to vector store",
        Colors.DARKCYAN,
    )

    batches = [
        splitted_docs[i : i + batch_size] for i in range(0, len(splitted_docs), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


def print_summary(tavily_crawl_results, all_docs, splitted_docs):
    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Pages crawled: {len(tavily_crawl_results)}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


async def main():
    log_header("DOCUMENTATION INGESTION PIPELINE")

    tavily_crawl_results = crawl_site()
    if tavily_crawl_results is None:
        return

    all_docs = make_documents(tavily_crawl_results)
    splitted_docs = split_documents(all_docs)
    await index_documents_async(splitted_docs, batch_size=500)
    print_summary(tavily_crawl_results, all_docs, splitted_docs)


if __name__ == "__main__":
    asyncio.run(main())