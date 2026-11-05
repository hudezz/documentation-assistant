"""Streamlit UI for the documentation assistant.

    uv run streamlit run frontend/app.py

UI only; all agent logic lives in core.py.
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

# core.py lives in the project root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import stream_llm  # noqa: E402

from components import (  # noqa: E402
    agent_pipeline,
    background_step,
    live_agent_demo,
    load_demo_runs,
    page_background,
)

st.set_page_config(
    page_title="LangChain Docs Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",  # leaves room for the motion-graphics stage
)


# --- Sidebar ------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def index_size() -> int | None:
    """Vector count in the Pinecone index, or None if unreachable."""
    try:
        from pinecone import Pinecone

        index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(os.environ["INDEX_NAME"])
        return index.describe_index_stats()["total_vector_count"]
    except Exception:
        return None


with st.sidebar:
    st.markdown("### 📚 Docs Assistant")
    st.caption("Answers questions about LangChain, LangGraph and Deep Agents using only the official docs, and cites them.")
    st.divider()
    size = index_size()
    st.metric("Chunks in Pinecone", f"{size:,}" if size is not None else "unavailable")
    st.markdown(f"**Index:** `{os.environ.get('INDEX_NAME', '?')}`")
    st.markdown("**Model:** `gpt-5.2`  \n**Embeddings:** `text-embedding-3-small`")
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Built with LangChain · Pinecone · Tavily · Streamlit")

# Motion-graphics stage (frontend/static/background.js), fed real index stats.
page_background(
    {
        "index": os.environ.get("INDEX_NAME", "langchain-docs"),
        "chunks": size or 0,
        "pages": size or 0,
        "llm": "gpt-5.2",
    }
)


# --- Layout -------------------------------------------------------------------
st.title("LangChain Documentation Assistant")
st.caption("An AI agent that searches the LangChain docs for you, then answers with sources.")

tab_ask, tab_live, tab_how = st.tabs(["💬 Ask the docs", "🤖 Live agent", "🧠 How it works"])


def show_sources(sources: list[str]) -> None:
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for url in sources:
                st.markdown(f"- [{url}]({url})")


# --- Ask the docs -------------------------------------------------------------
with tab_ask:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role", "content", "sources"}]

    if not st.session_state.messages:
        st.info("Try: *What are deep agents?* · *When should I use LangGraph instead of LangChain?*")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            show_sources(msg.get("sources", []))

    if prompt := st.chat_input("Ask about LangChain, LangGraph, Deep Agents…"):
        # Pass prior turns so follow-up questions have context.
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            viz = st.empty()        # in-chat pipeline, redrawn per step
            bg_signal = st.empty()  # progress signal for the stage animation
            status = st.status("Agent is working…", expanded=True)
            with viz:
                agent_pipeline(1)
            with bg_signal:
                background_step(1, query=prompt)

            answer, sources, found_docs = "", [], False
            start = time.time()
            try:
                for event in stream_llm(prompt, history):
                    if event["type"] == "thinking":
                        step = 3 if found_docs else 1  # thinking after retrieval == writing
                        status.write("✍️ Reading the docs and writing an answer…" if found_docs else "🧠 Planning…")
                    elif event["type"] == "search":
                        step = 2
                        status.write(f"🔎 Searching Pinecone for *“{event['query']}”*")
                    elif event["type"] == "results":
                        step, found_docs = 3, True
                        new = [d.metadata.get("source", "unknown") for d in event["docs"]]
                        sources += [u for u in dict.fromkeys(new) if u not in sources]
                        status.write(f"📄 Found {len(event['docs'])} matching chunks")
                    elif event["type"] == "answer":
                        answer, step = event["text"], 4
                    with viz:
                        agent_pipeline(step)
                    with bg_signal:
                        background_step(min(step, 3), query=prompt)

                elapsed = time.time() - start
                with bg_signal:
                    background_step(4, query=prompt, sources=len(sources), seconds=f"{elapsed:.1f}")
                status.update(
                    label=f"Done in {elapsed:.1f}s · {len(sources)} sources",
                    state="complete",
                    expanded=False,
                )
            except Exception as e:
                status.update(label="Something went wrong", state="error")
                answer = f"⚠️ The agent failed: `{e}`"

            st.markdown(answer or "_No answer was produced._")
            show_sources(sources)

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


# --- Live agent (replays frontend/demo_runs.json) ----------------------------
with tab_live:
    runs = load_demo_runs()
    if runs:
        st.caption(
            "This runs nonstop in your browser, even while you use the other tabs. "
            "It replays real recorded runs of the agent (the actual search queries, "
            "pages and answers), so it costs nothing to keep running."
        )
        live_agent_demo(runs)
    else:
        st.warning("No recordings yet. Run `uv run record_demo.py` to record some.")


# --- How it works -------------------------------------------------------------
with tab_how:
    agent_pipeline(4)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
#### 1 · Ingestion (offline, `ingestion.py`)
1. **Crawl:** Tavily crawls the LangChain docs site.
2. **Documents:** each page becomes a LangChain `Document` with its URL.
3. **Chunk:** pages are split into ~4,000-character chunks.
4. **Store:** chunks are embedded with OpenAI and saved to Pinecone
   in batches, concurrently.
"""
        )
    with col2:
        st.markdown(
            """
#### 2 · Answering (live, `core.py`)
1. **Plan:** the agent reads your question and decides to search.
2. **Search:** it calls `retrieve_context`, which finds the most
   similar chunks in Pinecone.
3. **Answer:** it writes an answer **only from those chunks**
   and cites the pages. If the docs don't say, it says so.
"""
        )
