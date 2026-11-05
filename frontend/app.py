"""Streamlit UI for the documentation assistant.

    uv run streamlit run frontend/app.py

UI only; agent logic is in core.py, crawling in ingestion.py.
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="Documentation Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",  # leaves room for the motion-graphics stage
)

# core.py lives in the project root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# API keys: Streamlit Cloud secrets when deployed, .env locally.
REQUIRED_KEYS = ("OPENAI_API_KEY", "PINECONE_API_KEY", "TAVILY_API_KEY", "INDEX_NAME")
try:
    for key in REQUIRED_KEYS:
        if not os.environ.get(key) and key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:  # no secrets.toml (local run)
    pass
load_dotenv(ROOT / ".env")
missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
if missing:
    st.error(f"Missing configuration: {', '.join(missing)}. Add them to `.env` locally, "
             "or to the app's Secrets on Streamlit Cloud, then reboot.")
    st.stop()

from core import stream_llm  # noqa: E402
from ingestion import ingest_site  # noqa: E402
from sources import MAX_SOURCES, label, list_sources, namespace_for, validate_url  # noqa: E402

from components import (  # noqa: E402
    agent_pipeline,
    background_ingest,
    background_step,
    live_agent_demo,
    load_demo_runs,
    page_background,
)

# Limits for user-added sites (the public demo runs on the owner's API keys).
CRAWL = dict(max_depth=2, limit=25, extract_depth="basic")
CRAWLS_PER_SESSION = 3


@st.cache_data(ttl=60, show_spinner=False)
def get_sources() -> tuple[dict, str | None]:
    """({namespace: chunk count}, error message if Pinecone is unreachable)."""
    try:
        return list_sources(), None
    except Exception as e:
        return {"": 0}, f"{type(e).__name__}: {e}"


sources, pinecone_error = get_sources()
if "pending_source" in st.session_state:          # set after indexing a new site
    st.session_state.source = st.session_state.pop("pending_source")
if st.session_state.get("source") not in sources:
    st.session_state.source = ""
ns = st.session_state.source
chats = st.session_state.setdefault("chats", {})  # one chat history per source
messages = chats.setdefault(ns, [])


# --- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Docs Assistant")
    st.caption("Answers questions about any documentation site using only its pages, and cites them.")
    st.divider()
    st.metric("Chunks in this source", f"{sources.get(ns, 0):,}")
    st.markdown(f"**Index:** `{os.environ.get('INDEX_NAME', '?')}`")
    st.markdown("**Model:** `gpt-5.2`  \n**Embeddings:** `text-embedding-3-small`")
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        chats[ns] = []
        st.rerun()
    st.caption("Built with LangChain · Pinecone · Tavily · Streamlit")

# Motion-graphics stage (frontend/static/background.js), fed the selected source's stats.
page_background(
    {
        "index": os.environ.get("INDEX_NAME", "langchain-docs"),
        "source": "docs.langchain.com" if ns == "" else ns,
        "chunks": sources.get(ns, 0),
        "pages": sources.get(ns, 0),
        "llm": "gpt-5.2",
    }
)


# --- Layout -------------------------------------------------------------------
st.title("Documentation Assistant")
st.caption("Point it at a documentation site, then ask questions. Answers come only from those pages, with sources.")

tab_ask, tab_live, tab_how = st.tabs(["💬 Ask the docs", "🤖 Live agent", "🧠 How it works"])


def show_sources(urls: list[str]) -> None:
    if urls:
        with st.expander(f"Sources ({len(urls)})"):
            for url in urls:
                st.markdown(f"- [{url}]({url})")


def add_site(raw_url: str, focus: str) -> None:
    """Crawl and index a site into its own namespace, then select it."""
    try:
        url = validate_url(raw_url)
    except ValueError as e:
        st.error(str(e))
        return
    new_ns = namespace_for(url)
    if new_ns in sources:                         # already indexed: just switch to it
        st.session_state.pending_source = new_ns
        st.rerun()
    if len(sources) - 1 >= MAX_SOURCES:
        st.error(f"This demo holds {MAX_SOURCES} sites and is full. Pick one from the list.")
        return
    if st.session_state.get("crawls", 0) >= CRAWLS_PER_SESSION:
        st.error(f"You can add {CRAWLS_PER_SESSION} sites per visit.")
        return
    st.session_state.crawls = st.session_state.get("crawls", 0) + 1

    signal = st.empty()  # progress signal for the stage animation
    messages_for = {
        "crawl": lambda d: f"🕷️ Crawling {d['url']}…",
        "documents": lambda d: f"📄 {d['pages']} pages crawled, {d['documents']} with readable text",
        "chunks": lambda d: f"✂️ Split into {d['chunks']} chunks",
        "store": lambda d: f"🧮 Embedding and saving {d['chunks']} chunks to Pinecone…",
    }
    with st.status(f"Indexing {url}", expanded=True) as status:
        def on_event(stage: str, **d) -> None:
            if stage in messages_for:
                status.write(messages_for[stage](d))
            with signal:
                background_ingest(stage, source=new_ns, **{k: v for k, v in d.items() if k != "url"})

        try:
            stats = ingest_site(url, new_ns, instructions=focus.strip() or None, on_event=on_event, **CRAWL)
        except Exception as e:
            status.update(label=f"Couldn't index {url}", state="error")
            st.error(str(e))
            return
        status.update(label=f"Indexed {stats['pages']} pages · {stats['chunks']} chunks", state="complete")

    get_sources.clear()
    st.session_state.pending_source = new_ns
    st.session_state.flash = f"Added **{new_ns}**: {stats['pages']} pages, {stats['chunks']} chunks. Ask away."
    st.rerun()


# --- Ask the docs -------------------------------------------------------------
with tab_ask:
    if flash := st.session_state.pop("flash", None):
        st.success(flash)
    if pinecone_error:
        st.error(f"Can't reach the Pinecone index `{os.environ.get('INDEX_NAME')}`. "
                 f"Check PINECONE_API_KEY and INDEX_NAME.\n\n`{pinecone_error[:300]}`")

    st.selectbox(
        "Ask about",
        list(sources),
        key="source",
        format_func=lambda n: f"{label(n)} · {sources[n]} chunks",
    )
    with st.expander("➕ Add a documentation site"):
        with st.form("add_site", clear_on_submit=True, border=False):
            url_in = st.text_input("Docs URL", placeholder="https://docs.pydantic.dev")
            focus_in = st.text_input("Focus (optional)", placeholder="e.g. only the API reference")
            submitted = st.form_submit_button("Crawl & index", type="primary")
        st.caption(f"Up to {CRAWL['limit']} pages per site · {CRAWLS_PER_SESSION} sites per visit.")
        if submitted:
            add_site(url_in, focus_in)

    if not messages:
        st.info(
            "Try: *What are deep agents?* · *When should I use LangGraph instead of LangChain?*"
            if ns == "" else f"Ask anything about **{label(ns)}**."
        )

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            show_sources(msg.get("sources", []))

    if prompt := st.chat_input(f"Ask about {label(ns)}…"):
        # Pass prior turns so follow-up questions have context.
        history = [{"role": m["role"], "content": m["content"]} for m in messages]
        messages.append({"role": "user", "content": prompt})
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

            answer, cited, found_docs = "", [], False
            start = time.time()
            try:
                for event in stream_llm(prompt, history, namespace=ns):
                    if event["type"] == "thinking":
                        step = 3 if found_docs else 1  # thinking after retrieval == writing
                        status.write("✍️ Reading the docs and writing an answer…" if found_docs else "🧠 Planning…")
                    elif event["type"] == "search":
                        step = 2
                        status.write(f"🔎 Searching Pinecone for *“{event['query']}”*")
                    elif event["type"] == "results":
                        step, found_docs = 3, True
                        new = [d.metadata.get("source", "unknown") for d in event["docs"]]
                        cited += [u for u in dict.fromkeys(new) if u not in cited]
                        status.write(f"📄 Found {len(event['docs'])} matching chunks")
                    elif event["type"] == "answer":
                        answer, step = event["text"], 4
                    with viz:
                        agent_pipeline(step)
                    with bg_signal:
                        background_step(min(step, 3), query=prompt)

                elapsed = time.time() - start
                with bg_signal:
                    background_step(4, query=prompt, sources=len(cited), seconds=f"{elapsed:.1f}")
                status.update(
                    label=f"Done in {elapsed:.1f}s · {len(cited)} sources",
                    state="complete",
                    expanded=False,
                )
            except Exception as e:
                status.update(label="Something went wrong", state="error")
                answer = f"⚠️ The agent failed: `{e}`"

            st.markdown(answer or "_No answer was produced._")
            show_sources(cited)

        messages.append({"role": "assistant", "content": answer, "sources": cited})


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
#### 1 · Ingestion (`ingestion.py`)
1. **Crawl:** Tavily crawls the docs site (built-in: LangChain; or one you add).
2. **Documents:** each page becomes a LangChain `Document` with its URL.
3. **Chunk:** pages are split into ~4,000-character chunks.
4. **Store:** chunks are embedded with OpenAI and saved to Pinecone
   in batches, concurrently, in a namespace per site.
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
