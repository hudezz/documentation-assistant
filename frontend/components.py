"""Streamlit wrappers around the custom front-end assets.

- agent_viz/: pipeline + replay animation, rendered in an iframe via st.iframe.
- static/: the motion-graphics stage, injected into the page itself.
"""

import html
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

VIZ_DIR = Path(__file__).parent / "agent_viz"
DEMO_RUNS_FILE = Path(__file__).parent / "demo_runs.json"


@lru_cache(maxsize=1)
def _template() -> str:
    """index.html with style.css and script.js inlined."""
    page = (VIZ_DIR / "index.html").read_text()
    page = page.replace("{{STYLE}}", (VIZ_DIR / "style.css").read_text())
    page = page.replace("{{SCRIPT}}", (VIZ_DIR / "script.js").read_text())
    return page


def _render(config: Dict[str, Any], height: int) -> None:
    # Escape "</" so config strings can't terminate the inline <script>.
    # (Recorded answers are LLM output; script.js only renders them as text.)
    config_json = json.dumps(config).replace("</", "<\\/")
    st.iframe(_template().replace("{{CONFIG}}", config_json), height=height)


def load_demo_runs() -> List[Dict[str, Any]]:
    """Runs recorded by record_demo.py; empty if none exist yet."""
    if not DEMO_RUNS_FILE.exists():
        return []
    return json.loads(DEMO_RUNS_FILE.read_text())


def agent_pipeline(step: int) -> None:
    """Static 4-step pipeline with `step` active (4 = complete)."""
    _render({"mode": "static", "step": step}, height=104)


def live_agent_demo(runs: List[Dict[str, Any]]) -> None:
    """Loop the recorded runs client-side (no API calls)."""
    _render({"mode": "loop", "runs": runs}, height=640)


# --- Motion-graphics stage ----------------------------------------------------
# Served from /app/static/ (enableStaticServing). Loaded by URL because st.html's
# DOMPurify pass drops inline scripts containing "<"-letter sequences.
STATIC_DIR = Path(__file__).parent / "static"


def _version() -> int:
    """Cache-busting token: mtime of the stage assets."""
    return int(max((STATIC_DIR / f).stat().st_mtime for f in ("background.js", "background.css")))


def _signal(**data_attrs: Dict[str, Any]) -> str:
    attrs = " ".join(f'data-{k}="{html.escape(json.dumps(v))}"' for k, v in data_attrs.items())
    return f'<div class="docs-bg-signal" {attrs}></div>'


def page_background(data: Dict[str, Any]) -> None:
    """Load the stage script (idempotent across reruns) and send it `data`."""
    v = _version()
    st.html(
        _signal(config=data) + f'<script src="/app/static/background.js?v={v}"></script>',
        unsafe_allow_javascript=True,
    )


def background_step(step: int, **info: Any) -> None:
    """Report agent progress to the stage.

    step: 1 planning, 2 searching, 3 writing, 4 done.
    info: query; on step 4 also sources and seconds.
    """
    st.html(_signal(live={"step": step, **info}))


def background_ingest(stage: str, **info: Any) -> None:
    """Report indexing progress to the stage (stage names from ingestion.ingest_site)."""
    st.html(_signal(live={"ingest": stage, **info}))
