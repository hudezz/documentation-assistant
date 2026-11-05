"""Record real agent runs for the UI's "Live agent" tab.

The tab replays these recordings client-side so it can loop indefinitely
without calling the API. Re-run after re-ingesting:

    uv run record_demo.py
"""

import json
import re
import time

from core import stream_llm

DEMO_QUESTIONS = [
    "What are deep agents?",
    "When should I use LangGraph instead of LangChain?",
    "How do subagents work in Deep Agents?",
    "How do I trace my agent with LangSmith?",
]

MAX_ANSWER_CHARS = 480  # answers are typed out on screen; keep them short


def clean_for_demo(text: str) -> str:
    """Strip markdown and inline citations so the typed-out answer reads cleanly."""
    text = re.sub(r"【[^】]*】", "", text)               # 【https://...】 citations
    text = re.sub(r"\[Source:[^\]]*\]", "", text)       # [Source: ...] citations
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"[*_`#>]", "", text)                 # bold, italics, code, headings
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_ANSWER_CHARS:
        text = text[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def record(question: str) -> dict:
    """Run one question and capture its searches, sources, answer and duration."""
    start = time.time()
    searches, sources, answer = [], [], ""

    for event in stream_llm(question):
        if event["type"] == "search":
            searches.append(event["query"])
        elif event["type"] == "results":
            for doc in event["docs"]:
                url = doc.metadata.get("source", "unknown")
                if url not in sources:
                    sources.append(url)
        elif event["type"] == "answer":
            answer = event["text"]

    return {
        "question": question,
        "searches": searches,
        "sources": sources[:6],
        "answer": clean_for_demo(answer),
        "seconds": round(time.time() - start, 1),
    }


if __name__ == "__main__":
    runs = []
    for q in DEMO_QUESTIONS:
        print(f"Recording: {q}")
        runs.append(record(q))
        print(f"  {len(runs[-1]['sources'])} sources, {runs[-1]['seconds']}s")

    with open("frontend/demo_runs.json", "w") as f:
        json.dump(runs, f, indent=2, ensure_ascii=False)
    print("Saved frontend/demo_runs.json")
