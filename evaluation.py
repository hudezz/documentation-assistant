"""Evaluation harness (WIP).

`target` is the function LangSmith's `evaluate()` calls per example: it runs
the real pipeline from core.py and returns only what evaluators score.
"""

from typing import Any, Dict, List

from core import run_llm


def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run one dataset example. Returns {"answer": str, "sources": list[str]}."""
    result = run_llm(inputs["question"])
    # URLs rather than full Documents: evaluators only check which pages were hit.
    sources = _unique_sources(result["context"])
    return {
        "answer": result["answer"],
        "sources": sources,
    }


def _unique_sources(docs) -> List[str]:
    """Source URLs in retrieval order, de-duplicated.

    The agent may call the retriever more than once, so pages can repeat.
    Order is kept for rank-based metrics.
    """
    seen = set()
    sources = []
    for doc in docs:
        url = doc.metadata.get("source", "unknown")
        if url not in seen:
            seen.add(url)
            sources.append(url)
    return sources


# Smoke test: uv run evaluation.py
if __name__ == "__main__":
    example_inputs = {"question": "What are deep agents?"}
    outputs = target(example_inputs)

    print("ANSWER:\n", outputs["answer"])
    print("\nSOURCES:")
    for url in outputs["sources"]:
        print(" -", url)
