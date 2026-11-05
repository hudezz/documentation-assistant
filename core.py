"""RAG agent over documentation stored in Pinecone (one namespace per site).

`run_llm` returns a finished answer; `stream_llm` yields progress events for
the UI. Both use the same agent: one retrieval tool + a citing system prompt.
"""

import os
from typing import Any, Dict, Iterator, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Must match the model used in ingestion.py, or query vectors won't line up.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

model = init_chat_model("gpt-5.2", model_provider="openai")


def make_retriever_tool(namespace: str = ""):
    """retrieve_context tool bound to one site's namespace ("" = built-in LangChain docs)."""
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings, namespace=namespace or None
    )

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant documentation to help answer the user's question."""
        retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(query)
        serialized = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        # content goes to the model; the artifact (raw Documents) is kept for citations
        return serialized, retrieved_docs

    return retrieve_context


SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers questions about a software project's documentation. "
    "You have access to a tool that retrieves relevant documentation. "
    "Use the tool to find relevant information before answering questions. "
    "Always cite the sources you use in your answers. "
    "If you cannot find the answer in the retrieved documentation, say so."
)


def stream_llm(query: str, history: List[Dict[str, str]] | None = None,
               namespace: str = "") -> Iterator[Dict[str, Any]]:
    """Run the agent and yield a progress event after each step.

    Events:
        {"type": "thinking"}
        {"type": "search",  "query": str}
        {"type": "results", "docs": list[Document]}
        {"type": "answer",  "text": str}

    `history` is prior turns as [{"role": ..., "content": ...}] for follow-ups;
    `namespace` picks which indexed site to search.
    """
    agent = create_agent(model, tools=[make_retriever_tool(namespace)], system_prompt=SYSTEM_PROMPT)
    messages = list(history or []) + [{"role": "user", "content": query}]

    yield {"type": "thinking"}

    # "updates" yields each node's output as it finishes: the model node emits
    # AIMessages (tool calls or the final answer), the tools node ToolMessages.
    for update in agent.stream({"messages": messages}, stream_mode="updates"):
        for node_output in update.values():
            for message in (node_output or {}).get("messages", []):
                if isinstance(message, ToolMessage):
                    docs = message.artifact if isinstance(message.artifact, list) else []
                    yield {"type": "results", "docs": docs}
                    yield {"type": "thinking"}
                elif getattr(message, "tool_calls", None):
                    for call in message.tool_calls:
                        yield {"type": "search", "query": call["args"].get("query", "")}
                elif message.content:
                    yield {"type": "answer", "text": message.content}


def run_llm(query: str, namespace: str = "") -> Dict[str, Any]:
    """Answer `query`. Returns {"answer": str, "context": list[Document]}."""
    agent = create_agent(model, tools=[make_retriever_tool(namespace)], system_prompt=SYSTEM_PROMPT)
    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})
    answer = response["messages"][-1].content

    # Retrieved Documents ride along as ToolMessage artifacts.
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }


if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)