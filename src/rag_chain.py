"""RAG question-answering chain powered by Groq + Chroma retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import GROQ_MODEL_NAME, RETRIEVER_K, validate_groq_api_key
from src.vectorstore import get_chunk_count, get_retriever

SYSTEM_PROMPT = """You are a careful document Q&A assistant for a RAG system.

Rules you MUST follow:
1. Answer ONLY using the information in the Context below. Do not use outside knowledge.
2. If the Context does not contain enough information to answer, reply exactly with:
   "I could not find this information in the indexed documents."
3. When you use information from the Context, cite the source document name(s)
   in parentheses, e.g. (source: report.pdf).
4. Be concise and accurate. Prefer quoting or closely paraphrasing the Context.
5. If multiple sources conflict, mention the conflict and cite each source.

Context:
{context}
"""


@dataclass
class RAGResponse:
    """Structured answer plus the retrieved source chunks."""

    answer: str
    sources: list[Document] = field(default_factory=list)

    @property
    def source_files(self) -> list[str]:
        """Unique source filenames, preserving retrieval order."""
        seen: set[str] = set()
        ordered: list[str] = []
        for doc in self.sources:
            name = doc.metadata.get("source_file") or doc.metadata.get("source", "unknown")
            name = Path(str(name)).name
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered


def _format_context(docs: list[Document]) -> str:
    """Format retrieved chunks for the system prompt."""
    if not docs:
        return "(No relevant documents were retrieved.)"

    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        name = doc.metadata.get("source_file") or Path(
            str(doc.metadata.get("source", "unknown"))
        ).name
        page = doc.metadata.get("page")
        header = f"[Snippet {i} | source: {name}"
        if page is not None:
            header += f" | page: {int(page) + 1}"
        header += "]"
        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def get_llm() -> ChatGroq:
    """Create a ChatGroq client using the free Groq API."""
    api_key = validate_groq_api_key()
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=api_key,
        temperature=0.1,
        max_tokens=2048,
    )


def _build_messages(
    question: str,
    context: str,
    chat_history: list[dict[str, Any]] | None,
) -> list[Any]:
    """Build the message list: system (with context) + optional history + question."""
    system = SystemMessage(content=SYSTEM_PROMPT.format(context=context))

    if not chat_history:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        return prompt.format_messages(context=context, question=question)

    messages: list[Any] = [system]
    for turn in chat_history[-6:]:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=question))
    return messages


def ask_question(
    question: str,
    *,
    k: int = RETRIEVER_K,
    chat_history: list[dict[str, Any]] | None = None,
) -> RAGResponse:
    """
    Retrieve relevant chunks and generate a grounded answer via Groq.

    ``chat_history`` is optional prior turns (role/content dicts) for light
    conversational continuity; answers remain grounded in retrieved context.
    """
    if get_chunk_count() == 0:
        return RAGResponse(
            answer=(
                "The vector database is empty. Use **Re-index / Sync Documents** "
                "in the sidebar to load files from your docs folder first."
            ),
            sources=[],
        )

    question = (question or "").strip()
    if not question:
        return RAGResponse(answer="Please enter a question.", sources=[])

    retriever = get_retriever(k=k)
    sources: list[Document] = list(retriever.invoke(question))
    context = _format_context(sources)

    llm = get_llm()
    messages = _build_messages(question, context, chat_history)
    response = llm.invoke(messages)
    answer = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )

    if not sources:
        not_found = "I could not find this information in the indexed documents."
        if not_found.lower() not in answer.lower():
            answer = not_found

    return RAGResponse(answer=answer.strip(), sources=sources)
