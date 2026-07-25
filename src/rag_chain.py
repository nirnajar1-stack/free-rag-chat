"""שרשרת שאלות ותשובות מבוססת RAG עם Groq ושליפה מ-Chroma."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import (
    GROQ_MODEL_NAME,
    RETRIEVER_FETCH_K,
    RETRIEVER_MAX_SOURCES,
    RETRIEVER_MIN_SCORE,
    validate_groq_api_key,
)
from src.vectorstore import get_chunk_count, retrieve_relevant_documents

SYSTEM_PROMPT = """את/ה עוזר/ת לשאלות ותשובות על מסמכים במערכת RAG.
תמיד ענה בעברית ברורה וטבעית, אלא אם המשתמש ביקש במפורש שפה אחרת.

כללים שחובה לעקוב אחריהם:
1. ענה אך ורק על סמך המידע ב"הקשר" למטה. אל תשתמש בידע חיצוני.
2. אם אין ב"הקשר" מספיק מידע כדי לענות, השב בדיוק:
   "לא מצאתי את המידע הזה במסמכים שבאינדקס."
3. כשאת/ה משתמש/ת במידע מההקשר, ציין/י רק את שם הקובץ שבאמת שימש אותך,
   למשל (מקור: report.pdf). אל תציין קבצים שלא תרמו לתשובה.
4. היה/י תמציתי/ת ומדויק/ת. העדף/י ציטוט או ניסוח קרוב להקשר.
5. אם מקורות סותרים זה את זה, ציין/י את הסתירה ואת המקורות.

הקשר:
{context}
"""

NOT_FOUND_HE = "לא מצאתי את המידע הזה במסמכים שבאינדקס."


@dataclass
class RAGResponse:
    """תשובה מובנית יחד עם קטעי המקור שנשלפו."""

    answer: str
    sources: list[Document] = field(default_factory=list)

    @property
    def source_files(self) -> list[str]:
        """שמות קבצי מקור ייחודיים לפי סדר השליפה."""
        seen: set[str] = set()
        ordered: list[str] = []
        for doc in self.sources:
            name = _source_name(doc)
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered


def _source_name(doc: Document) -> str:
    name = doc.metadata.get("source_file") or doc.metadata.get("source", "לא ידוע")
    return Path(str(name)).name


def _format_context(docs: list[Document]) -> str:
    """עיצוב קטעים שנשלפו עבור ה-prompt."""
    if not docs:
        return "(לא נשלפו מסמכים רלוונטיים.)"

    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        name = _source_name(doc)
        page = doc.metadata.get("page")
        header = f"[קטע {i} | מקור: {name}"
        if page is not None:
            header += f" | עמוד: {int(page) + 1}"
        header += "]"
        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def _filter_sources_for_display(answer: str, sources: list[Document]) -> list[Document]:
    """
    Keep sources that were actually cited, otherwise the top-scoring ones only.
    Deduplicate near-identical chunks from the same file.
    """
    if not sources:
        return []

    cited = [doc for doc in sources if _source_name(doc) in answer]
    pool = cited if cited else sources

    selected: list[Document] = []
    seen_keys: set[str] = set()
    for doc in pool:
        name = _source_name(doc)
        page = doc.metadata.get("page")
        preview = " ".join(doc.page_content.split())[:120]
        key = f"{name}|{page}|{preview}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(doc)
        if len(selected) >= RETRIEVER_MAX_SOURCES:
            break
    return selected


def get_llm() -> ChatGroq:
    """יצירת לקוח ChatGroq."""
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
    """בניית הודעות: מערכת + היסטוריה אופציונלית + שאלה."""
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
    chat_history: list[dict[str, Any]] | None = None,
) -> RAGResponse:
    """שליפת קטעים רלוונטיים ותשובה מבוססת-הקשר דרך Groq."""
    if get_chunk_count() == 0:
        return RAGResponse(
            answer=(
                "מאגר הווקטורים ריק. העלה/י מסמכים בסרגל הצד ולחץ/י על "
                "**סנכרון / בניית אינדקס**."
            ),
            sources=[],
        )

    question = (question or "").strip()
    if not question:
        return RAGResponse(answer="נא להזין שאלה.", sources=[])

    sources = retrieve_relevant_documents(
        question,
        fetch_k=RETRIEVER_FETCH_K,
        min_score=RETRIEVER_MIN_SCORE,
        max_docs=RETRIEVER_MAX_SOURCES,
    )
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
        if NOT_FOUND_HE not in answer and "could not find" not in answer.lower():
            answer = NOT_FOUND_HE

    display_sources = _filter_sources_for_display(answer.strip(), sources)
    return RAGResponse(answer=answer.strip(), sources=display_sources)
