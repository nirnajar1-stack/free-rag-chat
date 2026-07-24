"""שרשרת שאלות ותשובות מבוססת RAG עם Groq ושליפה מ-Chroma."""

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

SYSTEM_PROMPT = """את/ה עוזר/ת לשאלות ותשובות על מסמכים במערכת RAG.
תמיד ענה בעברית ברורה וטבעית, אלא אם המשתמש ביקש במפורש שפה אחרת.

כללים שחובה לעקוב אחריהם:
1. ענה אך ורק על סמך המידע ב"הקשר" למטה. אל תשתמש בידע חיצוני.
2. אם אין ב"הקשר" מספיק מידע כדי לענות, השב בדיוק:
   "לא מצאתי את המידע הזה במסמכים שבאינדקס."
3. כשאת/ה משתמש/ת במידע מההקשר, ציין/י את שם קובץ המקור בסוגריים,
   למשל (מקור: report.pdf).
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
            name = doc.metadata.get("source_file") or doc.metadata.get("source", "לא ידוע")
            name = Path(str(name)).name
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered


def _format_context(docs: list[Document]) -> str:
    """עיצוב קטעים שנשלפו עבור ה-prompt."""
    if not docs:
        return "(לא נשלפו מסמכים רלוונטיים.)"

    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        name = doc.metadata.get("source_file") or Path(
            str(doc.metadata.get("source", "לא ידוע"))
        ).name
        page = doc.metadata.get("page")
        header = f"[קטע {i} | מקור: {name}"
        if page is not None:
            header += f" | עמוד: {int(page) + 1}"
        header += "]"
        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


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
    k: int = RETRIEVER_K,
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
        if NOT_FOUND_HE not in answer and "could not find" not in answer.lower():
            answer = NOT_FOUND_HE

    return RAGResponse(answer=answer.strip(), sources=sources)
