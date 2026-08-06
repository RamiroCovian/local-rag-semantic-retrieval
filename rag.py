"""Cadena RAG asíncrona: retrieval + generación grounded + parser Pydantic."""

from __future__ import annotations

import argparse
import asyncio

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config import TOP_K, get_chat_model
from ingest import get_vectorstore
from schemas import RAGResponse

SYSTEM_PROMPT = """Eres un asistente técnico sobre historia argentina.
Responde ÚNICAMENTE basándote en el CONTEXTO proporcionado.
Si la respuesta no está en el contexto, di exactamente: "No lo sé".
No inventes fechas, nombres ni hechos.
En references incluí solo los nombres de archivo (source) del contexto que usaste.
Si respondiste "No lo sé", devolvé references como lista vacía.

{format_instructions}
"""

USER_PROMPT = """CONTEXTO:
{context}

PREGUNTA:
{question}
"""


def format_docs(docs: list[Document]) -> str:
    """Une fragmentos recuperados con su fuente visible."""
    if not docs:
        return "No hay contexto disponible."

    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "desconocido")
        parts.append(f"[Fuente {index}: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def get_retriever(k: int = TOP_K):
    """Retriever de similitud sobre ChromaDB persistente."""
    store = get_vectorstore()
    return store.as_retriever(search_kwargs={"k": k})


def build_rag_chain(k: int = TOP_K):
    """Construye la cadena LCEL: retriever → prompt → LLM → Pydantic parser."""
    parser = PydanticOutputParser(pydantic_object=RAGResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    retriever = get_retriever(k=k)
    llm = get_chat_model()

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | parser
    )


async def get_rag_response(query: str, *, k: int = TOP_K) -> RAGResponse:
    """Busca contexto, genera respuesta grounded y parsea a RAGResponse."""
    if not query or not query.strip():
        raise ValueError("La consulta no puede estar vacía.")

    chain = build_rag_chain(k=k)
    return await chain.ainvoke(query.strip())


async def _run_cli(query: str) -> None:
    response = await get_rag_response(query)
    print(f"\nRespuesta: {response.answer}")
    print(f"Referencias: {response.references}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta el sistema RAG local.")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Pregunta opcional. Si no se pasa, se pide por consola.",
    )
    args = parser.parse_args()

    if args.query:
        asyncio.run(_run_cli(args.query))
        return

    print("RAG local — historia argentina")
    print('Escribí tu pregunta (o "salir" para terminar).\n')

    while True:
        query = input("Pregunta: ").strip()
        if not query:
            continue
        if query.lower() in {"salir", "exit", "q"}:
            print("Listo.")
            break
        asyncio.run(_run_cli(query))


if __name__ == "__main__":
    main()
