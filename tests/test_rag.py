"""Tests de la consigna: pregunta con contexto y pregunta trampa."""

import pytest
from langchain_core.documents import Document

from config import GOOGLE_API_KEY, OPENAI_API_KEY, VECTORSTORE_DIR
from rag import format_docs, get_rag_response
from schemas import RAGResponse


def _has_rag_runtime() -> bool:
    has_key = bool((GOOGLE_API_KEY or OPENAI_API_KEY or "").strip())
    has_store = VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir())
    return has_key and has_store


requires_rag_runtime = pytest.mark.skipif(
    not _has_rag_runtime(),
    reason="Requiere API key en .env y vectorstore indexado (python ingest.py).",
)


def test_format_docs_incluye_fuente():
    docs = [
        Document(
            page_content="La independencia se declaró en 1816.",
            metadata={"source": "independencia.md"},
        )
    ]

    formatted = format_docs(docs)

    assert "independencia.md" in formatted
    assert "1816" in formatted


@pytest.mark.asyncio
async def test_get_rag_response_rechaza_query_vacia():
    with pytest.raises(ValueError, match="vacía"):
        await get_rag_response("   ")


@requires_rag_runtime
@pytest.mark.asyncio
async def test_pregunta_con_contexto_en_documentos():
    """La respuesta debe estar grounded en el dataset."""
    response = await get_rag_response(
        "¿En qué año se declaró la independencia de la Argentina?"
    )

    assert isinstance(response, RAGResponse)
    assert "1816" in response.answer
    assert response.references
    assert any("independencia" in ref.lower() for ref in response.references)


@requires_rag_runtime
@pytest.mark.asyncio
async def test_pregunta_trampa_no_alucina():
    """Si no está en el contexto, debe decir que no lo sabe."""
    response = await get_rag_response("¿Quién inventó el dulce de leche?")

    assert isinstance(response, RAGResponse)
    assert "no lo sé" in response.answer.lower()
    assert response.references == []
