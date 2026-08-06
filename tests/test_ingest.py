"""Tests unitarios de ingesta (sin llamar al LLM)."""

from ingest import chunk_documents, load_documents
from schemas import RAGResponse


def test_load_documents_lee_dataset(data_dir):
    docs = load_documents(data_dir)

    assert len(docs) == 4
    sources = {doc.metadata["source"] for doc in docs}
    assert "independencia.md" in sources
    assert "siglo-xx.md" in sources


def test_chunk_documents_genera_fragmentos(data_dir):
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)

    assert len(chunks) >= len(docs)
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_rag_response_schema():
    response = RAGResponse(
        answer="No lo sé",
        references=[],
    )

    assert response.answer == "No lo sé"
    assert response.references == []
