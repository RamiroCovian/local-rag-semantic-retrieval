"""Esquemas Pydantic de salida del RAG."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    """Respuesta grounded del sistema RAG."""

    answer: str = Field(
        description="Respuesta al usuario basada solo en el contexto recuperado."
    )
    references: list[str] = Field(
        default_factory=list,
        description="Nombres de archivos fuente usados para responder.",
    )
