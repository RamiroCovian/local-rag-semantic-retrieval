"""Configuración central del proyecto RAG."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"

COLLECTION_NAME = "historia_argentina"

# Chunking (consigna: mínimo 500 tokens con 50 de overlap)
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# Recuperación
TOP_K = 4

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-3-5-haiku-latest")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")

OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)
GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "models/gemini-embedding-2",
)


def get_embeddings():
    """Devuelve el modelo de embeddings según EMBEDDING_PROVIDER.

    Debe ser el mismo al indexar y al consultar.
    """
    if EMBEDDING_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "Falta OPENAI_API_KEY en .env para generar embeddings."
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
        )

    if EMBEDDING_PROVIDER in {"gemini", "google"}:
        if not GOOGLE_API_KEY:
            raise ValueError(
                "Falta GOOGLE_API_KEY en .env para generar embeddings."
            )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )

    raise ValueError(
        f"EMBEDDING_PROVIDER no soportado: {EMBEDDING_PROVIDER!r}. "
        "Usá 'gemini' u 'openai'."
    )
