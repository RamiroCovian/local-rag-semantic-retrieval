"""Ingesta de documentos: carga, chunking y persistencia en ChromaDB."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    COLLECTION_NAME,
    DATA_DIR,
    VECTORSTORE_DIR,
    get_embeddings,
)


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Lee archivos .txt y .md de /data y los convierte en Documents."""
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de datos: {data_dir}")

    patterns = ("*.md", "*.txt")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(data_dir.glob(pattern)))

    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos .md/.txt en {data_dir}"
        )

    documents: list[Document] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "path": str(path)},
            )
        )

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Fragmenta documentos con RecursiveCharacterTextSplitter (por tokens)."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )
    return splitter.split_documents(documents)


def vectorstore_has_data(persist_directory: Path = VECTORSTORE_DIR) -> bool:
    """True si ya hay una colección persistida con documentos."""
    if not persist_directory.exists() or not any(persist_directory.iterdir()):
        return False

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
    )
    result = store.get(limit=1)
    return bool(result.get("ids"))


def get_vectorstore(persist_directory: Path = VECTORSTORE_DIR) -> Chroma:
    """Abre el vectorstore persistente existente."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
    )


def ingest_documents(
    *,
    force: bool = False,
    data_dir: Path = DATA_DIR,
    persist_directory: Path = VECTORSTORE_DIR,
) -> Chroma:
    """Indexa documentos en ChromaDB. Omite reindexado si ya existe (salvo force)."""
    if not force and vectorstore_has_data(persist_directory):
        print(
            f"[ingest] Vectorstore ya existe en {persist_directory}. "
            "No se reindexa. Usá --force para forzar."
        )
        return get_vectorstore(persist_directory)

    documents = load_documents(data_dir)
    chunks = chunk_documents(documents)

    print(f"[ingest] Documentos cargados: {len(documents)}")
    print(f"[ingest] Chunks generados: {len(chunks)}")
    print(f"[ingest] Persistiendo en: {persist_directory}")

    if force and persist_directory.exists():
        shutil.rmtree(persist_directory)

    persist_directory.mkdir(parents=True, exist_ok=True)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory),
    )

    print(f"[ingest] Listo. Colección '{COLLECTION_NAME}' con {len(chunks)} chunks.")
    return store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingesta de documentos hacia ChromaDB local."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reindexa aunque el vectorstore ya exista.",
    )
    args = parser.parse_args()
    ingest_documents(force=args.force)


if __name__ == "__main__":
    main()
