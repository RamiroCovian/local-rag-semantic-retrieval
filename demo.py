"""Demo de las dos pruebas pedidas en la consigna."""

from __future__ import annotations

import asyncio

from rag import get_rag_response


async def run_demo() -> None:
    cases = [
        {
            "title": "1) Pregunta con respuesta en los documentos",
            "query": "¿En qué año se declaró la independencia de la Argentina?",
            "expect": "Debe mencionar 1816 y citar fuentes.",
        },
        {
            "title": "2) Pregunta trampa (no está en el contexto)",
            "query": "¿Quién inventó el dulce de leche?",
            "expect": 'Debe responder "No lo sé" sin alucinar.',
        },
    ]

    print("=" * 60)
    print("DEMO RAG — pruebas de la preentrega")
    print("=" * 60)

    for case in cases:
        print(f"\n{case['title']}")
        print(f"Pregunta: {case['query']}")
        print(f"Esperado: {case['expect']}")
        response = await get_rag_response(case["query"])
        print(f"Respuesta: {response.answer}")
        print(f"Referencias: {response.references}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(run_demo())
