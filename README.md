# local-rag-semantic-retrieval

Sistema RAG local (Retrieval-Augmented Generation) sobre documentos de **historia argentina**.

Flujo end-to-end:

1. Ingesta de `.md` / `.txt` con chunking
2. Persistencia en **ChromaDB** (`./vectorstore`)
3. Recuperación semántica + generación grounded con LangChain (LCEL)
4. Respuesta parseada a Pydantic (`answer` + `references`)

Si la respuesta no está en el contexto, el modelo responde **"No lo sé"**.

---

## Requisitos

- Python 3.11+
- Una API key del proveedor que uses (`gemini`, `openai` o `anthropic`)

---

## Setup

```powershell
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Editá `.env` y completá al menos:

```env
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
GOOGLE_API_KEY=tu-key-aqui
GEMINI_CHAT_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2
```

> No subas `.env` al repositorio. `.gitignore` ya lo excluye.

### Providers soportados

| Variable | Valores |
| --- | --- |
| `LLM_PROVIDER` | `gemini` \| `openai` \| `anthropic` |
| `EMBEDDING_PROVIDER` | `gemini` \| `openai` |

Usá el **mismo** `EMBEDDING_PROVIDER` / modelo al indexar y al consultar.

---

## Dataset

Los documentos están en `data/`:

- `independencia.md`
- `organizacion-nacional.md`
- `inmigracion-y-modernizacion.md`
- `siglo-xx.md`

---

## Uso

### 1) Ingesta (chunking + ChromaDB)

```powershell
python ingest.py
```

- Chunking: **500 tokens** con **50 de overlap** (`RecursiveCharacterTextSplitter`)
- Persistencia: `./vectorstore`
- Si el vectorstore ya existe, no reindexa (salvo `--force`)

```powershell
python ingest.py --force
```

### 2) Consulta RAG (async)

```powershell
python rag.py "¿En qué año se declaró la independencia de la Argentina?"
```

Función principal:

```python
from rag import get_rag_response

response = await get_rag_response("tu pregunta")
# response.answer
# response.references
```

### 3) Demo de las dos pruebas de la consigna

```powershell
python demo.py
```

1. Pregunta con respuesta en los documentos  
2. Pregunta trampa (debe responder "No lo sé")

### 4) Tests

```powershell
pytest -v
```

---

## Estructura

```text
.
├── data/                 # Dataset de ejemplo
├── tests/                # Pytest (ingesta + RAG)
├── vectorstore/          # ChromaDB local (no versionado)
├── config.py             # Providers, paths, embeddings, LLM
├── ingest.py             # Carga + chunking + persistencia
├── rag.py                # Cadena LCEL async + get_rag_response
├── schemas.py            # RAGResponse (Pydantic)
├── demo.py               # Demo de las 2 pruebas
├── requirements.txt
├── .env.example
└── README.md
```

---

## Notas

- `top_k` por defecto: **4** (evitar contexto excesivo)
- El prompt actúa como filtro de veracidad: solo responde con el contexto recuperado
- La salida pasa por `PydanticOutputParser` hacia `RAGResponse`
