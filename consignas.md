# Pre-entrega 3: Sistema de recuperación semántica local (RAG)

## ¿Qué debes construir?

Debes entregar un script o notebook de Python que implemente un flujo **End-to-End de RAG**. El sistema debe ser capaz de:

1. Recibir una consulta del usuario
2. Buscar la información relevante en una base vectorial previamente poblada
3. Generar una respuesta que utilice **exclusivamente** esa información

---

## Componentes a entregar

### 1. Módulo de Ingesta (Setup)

Una función que tome un conjunto de documentos (archivos `.txt` o `.md` simples), los fragmente (**chunking**) y los persista en una colección de **ChromaDB**.

### 2. Capa de Recuperación (Retriever)

Una lógica que convierta la pregunta del usuario en un embedding y recupere los fragmentos más relevantes.

### 3. Generación Grounded

Una cadena de **LangChain (LCEL)** que reciba los documentos recuperados y la pregunta, generando una respuesta.

> El prompt debe instruir al modelo a decir **"No lo sé"** si la respuesta no está en el contexto.

---

## Pasos sugeridos

1. **Puebla tu "Cerebro"**  
   Elige 3 o 4 archivos de texto sobre un tema específico (ej. manuales técnicos, apuntes de clase o normativas). Usa las técnicas de `RecursiveCharacterTextSplitter` que vimos en la Unidad 2 para procesarlos.

2. **Configura ChromaDB**  
   Inicializa el cliente persistente en una carpeta local (ej. `./vectorstore`). Asegúrate de usar el **mismo modelo de embeddings** para indexar y para consultar.

3. **Construye el Prompt de Sistema**  
   Diseña un prompt que actúe como un "filtro de veracidad".  
   Ejemplo: *"Eres un asistente técnico. Responde solo basándote en el CONTEXTO proporcionado. Si la respuesta no está allí, di que no tienes acceso a esa información."*

4. **Crea la Cadena LCEL**  
   Une el retriever con el transformador de documentos y el modelo de lenguaje. Recuerda que el output debe pasar por un `PydanticOutputParser`.

---

## Errores comunes a evitar

| Error | Descripción |
| --- | --- |
| **El "Contexto Infinito"** | No intentes pasar 50 fragmentos al LLM. Esto causará errores de límite de tokens o degradación de la atención (*Lost in the Middle*). Mantén un `top_k` de entre **3 y 5**. |
| **Embeddings no coincidentes** | Error #1. Si indexas con `OpenAIEmbeddings` y consultas con `HuggingFaceEmbeddings`, la distancia vectorial no tendrá sentido y los resultados serán aleatorios. |
| **Falta de persistencia** | Asegúrate de que el script verifique si la base de datos ya existe antes de volver a indexar todo, optimizando tiempo y costo de ejecución. |

---

## Qué entregás y en qué formato

| Campo | Detalle |
| --- | --- |
| **Tipo** | Código — un repositorio de GitHub |
| **Artefacto** | Repo con script de ingesta (chunking + ChromaDB), script/notebook de la cadena RAG asíncrona, dataset de ejemplo (`.txt`/`.md`) y `README.md` |
| **Qué NO hace falta** | No es un documento de análisis; el `README.md` alcanza como documentación |

### El repositorio debe contener

1. Script de ingesta de documentos
2. Script/Notebook con la cadena RAG asíncrona
3. Un dataset de ejemplo (`.txt` / `.md`)
4. Un archivo `README.md` explicando cómo ejecutar el sistema

---

## Entregable (checklist)

- [ ] Crear un entorno virtual e instalar las dependencias: `langchain`, `chromadb`, `openai` (o el provider seleccionado) y `pydantic`
- [ ] Implementar un script de carga que lea archivos de una carpeta `/data`, aplique chunking estratégico (**mínimo 500 tokens con 50 de overlap**) y los guarde en una instancia local de ChromaDB
- [ ] Definir una clase o función asíncrona `get_rag_response(query: str)` que:
  - [ ] a. Realice una búsqueda de similitud en ChromaDB
  - [ ] b. Construya un prompt incluyendo los fragmentos recuperados
  - [ ] c. Llame al LLM de forma asíncrona
  - [ ] d. Parsee la respuesta a un modelo Pydantic que incluya el **texto** y las **referencias**
- [ ] Realizar dos pruebas:
  - [ ] Una pregunta cuya respuesta **esté** en los documentos
  - [ ] Una "pregunta trampa" cuya respuesta **no** esté, verificando que el modelo no alucine
- [ ] Subir el código **sin** incluir llaves de API (usar variables de entorno / `.env`)
