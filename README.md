# Document RAG
<img width="1269" height="598" alt="Screenshot 2026-08-23 at 6 22 08 pm" src="https://github.com/user-attachments/assets/ebc53385-5a32-4adc-9d30-209963bf7236" />

This is a local Streamlit RAG application. It accepts up to five documents,
stores their chunks and embeddings in persistent Chroma, and answers questions
using a selected xAI or Groq chat model.

## Setup

```bash
uv sync
cp .env.example .env
```

Add the API key for the provider you plan to use to `.env`:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key
GROQ_MODEL_NAME=openai/gpt-oss-20b
```

Or use xAI:

```env
LLM_PROVIDER=xai
XAI_API_KEY=your_xai_key
XAI_MODEL_NAME=grok-3-mini
```

## Run

```bash
PYTHONPATH=. uv run streamlit run app/main.py
```

The application opens at `http://localhost:8501`. Indexed chunks are stored in
`chroma_db/`, which is intentionally ignored by Git.

Chunk size and chunk overlap can be configured from the Chunking section in the
Streamlit sidebar before indexing documents. The overlap must be smaller than
the chunk size.

The sidebar also supports Recursive character, Character, Token, Markdown
header, Python code, and Semantic chunking strategies. Semantic chunking uses
the local embedding model to split text by meaning.

Each uploaded file must be 50 MB or smaller.

## Supported files

TXT, Markdown, CSV, JSON, HTML, XML, Python, PDF, DOCX, and XLSX files are
supported. Scanned PDFs and images require OCR and are not currently supported.

## Architecture

`loaders.py` extracts and chunks files. `vector_store.py` persists and searches
embeddings. `rag_chain.py` builds grounded prompts and calls the selected
provider. `main.py` provides the Streamlit interface.

Indexing, document removal, and question-answering failures are shown as safe
alerts in the web interface. Detailed exception text is not displayed to avoid
leaking provider or document information.
