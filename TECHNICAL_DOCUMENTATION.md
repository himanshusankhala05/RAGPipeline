# Document RAG Application

## 1. Purpose

Document RAG is a Python application that lets a user upload up to five documents, index their contents in a local Chroma vector database, and ask questions about those documents through a Streamlit interface.

The application uses a selected Groq or xAI chat model to generate answers from the document context retrieved by Chroma.

## 2. Technology Stack

- Python 3.11 or newer
- UV for dependency and virtual-environment management
- Streamlit for the user interface
- ChromaDB for persistent local vector storage
- Sentence Transformers for local text embeddings
- LangChain text splitters for chunking
- OpenAI Python SDK for OpenAI-compatible Groq and xAI APIs
- pypdf for PDF extraction
- python-docx for DOCX extraction
- openpyxl for XLSX extraction
- python-dotenv for environment variables

## 3. Architecture

```text
Streamlit UI
    |
    v
Uploaded files
    |
    v
Temporary file bridge
    |
    v
Document loaders and text extraction
    |
    v
Recursive character chunking
    |
    v
Local Sentence Transformer embeddings
    |
    v
Persistent Chroma collection
    |
    v
Similarity search
    |
    v
Grounded prompt
    |
    v
Groq or xAI chat model
    |
    v
Formatted answer and sources
```

The project follows a modular application architecture:

- `app/main.py`: Streamlit interface and application orchestration
- `app/loaders.py`: File extraction, chunking, metadata, and IDs
- `app/vector_store.py`: Chroma connection, indexing, search, listing, and deletion
- `app/rag_chain.py`: Retrieval, prompt construction, answer cleanup, and generation
- `app/llm_client.py`: Provider and model client selection
- `app/config.py`: Runtime settings and environment-variable loading
- `tests/test_rag.py`: Unit tests for pure ingestion and RAG logic

## 4. Document Ingestion

When a user clicks `Index documents`, `main.py` performs the following steps:

1. Receives Streamlit `UploadedFile` objects.
2. Saves each upload temporarily while preserving its extension.
3. Passes temporary paths to `load_documents()`.
4. Extracts text according to the file type.
5. Splits text into chunks using the chunk size and overlap selected in the Streamlit sidebar. The defaults are approximately 1,000 characters with 150 characters of overlap.
6. Adds source filename, file type, chunk index, and SHA-256 document hash as metadata.
7. Sends chunks, metadata, and IDs to Chroma.
8. Deletes temporary files in a `finally` block.

The original uploaded files are not permanently stored by the application. Their chunks, embeddings, and metadata remain in Chroma.

### Chunking Strategies

The sidebar lets the user choose one of these strategies before indexing:

- Recursive character: recursively splits using natural separators.
- Character: splits using the configured separator and character length.
- Token: splits by token count using `tiktoken`.
- Markdown header: keeps Markdown header structure in the resulting chunks.
- Python code: splits source code using Python-aware separators.
- Semantic: splits when the meaning changes, using the local embedding model.

Chunk size and overlap apply to the character, recursive character, token, and
Python code strategies. Markdown header and semantic splitting use their own
boundaries; the sidebar displays a note for semantic mode.

## 5. Supported File Types

The current implementation supports:

- TXT
- Markdown
- CSV
- JSON
- HTML
- XML
- Python source files
- PDF
- DOCX
- XLSX

Scanned PDFs and image files are not supported because they require OCR.

## 6. Chroma Vector Database

Chroma uses a persistent local client and the `documents` collection. Data is stored in:

```text
chroma_db/
```

The embedding model is:

```text
all-MiniLM-L6-v2
```

During indexing, Chroma stores:

- Original text chunks
- Vector embeddings
- Source filename
- File type
- Chunk index
- Document content hash

During a search, the question is embedded and compared with stored vectors. The original documents are not read again during search.

The vector-store module supports:

- Creating or opening the collection
- Upserting chunks
- Similarity search
- Safe empty-collection handling
- Counting stored chunks
- Listing unique indexed source filenames
- Deleting all chunks for a source filename
- Listing indexed document hashes

## 7. Duplicate Protection

The application calculates a SHA-256 hash from each uploaded file's bytes.

An upload is rejected when either condition is true:

- Its content hash already exists in Chroma.
- Its filename already exists in Chroma or appears more than once in the current upload.

This prevents repeated indexing of identical content and duplicate filenames.

## 8. Document Deletion

Removing a file from the Streamlit uploader does not delete its Chroma data.

Users delete indexed documents explicitly from the collapsed `Indexed documents` section. The Remove action calls `delete_documents_by_source()` and deletes all chunks whose metadata contains that source filename.

This design prevents a browser refresh or uploader reset from unexpectedly deleting indexed data.

## 9. RAG Question Flow

When the user asks a question:

1. The question is embedded and searched in Chroma.
2. The most relevant chunks are selected.
3. The chunks and source names are added to a prompt.
4. The selected provider and model receive the prompt.
5. The generated response is cleaned of `<think>`, `<thinking>`, and `<analysis>` blocks.
6. Streamlit renders the answer as Markdown.
7. Unique source filenames are shown in an expandable Sources section.

The system prompt instructs the model to answer only from the retrieved context and to state when the answer is not present in the uploaded documents.

## 10. LLM Providers and Models

The application uses the OpenAI-compatible Python SDK. The SDK is only the client interface; requests are sent to the selected provider's API endpoint.

### xAI

```env
LLM_PROVIDER=xai
XAI_API_KEY=your_xai_key
XAI_MODEL_NAME=grok-3-mini
```

Endpoint:

```text
https://api.x.ai/v1
```

### Groq

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key
GROQ_MODEL_NAME=openai/gpt-oss-20b
```

Endpoint:

```text
https://api.groq.com/openai/v1
```

The Streamlit sidebar currently exposes the configured xAI model and these Groq model choices:

- `openai/gpt-oss-20b`
- `openai/gpt-oss-120b`
- `qwen/qwen3.6-27b`

Provider selection is implemented in `app/llm_client.py`, so adding another OpenAI-compatible provider does not require changing the RAG retrieval logic.

## 11. Configuration

Runtime settings are defined in `app/config.py`:

- Project root
- Documents directory
- Chroma database path
- Chroma collection name
- Embedding model
- Maximum document count
- Chunk size
- Chunk overlap
- Active LLM provider
- Provider API keys
- Provider model names
- Provider endpoints

Secrets belong in `.env`. The `.env` file is ignored by Git. `.env.example` provides a safe template.

The Streamlit sidebar exposes `Chunk size` and `Chunk overlap` controls for
each indexing operation. Chunk size is limited to 100-10,000 characters and
overlap to 0-5,000 characters. Indexing is blocked when overlap is greater than
or equal to chunk size.

## 12. Local Setup

From the project root:

```bash
uv sync
cp .env.example .env
```

Add a valid API key to `.env`, then run:

```bash
PYTHONPATH=. uv run streamlit run app/main.py
```

The application is normally available at:

```text
http://localhost:8501
```

## 13. Testing and Validation

Run the unit tests with the project environment:

```bash
uv run python -m unittest discover -s tests -v
```

Compile all Python files:

```bash
python3 -m py_compile app/*.py tests/test_rag.py
```

Check Git whitespace errors:

```bash
git diff --check
```

The tests cover:

- Text extraction and chunk metadata
- SHA-256 document hash creation
- Reasoning-tag removal
- Prompt construction

Live embedding and provider API tests require network access and valid API credentials.

## 14. Security Considerations

- Never commit `.env`.
- Never print API keys in terminal output or logs.
- Rotate any key that has been exposed.
- Keep `chroma_db/` private because it contains document content and embeddings.
- Add authentication and per-user data isolation before deploying publicly.
- Add request rate limits and file-size limits for public deployments.
- Avoid displaying sensitive document text in logs or error messages.

## 15. Known Limitations

- Local Chroma storage is single-user and is not suitable for shared public deployment without isolation.
- File count is limited to five and each uploaded file is limited to 50 MB.
- Scanned PDFs and images require OCR.
- Existing Chroma records created before document-hash metadata may need cleanup.
- The application depends on provider model IDs remaining active.
- Semantic chunking currently depends on `langchain-experimental`, which emits a
    sunset warning and should be monitored for a maintained replacement.
- API failures could be handled with more specific provider exception messages.
- Streamlit displays safe user-facing alerts for indexing, removal, and question
    failures without exposing raw provider or file error details.
- Document updates currently require deleting and re-indexing the document.

## 16. Recommended Future Improvements

1. Add file-size validation.
2. Add OCR support for scanned PDFs and images.
3. Add per-user collections or a server-side database for document ownership.
4. Add persistent document records instead of relying only on Chroma metadata.
5. Add retry and timeout handling for provider requests.
6. Add structured logging without secrets or document contents.
7. Add integration tests using mocked Chroma and LLM clients.
8. Cache the embedding function and Chroma collection for better Streamlit performance.
9. Add document update and bulk-delete operations.
10. Pin and regularly review dependency versions for production deployments.
