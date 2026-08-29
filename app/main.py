import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from app.config import (
	CHUNK_OVERLAP,
	CHUNK_SIZE,
	GROQ_MODEL_NAME,
	LLM_PROVIDER,
	MAX_DOCUMENTS,
	MAX_UPLOAD_SIZE_MB,
	XAI_MODEL_NAME,
)
from app.loaders import CHUNKING_STRATEGIES, load_documents
from app.rag_chain import ask_question, retrieve_context
from app.vector_store import (
	add_documents,
	delete_documents_by_source,
	document_count,
	list_indexed_document_hashes,
	list_indexed_sources,
)


SUPPORTED_FILE_TYPES = [
	"csv",
	"docx",
	"html",
	"json",
	"md",
	"pdf",
	"py",
	"txt",
	"xlsx",
	"xml",
]

MODEL_OPTIONS = {
	"xai": [XAI_MODEL_NAME],
	"groq": [
		GROQ_MODEL_NAME,
		"openai/gpt-oss-120b",
		"qwen/qwen3.6-27b",
	],
}


def show_user_error(operation: str, error: Exception) -> None:
	"""Show a useful alert without exposing provider or file internals."""
	status_code = getattr(error, "status_code", None)
	error_name = type(error).__name__

	if isinstance(error, ValueError):
		message = str(error)
	elif status_code == 401:
		message = "The API key is invalid or missing. Check your .env file."
	elif status_code == 429:
		message = "The provider rate limit was reached. Please try again later."
	elif isinstance(error, (ConnectionError, TimeoutError)):
		message = "The provider could not be reached. Check your internet connection."
	else:
		message = f"{operation} failed. Please try again."

	st.error(message)
	with st.expander("Technical details"):
		st.caption(f"Error type: {error_name}")


def save_uploaded_files(uploaded_files) -> list[Path]:
	"""Save Streamlit uploads temporarily so the loader can read them."""
	temporary_paths: list[Path] = []

	for uploaded_file in uploaded_files:
		suffix = Path(uploaded_file.name).suffix.lower()
		temporary_file = tempfile.NamedTemporaryFile(
			delete=False,
			suffix=suffix,
		)
		temporary_file.write(uploaded_file.getvalue())
		temporary_file.close()
		temporary_paths.append(Path(temporary_file.name))

	return temporary_paths


def index_uploaded_files(
	uploaded_files,
	chunk_size: int = CHUNK_SIZE,
	chunk_overlap: int = CHUNK_OVERLAP,
	chunking_strategy: str = "recursive_character",
) -> int:
	"""Extract, chunk, and store uploaded files in Chroma."""
	temporary_paths = save_uploaded_files(uploaded_files)
	try:
		chunks, metadatas, ids = load_documents(
			temporary_paths,
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
			chunking_strategy=chunking_strategy,
		)
		# Restore the original upload names in Chroma metadata.
		temporary_name_to_original_name = {
			temporary_path.name: uploaded_file.name
			for temporary_path, uploaded_file in zip(
				temporary_paths,
				uploaded_files,
				strict=True,
			)
		}
		for metadata in metadatas:
			metadata["source"] = temporary_name_to_original_name.get(
				metadata["source"],
				metadata["source"],
			)

		add_documents(chunks=chunks, metadatas=metadatas, ids=ids)
		return len(chunks)
	finally:
		for temporary_path in temporary_paths:
			temporary_path.unlink(missing_ok=True)


def find_duplicate_uploads(uploaded_files) -> list[str]:
	"""Find uploads duplicated by content or filename."""
	indexed_hashes = list_indexed_document_hashes()
	indexed_sources = set(list_indexed_sources())
	seen_hashes: set[str] = set()
	seen_sources: set[str] = set()
	duplicates: list[str] = []

	for uploaded_file in uploaded_files:
		document_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
		if (
			document_hash in indexed_hashes
			or document_hash in seen_hashes
			or uploaded_file.name in indexed_sources
			or uploaded_file.name in seen_sources
		):
			duplicates.append(uploaded_file.name)
		seen_hashes.add(document_hash)
		seen_sources.add(uploaded_file.name)

	return duplicates


def render_upload_section(
	chunk_size: int,
	chunk_overlap: int,
	chunking_strategy: str,
) -> None:
	st.header("1. Add documents")
	uploaded_files = st.file_uploader(
		"Choose up to five documents",
		type=SUPPORTED_FILE_TYPES,
		accept_multiple_files=True,
	)

	if len(uploaded_files) > MAX_DOCUMENTS:
		st.error(f"Please select no more than {MAX_DOCUMENTS} files.")
		return
	oversized_files = [
		uploaded_file.name
		for uploaded_file in uploaded_files
		if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024
	]
	if oversized_files:
		st.error(
			f"Each file must be {MAX_UPLOAD_SIZE_MB} MB or smaller: "
			+ ", ".join(oversized_files)
		)
		return
	if chunk_overlap >= chunk_size:
		st.error("Chunk overlap must be smaller than chunk size.")
		return

	if st.button("Index documents", disabled=not uploaded_files):
		try:
			duplicate_uploads = find_duplicate_uploads(uploaded_files)
			if duplicate_uploads:
				st.warning(
					"Already indexed or duplicated in this upload: "
					+ ", ".join(duplicate_uploads)
				)
				return

			with st.spinner("Reading and indexing documents..."):
				chunk_count = index_uploaded_files(
					uploaded_files,
					chunk_size=chunk_size,
					chunk_overlap=chunk_overlap,
					chunking_strategy=chunking_strategy,
				)
			st.success(f"Added {chunk_count} chunks to Chroma.")
		except Exception as error:
			show_user_error("Document indexing", error)


def render_indexed_documents() -> None:
	"""Show persisted documents and let the user remove them explicitly."""
	with st.expander("Indexed documents", expanded=False):
		sources = list_indexed_sources()
		if not sources:
			st.caption("No documents have been indexed yet.")
			return

		st.caption(f"{len(sources)} document(s) stored in Chroma")
		for source in sources:
			column_name, column_action = st.columns([4, 1])
			column_name.write(source)
			if column_action.button("Remove", key=f"remove-{source}"):
				try:
					delete_documents_by_source(source)
				except Exception as error:
					show_user_error("Document removal", error)
				else:
					st.success(f"Removed {source}.")
					st.rerun()


def render_model_section() -> tuple[str, str]:
	st.sidebar.header("Model")
	provider_options = list(MODEL_OPTIONS)
	default_provider_index = (
		provider_options.index(LLM_PROVIDER)
		if LLM_PROVIDER in provider_options
		else 0
	)
	provider = st.sidebar.selectbox(
		"Provider",
		provider_options,
		index=default_provider_index,
	)
	model_name = st.sidebar.selectbox(
		"Model",
		MODEL_OPTIONS[provider],
	)
	return provider, model_name


def render_chunking_section() -> tuple[int, int, str]:
	st.sidebar.header("Chunking")
	strategy_label = st.sidebar.selectbox(
		"Chunking strategy",
		list(CHUNKING_STRATEGIES),
		help="Choose how document text is divided before indexing.",
	)
	chunk_size = st.sidebar.number_input(
		"Chunk size",
		min_value=100,
		max_value=10000,
		value=CHUNK_SIZE,
		step=100,
		help="Approximate number of characters in each chunk.",
	)
	chunk_overlap = st.sidebar.number_input(
		"Chunk overlap",
		min_value=0,
		max_value=5000,
		value=CHUNK_OVERLAP,
		step=50,
		help="Characters shared between neighboring chunks.",
	)
	if chunk_overlap >= chunk_size:
		st.sidebar.error("Chunk overlap must be smaller than chunk size.")
	if strategy_label == "Semantic":
		st.sidebar.info("Semantic chunking uses meaning instead of chunk size.")
	return (
		int(chunk_size),
		int(chunk_overlap),
		CHUNKING_STRATEGIES[strategy_label],
	)


def render_question_section(provider: str, model_name: str) -> None:
	st.header("2. Ask a question")
	st.caption(f"Stored chunks: {document_count()}")
	question = st.text_input("Question", placeholder="What do the documents say?")

	if st.button("Ask", disabled=not question.strip()):
		if document_count() == 0:
			st.warning("Index at least one document before asking a question.")
			return

		try:
			with st.spinner("Searching documents and generating an answer..."):
				relevant_documents = retrieve_context(question)
				answer = ask_question(
					question,
					provider=provider,
					model_name=model_name,
					documents=relevant_documents,
				)
			st.subheader("Answer")
			st.caption(f"Generated with {provider} / {model_name}")
			with st.container(border=True):
				st.markdown(answer)

			with st.expander("Sources"):
				sources = dict.fromkeys(
					document["metadata"].get("source", "Unknown")
					for document in relevant_documents
				)
				for source in sources:
					st.write(source)
		except Exception as error:
			show_user_error("Question answering", error)


def main() -> None:
	st.set_page_config(page_title="Document RAG", page_icon="📚")
	st.title("Document Question Answering")
	st.write("Upload documents, index them, and ask questions about their content.")
	provider, model_name = render_model_section()
	chunk_size, chunk_overlap, chunking_strategy = render_chunking_section()

	render_upload_section(chunk_size, chunk_overlap, chunking_strategy)
	render_indexed_documents()
	st.divider()
	render_question_section(provider, model_name)


if __name__ == "__main__":
	main()
