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
	XAI_MODEL_NAME,
)
from app.loaders import load_documents
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


def index_uploaded_files(uploaded_files) -> int:
	"""Extract, chunk, and store uploaded files in Chroma."""
	temporary_paths = save_uploaded_files(uploaded_files)
	try:
		chunks, metadatas, ids = load_documents(
			temporary_paths,
			chunk_size=CHUNK_SIZE,
			chunk_overlap=CHUNK_OVERLAP,
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


def render_upload_section() -> None:
	st.header("1. Add documents")
	uploaded_files = st.file_uploader(
		"Choose up to five documents",
		type=SUPPORTED_FILE_TYPES,
		accept_multiple_files=True,
	)

	if len(uploaded_files) > MAX_DOCUMENTS:
		st.error(f"Please select no more than {MAX_DOCUMENTS} files.")
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
				chunk_count = index_uploaded_files(uploaded_files)
			st.success(f"Added {chunk_count} chunks to Chroma.")
		except (OSError, ValueError) as error:
			st.error(str(error))


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
				delete_documents_by_source(source)
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
		except (RuntimeError, ValueError, ConnectionError) as error:
			st.error(str(error))


def main() -> None:
	st.set_page_config(page_title="Document RAG", page_icon="📚")
	st.title("Document Question Answering")
	st.write("Upload documents, index them, and ask questions about their content.")
	provider, model_name = render_model_section()

	render_upload_section()
	render_indexed_documents()
	st.divider()
	render_question_section(provider, model_name)


if __name__ == "__main__":
	main()
