import re

from app.llm_client import get_llm_client
from app.vector_store import search_documents


SYSTEM_PROMPT = """
You are a helpful document question-answering assistant.
Answer using only the provided context.
If the context does not contain the answer, say you do not know based on the
uploaded documents. Do not invent facts.
Return only the final answer. Do not include thinking, analysis, or reasoning
tags in your response.
Keep the answer concise. Start with a short Markdown title, followed by one or
two short paragraphs. Avoid bullet points unless a list is essential.
""".strip()


def clean_answer(answer: str) -> str:
	"""Remove reasoning blocks that some models include in their response."""
	cleaned_answer = re.sub(
		r"<(?:think|thinking)>.*?</(?:think|thinking)>",
		"",
		answer,
		flags=re.IGNORECASE | re.DOTALL,
	)
	cleaned_answer = re.sub(
		r"<analysis>.*?</analysis>",
		"",
		cleaned_answer,
		flags=re.IGNORECASE | re.DOTALL,
	)
	return cleaned_answer.strip()


def retrieve_context(
	question: str,
	number_of_results: int = 4,
) -> list[dict]:
	"""Retrieve the document chunks most relevant to a question."""
	return search_documents(
		question=question,
		number_of_results=number_of_results,
	)


def build_prompt(question: str, documents: list[dict]) -> str:
	"""Create the user prompt sent to the language model."""
	if not documents:
		context = "No relevant document content was found."
	else:
		context = "\n\n".join(
			f"Source: {document['metadata'].get('source', 'Unknown')}\n"
			f"{document['text']}"
			for document in documents
		)

	return f"""Context:
{context}

Question:
{question}

Answer using only the context above."""


def ask_question(
	question: str,
	number_of_results: int = 4,
	provider: str | None = None,
	model_name: str | None = None,
	documents: list[dict] | None = None,
) -> str:
	"""Retrieve relevant chunks and return an answer from Grok."""
	if not question.strip():
		raise ValueError("Question cannot be empty")

	retrieved_documents = documents or retrieve_context(question, number_of_results)
	prompt = build_prompt(question, retrieved_documents)

	client, selected_model_name = get_llm_client(provider, model_name)
	response = client.chat.completions.create(
		model=selected_model_name,
		temperature=0,
		messages=[
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": prompt},
		],
	)

	answer = response.choices[0].message.content
	if not answer:
		raise RuntimeError("The language model returned an empty answer")

	cleaned_answer = clean_answer(answer)
	if not cleaned_answer:
		raise RuntimeError("The language model returned no final answer")
	return cleaned_answer
