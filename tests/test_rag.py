import tempfile
import unittest
from pathlib import Path

from app.loaders import load_documents
from app.rag_chain import build_prompt, clean_answer
from app.vector_store import search_documents


class RagLogicTests(unittest.TestCase):
    def test_load_documents_returns_chunks_metadata_and_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_text("A short document about electric vehicles.", encoding="utf-8")

            chunks, metadatas, ids = load_documents([path])

        self.assertEqual(len(chunks), len(metadatas))
        self.assertEqual(len(chunks), len(ids))
        self.assertEqual(metadatas[0]["source"], "example.txt")
        self.assertEqual(len(metadatas[0]["document_hash"]), 64)

    def test_clean_answer_removes_reasoning_tags(self) -> None:
        answer = "<think>private reasoning</think>\n## Answer\nGoogle was founded in 1998."

        self.assertEqual(clean_answer(answer), "## Answer\nGoogle was founded in 1998.")

    def test_build_prompt_contains_question_and_source(self) -> None:
        prompt = build_prompt(
            "Where is Google headquartered?",
            [{"text": "Google is headquartered in Mountain View.", "metadata": {"source": "Google.txt"}}],
        )

        self.assertIn("Where is Google headquartered?", prompt)
        self.assertIn("Google.txt", prompt)
        self.assertIn("Mountain View", prompt)

    def test_search_documents_supports_source_filter_and_result_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "alpha.txt"
            second_path = Path(directory) / "beta.txt"
            first_path.write_text("Alpha team works on search configs.", encoding="utf-8")
            second_path.write_text("Beta team works on indexing pipelines.", encoding="utf-8")

            from app.vector_store import add_documents

            alpha_chunks, alpha_metadatas, alpha_ids = load_documents([first_path])
            beta_chunks, beta_metadatas, beta_ids = load_documents([second_path])
            add_documents(alpha_chunks + beta_chunks, alpha_metadatas + beta_metadatas, alpha_ids + beta_ids)

            filtered_results = search_documents(
                "search configs",
                number_of_results=1,
                where={"source": "alpha.txt"},
            )

            self.assertEqual(len(filtered_results), 1)
            self.assertEqual(filtered_results[0]["metadata"]["source"], "alpha.txt")
            self.assertIn("search configs", filtered_results[0]["text"].lower())

    def test_search_documents_supports_multi_query_and_hyde_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query_demo.txt"
            path.write_text(
                "This document explains retrieval strategies, search modes, reranking, and HyDE design.",
                encoding="utf-8",
            )

            from app.vector_store import add_documents

            chunks, metadatas, ids = load_documents([path])
            add_documents(chunks, metadatas, ids)

            multi_query_results = search_documents(
                "retrieval strategies and search modes",
                number_of_results=2,
                search_mode="multi_query",
            )
            hyde_results = search_documents(
                "HyDE search methods",
                number_of_results=2,
                search_mode="hyde",
            )

            self.assertTrue(len(multi_query_results) >= 1)
            self.assertTrue(len(hyde_results) >= 1)
            self.assertIn("retrieval", multi_query_results[0]["text"].lower())


if __name__ == "__main__":
    unittest.main()
