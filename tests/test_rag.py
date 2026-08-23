import tempfile
import unittest
from pathlib import Path

from app.loaders import load_documents
from app.rag_chain import build_prompt, clean_answer


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


if __name__ == "__main__":
    unittest.main()
