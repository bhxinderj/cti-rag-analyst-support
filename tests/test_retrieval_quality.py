import unittest

from src.cti_rag.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk


class TestRetrievalQualitySignals(unittest.TestCase):
    def test_rerank_applies_small_quality_signal_bonus(self):
        retriever = object.__new__(HybridRetriever)
        retriever.entity_match_boost = 0.0
        retriever.quality_signal_boost = 0.15
        retriever.rerank_top_k = 5

        class FakeReranker:
            @staticmethod
            def predict(pairs):
                return [0.5, 0.85]

        retriever.reranker = FakeReranker()

        chunks = [
            RetrievedChunk(
                doc_id="rich_doc",
                content="Title: Rich\nContent: Strong CTI context.",
                score=0.0,
                metadata={"cti_signal_score": 4.0},
                rank=0,
            ),
            RetrievedChunk(
                doc_id="weak_doc",
                content="Title: Weak\nContent: Weak context.",
                score=0.0,
                metadata={"cti_signal_score": 0.0},
                rank=1,
            ),
        ]

        reranked = HybridRetriever._rerank(retriever, "Explain this incident", chunks)

        self.assertEqual([chunk.doc_id for chunk in reranked], ["rich_doc", "weak_doc"])
        self.assertEqual(reranked[0].score, 1.1)
        self.assertEqual(reranked[1].score, 0.85)

    def test_build_bm25_metadata_prefers_persisted_signal_metadata(self):
        retriever = object.__new__(HybridRetriever)
        retriever.bm25_doc_ids = ["misp_event_1"]
        retriever.bm25_corpus = ["Title: Example\nContent: Example content"]
        retriever.bm25_metadatas = [
            {
                "source": "misp",
                "title": "Example",
                "cti_signal_score": 3.5,
                "has_cve": True,
            }
        ]

        metadata = HybridRetriever._build_bm25_metadata(retriever, 0)

        self.assertEqual(metadata["source"], "misp")
        self.assertEqual(metadata["title"], "Example")
        self.assertEqual(metadata["cti_signal_score"], 3.5)
        self.assertTrue(metadata["has_cve"])


if __name__ == "__main__":
    unittest.main()
