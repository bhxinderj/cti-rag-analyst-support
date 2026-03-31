from src.cti_rag.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from unittest.mock import patch


def test_hybrid_retriever_records_stage_trace():
    retriever = object.__new__(HybridRetriever)
    retriever.mode = "hybrid"
    retriever.bm25_top_k = 2
    retriever.vector_top_k = 2
    retriever.rerank_top_k = 1
    retriever.last_trace = {}

    retriever._search_bm25 = lambda query, top_k: [
        RetrievedChunk(
            doc_id="nvd_CVE-2024-3094",
            content="bm25 hit",
            score=12.0,
            metadata={"source": "nvd", "title": "XZ Utils vulnerability"},
            rank=0,
        )
    ]
    retriever._search_vector = lambda query, top_k: [
        RetrievedChunk(
            doc_id="misp_event_1",
            content="vector hit",
            score=0.77,
            metadata={"source": "misp", "title": "MISP event"},
            rank=0,
        )
    ]
    retriever._reciprocal_rank_fusion = lambda bm25_results, vector_results: [
        RetrievedChunk(
            doc_id="nvd_CVE-2024-3094",
            content="fused hit",
            score=0.51,
            metadata={"source": "nvd", "title": "XZ Utils vulnerability"},
            rank=0,
        ),
        RetrievedChunk(
            doc_id="misp_event_1",
            content="fused hit",
            score=0.49,
            metadata={"source": "misp", "title": "MISP event"},
            rank=1,
        ),
    ]
    retriever._rerank = lambda query, chunks: [
        RetrievedChunk(
            doc_id="nvd_CVE-2024-3094",
            content="final hit",
            score=101.5,
            metadata={"source": "nvd", "title": "XZ Utils vulnerability"},
            rank=0,
        )
    ]

    results = HybridRetriever.retrieve(retriever, "What is CVE-2024-3094?")

    assert [chunk.doc_id for chunk in results] == ["nvd_CVE-2024-3094"]
    assert retriever.last_trace["mode"] == "hybrid"
    assert retriever.last_trace["stages"]["bm25"][0]["doc_id"] == "nvd_CVE-2024-3094"
    assert retriever.last_trace["stages"]["vector"][0]["doc_id"] == "misp_event_1"
    assert retriever.last_trace["stages"]["fused"][0]["score"] == 0.51
    assert retriever.last_trace["stages"]["reranked"][0]["score"] == 101.5
    assert retriever.last_trace["stages"]["final"][0]["doc_id"] == "nvd_CVE-2024-3094"


def test_bm25_mode_uses_same_reranking_step_for_ablation_fairness():
    retriever = object.__new__(HybridRetriever)
    retriever.mode = "bm25"
    retriever.bm25_top_k = 20
    retriever.rerank_top_k = 5
    retriever.last_trace = {}

    calls = []

    def fake_bm25(query, top_k):
        calls.append(("bm25", query, top_k))
        return [
            RetrievedChunk(
                doc_id="nvd_CVE-2024-3094",
                content="bm25 hit",
                score=12.0,
                metadata={"source": "nvd", "title": "XZ Utils vulnerability"},
                rank=0,
            )
        ]

    def fake_rerank(query, chunks):
        calls.append(("rerank", query, [chunk.doc_id for chunk in chunks]))
        return chunks

    retriever._search_bm25 = fake_bm25
    retriever._rerank = fake_rerank

    results = HybridRetriever.retrieve(retriever, "What is CVE-2024-3094?")

    assert [chunk.doc_id for chunk in results] == ["nvd_CVE-2024-3094"]
    assert calls == [
        ("bm25", "What is CVE-2024-3094?", 20),
        ("rerank", "What is CVE-2024-3094?", ["nvd_CVE-2024-3094"]),
    ]
    assert retriever.last_trace["stages"]["bm25"][0]["doc_id"] == "nvd_CVE-2024-3094"
    assert retriever.last_trace["stages"]["reranked"][0]["doc_id"] == "nvd_CVE-2024-3094"


def test_vector_mode_uses_same_reranking_step_for_ablation_fairness():
    retriever = object.__new__(HybridRetriever)
    retriever.mode = "vector"
    retriever.vector_top_k = 20
    retriever.rerank_top_k = 5
    retriever.last_trace = {}

    calls = []

    def fake_vector(query, top_k):
        calls.append(("vector", query, top_k))
        return [
            RetrievedChunk(
                doc_id="misp_event_1",
                content="vector hit",
                score=0.77,
                metadata={"source": "misp", "title": "MISP event"},
                rank=0,
            )
        ]

    def fake_rerank(query, chunks):
        calls.append(("rerank", query, [chunk.doc_id for chunk in chunks]))
        return chunks

    retriever._search_vector = fake_vector
    retriever._rerank = fake_rerank

    results = HybridRetriever.retrieve(retriever, "Which malware uses T1059?")

    assert [chunk.doc_id for chunk in results] == ["misp_event_1"]
    assert calls == [
        ("vector", "Which malware uses T1059?", 20),
        ("rerank", "Which malware uses T1059?", ["misp_event_1"]),
    ]
    assert retriever.last_trace["stages"]["vector"][0]["doc_id"] == "misp_event_1"
    assert retriever.last_trace["stages"]["reranked"][0]["doc_id"] == "misp_event_1"


def test_bm25_search_does_not_double_score_query_entities():
    retriever = object.__new__(HybridRetriever)
    retriever.bm25_doc_ids = ["nvd_CVE-2021-44228"]
    retriever.bm25_corpus = ["Title: CVE-2021-44228\nContent: Example"]

    class FakeBM25:
        def __init__(self):
            self.calls = []

        def get_scores(self, tokens):
            self.calls.append(tokens)
            return [7.5]

    fake_bm25 = FakeBM25()
    retriever.bm25 = fake_bm25

    results = HybridRetriever._search_bm25(
        retriever,
        "What is CVE-2021-44228 and how has it been exploited?",
        top_k=1,
    )

    assert [chunk.score for chunk in results] == [7.5]
    assert fake_bm25.calls == [[
        "what",
        "is",
        "cve-2021-44228",
        "and",
        "how",
        "has",
        "it",
        "been",
        "exploited",
    ]]


def test_rerank_applies_small_configured_entity_bonus():
    retriever = object.__new__(HybridRetriever)
    retriever.entity_match_boost = 1.0
    retriever.rerank_top_k = 5

    class FakeReranker:
        @staticmethod
        def predict(pairs):
            assert len(pairs) == 2
            return [0.4, 0.9]

    retriever.reranker = FakeReranker()

    chunks = [
        RetrievedChunk(
            doc_id="nvd_CVE-2021-44228",
            content="Title: CVE-2021-44228\nContent: Exact identifier present.",
            score=0.0,
            metadata={},
            rank=0,
        ),
        RetrievedChunk(
            doc_id="generic_doc",
            content="Title: Generic\nContent: No identifier here.",
            score=0.0,
            metadata={},
            rank=1,
        ),
    ]

    reranked = HybridRetriever._rerank(
        retriever,
        "What is CVE-2021-44228 and how has it been exploited?",
        chunks,
    )

    assert [chunk.doc_id for chunk in reranked] == ["nvd_CVE-2021-44228", "generic_doc"]
    assert reranked[0].score == 1.4
    assert reranked[1].score == 0.9


def test_load_reranker_prefers_local_snapshot_and_offline_mode():
    retriever = object.__new__(HybridRetriever)

    with patch.object(HybridRetriever, "_resolve_local_hf_snapshot", return_value="/tmp/local-model"), patch(
        "src.cti_rag.retrieval.hybrid_retriever.CrossEncoder"
    ) as mock_cross_encoder:
        HybridRetriever._load_reranker(retriever, "cross-encoder/ms-marco-MiniLM-L-6-v2")

    mock_cross_encoder.assert_called_once_with("/tmp/local-model", local_files_only=True)


def test_load_reranker_fails_fast_when_local_snapshot_is_missing():
    retriever = object.__new__(HybridRetriever)

    with patch.object(HybridRetriever, "_resolve_local_hf_snapshot", return_value=None):
        try:
            HybridRetriever._load_reranker(retriever, "cross-encoder/ms-marco-MiniLM-L-6-v2")
        except FileNotFoundError as exc:
            assert "local Hugging Face cache" in str(exc)
        else:
            raise AssertionError("Expected FileNotFoundError when local reranker snapshot is missing")
