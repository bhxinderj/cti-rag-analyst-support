from src.cti_rag.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk


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
