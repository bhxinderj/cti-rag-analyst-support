from types import SimpleNamespace

from src.cti_rag.rag.chain import RAGChain, RetrievedChunk, _assess_context_support


class DummyRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, question: str):
        return list(self._chunks)


class DummyLLM:
    def __init__(self, content: str):
        self.content = content
        self.invocations = 0

    def invoke(self, messages):
        self.invocations += 1
        return SimpleNamespace(content=self.content)


def _build_chain(chunks, llm_content="Summary: test\nEvidence:\n- test [Source: test]"):
    chain = object.__new__(RAGChain)
    chain.retriever = DummyRetriever(chunks)
    chain.llm = DummyLLM(llm_content)
    chain.retrieval_mode = "hybrid"
    return chain


def test_assess_context_support_requires_exact_entity_match():
    chunk_dicts = [
        {
            "doc_id": "nvd_CVE-2024-3094",
            "title": "XZ Utils vulnerability",
            "content": "CVE-2024-3094 is a supply chain backdoor in XZ Utils.",
        }
    ]

    reason = _assess_context_support("What is CVE-1999-0001?", chunk_dicts)

    assert reason == "The retrieved context does not mention cve-1999-0001."


def test_query_abstains_before_generation_when_context_is_not_supported():
    chunks = [
        RetrievedChunk(
            doc_id="nvd_CVE-2024-3094",
            content="CVE-2024-3094 is a supply chain backdoor in XZ Utils.",
            score=0.9,
            metadata={"source": "nvd", "title": "XZ Utils vulnerability"},
        )
    ]
    chain = _build_chain(chunks)

    response = chain.query("What is CVE-1999-0001?")

    assert response.answer.startswith("Summary: Insufficient evidence in the retrieved context")
    assert "cve-1999-0001" in response.answer.lower()
    assert chain.llm.invocations == 0


def test_query_normalizes_citation_aliases_to_canonical_labels():
    chunks = [
        RetrievedChunk(
            doc_id="nvd_CVE-2024-3094",
            content="CVE-2024-3094 is a critical XZ Utils vulnerability with active exploitation concerns.",
            score=0.9,
            metadata={"source": "nvd", "title": "XZ Utils vulnerability"},
        )
    ]
    chain = _build_chain(
        chunks,
        llm_content=(
            "Summary: CVE-2024-3094 affects XZ Utils. [Source: nvd_CVE-2024-3094]\n"
            "Evidence:\n"
            "- Active exploitation concerns are described. [Sources: XZ Utils vulnerability]"
        ),
    )

    response = chain.query("What is CVE-2024-3094?")

    assert "[Source: XZ Utils vulnerability | nvd_CVE-2024-3094]" in response.answer
    assert "[Source: nvd_CVE-2024-3094]" not in response.answer
    assert "[Sources:" not in response.answer
    assert chain.llm.invocations == 1
