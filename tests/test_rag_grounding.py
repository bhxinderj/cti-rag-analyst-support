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


def test_query_records_used_chunk_provenance_for_exact_cve_match():
    chunks = [
        RetrievedChunk(
            doc_id="nvd_CVE-2024-3094",
            content="CVE-2024-3094 is the XZ Utils backdoor with SSH authentication impact.",
            score=0.91,
            metadata={"source": "nvd", "title": "XZ Utils vulnerability"},
            rank=0,
        ),
        RetrievedChunk(
            doc_id="cisa_kev_CVE-2021-44228",
            content="CVE-2021-44228 affects Apache Log4j2.",
            score=0.12,
            metadata={"source": "cisa_kev", "title": "Log4Shell"},
            rank=1,
        ),
    ]
    chain = _build_chain(
        chunks,
        llm_content=(
            "Summary: CVE-2024-3094 is the XZ Utils backdoor. "
            "[Source: XZ Utils vulnerability | nvd_CVE-2024-3094]"
        ),
    )

    response = chain.query("What is CVE-2024-3094?")

    assert response.abstention_reason is None
    assert response.prompt_context.startswith("[1] Source ID: nvd_CVE-2024-3094")
    assert response.source_documents[0]["doc_id"] == "nvd_CVE-2024-3094"
    assert response.source_documents[0]["rank"] == 1
    assert response.source_documents[0]["score"] == 0.91
    assert response.source_documents[0]["cited_in_answer"] is True
    assert response.source_documents[1]["rank"] == 2
    assert response.source_documents[1]["cited_in_answer"] is False


def test_query_accepts_natural_language_cve_question_when_context_is_supportive():
    chunks = [
        RetrievedChunk(
            doc_id="nvd_CVE-2024-3094",
            content=(
                "The XZ Utils backdoor was a March 2024 supply chain compromise "
                "that affected SSH authentication."
            ),
            score=0.88,
            metadata={"source": "nvd", "title": "XZ Utils vulnerability"},
        )
    ]
    chain = _build_chain(
        chunks,
        llm_content=(
            "Summary: The XZ Utils backdoor is tracked as CVE-2024-3094. "
            "[Source: XZ Utils vulnerability | nvd_CVE-2024-3094]"
        ),
    )

    response = chain.query("What happened with the XZ Utils backdoor in March 2024?")

    assert response.abstention_reason is None
    assert "CVE-2024-3094" in response.answer
    assert chain.llm.invocations == 1


def test_query_abstains_when_retrieval_is_only_weakly_related():
    chunks = [
        RetrievedChunk(
            doc_id="generic_guidance_1",
            content="Patch management guidance for enterprise software inventories.",
            score=0.2,
            metadata={"source": "note", "title": "Generic guidance"},
        )
    ]
    chain = _build_chain(chunks)

    response = chain.query(
        "How are public-facing application vulnerabilities used for initial access?"
    )

    assert response.abstention_reason == "The retrieved context is only weakly related to the question."
    assert response.answer.startswith("Summary: Insufficient evidence in the retrieved context")
    assert chain.llm.invocations == 0
