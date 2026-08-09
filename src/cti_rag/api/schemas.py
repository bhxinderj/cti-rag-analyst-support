"""
API request/response schemas for the CTI-RAG REST interface.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field(default="hybrid", pattern=r"^(hybrid|bm25|vector)$")
    templated: bool = Field(
        default=False,
        description="Use Phase-2 templated pipeline (router + fact bundle + L1/L2 templates).",
    )
    extended: bool = Field(
        default=False,
        description=(
            "Extended Analysis mode: hosted generation with analyst-assist "
            "instructions and deeper retrieval. Question and retrieved "
            "context leave the local machine. Overrides `templated`/`mode`."
        ),
    )
    # Conversation wrapper (v1): the client supplies minimal rolling state;
    # the API stays stateless. Generation never sees the history — only the
    # resolved standalone question reaches the frozen pipeline.
    last_question: str | None = Field(
        default=None,
        description="Previous user question, for condense-then-retrieve.",
    )
    last_entities: list[str] = Field(
        default_factory=list,
        description="Primary entities of the previous routing decision.",
    )


class SourceDocument(BaseModel):
    doc_id: str
    title: str
    source: str
    score: float
    rank: int
    citation_label: str
    cited_in_answer: bool


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceDocument]
    retrieval_mode: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    grounded: bool
    abstention_reason: str | None = None
    grounding_warnings: list[str] = []
    # Phase-2 templated-pipeline fields (None when the legacy path was used).
    pipeline: str = "legacy"
    generation_model: str | None = None
    # Conversation wrapper: set when the question was rewritten before retrieval.
    resolved_question: str | None = None
    resolution_method: str | None = None
    template: str | None = None
    routing_decision: dict | None = None
    l1_block: str | None = None
    l2_output: str | None = None


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    index_loaded: bool
    active_setup: str
