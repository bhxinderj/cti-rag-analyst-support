"""
Conversation wrapper: condense-then-retrieve over the frozen pipeline
(Option B2, v1).

The retriever needs a self-contained query — follow-up questions like
"and how do I mitigate it?" carry no entities and would retrieve
noise. This wrapper resolves a follow-up into a standalone question
BEFORE the unchanged pipeline runs:

  1. **Passthrough** when the question already names CTI entities
     (router extraction) or no history exists.
  2. **Condense**: a single short LLM call rewrites the follow-up into
     a standalone CTI question, given the previous question and the
     entities of the previous routing decision. Runs on the same model
     as generation (local mode condenses locally).
  3. **Entity carry-over**: if condensing fails or the rewrite still
     names no entity, the previous routing decision's primary entities
     are appended as explicit context — the documented ship-criterion
     fallback that needs no model at all.

v1 deliberately keeps conversation history OUT of the generation
prompt (local and hosted): generation sees only the resolved
standalone question, so the frozen pipeline behaves exactly as
evaluated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from .router import classify_query

logger = logging.getLogger(__name__)

_MAX_RESOLVED_LENGTH = 500

_CONDENSE_SYSTEM_PROMPT = (
    "You rewrite follow-up questions from a CTI analyst conversation "
    "into fully self-contained questions for a retrieval system.\n"
    "Rules:\n"
    "1. Resolve pronouns and references ('it', 'this vulnerability', "
    "'the second one') using the conversation context.\n"
    "2. Keep CTI identifiers (CVE, ATT&CK technique, actor names) exactly "
    "as they appear in the context.\n"
    "3. Preserve the analyst's actual information need — do not broaden "
    "or answer the question.\n"
    "4. Output ONLY the rewritten question, one line, no quotes, no "
    "explanation."
)


@dataclass
class ConversationState:
    """Minimal rolling state the wrapper needs between turns."""

    last_question: str | None = None
    last_entities: list[str] = field(default_factory=list)

    def update(self, question: str, routing_decision: dict | None) -> None:
        self.last_question = question
        entities = (routing_decision or {}).get("primary_entities") or []
        if entities:
            self.last_entities = list(entities)


@dataclass
class ResolvedQuery:
    """Outcome of the resolution step, with a UI-facing trace."""

    question: str
    method: str  # "passthrough" | "condensed" | "carry_over"
    original_question: str = ""


def _carry_over(question: str, entities: list[str]) -> str:
    context = ", ".join(entities[:3])
    return f"{question} (in the context of {context})"


def resolve_question(
    question: str,
    state: ConversationState,
    condense_llm=None,
) -> ResolvedQuery:
    """Resolve a possibly-elliptical follow-up into a standalone question."""
    question = question.strip()

    # No history: nothing to resolve.
    if not state.last_question:
        return ResolvedQuery(question=question, method="passthrough")

    # Standalone already: the router finds explicit CTI entities.
    if classify_query(question).primary_entities:
        return ResolvedQuery(question=question, method="passthrough")

    condensed: str | None = None
    if condense_llm is not None:
        context_lines = [f"Previous question: {state.last_question}"]
        if state.last_entities:
            context_lines.append(
                "Entities under discussion: " + ", ".join(state.last_entities)
            )
        try:
            response = condense_llm.invoke(
                [
                    SystemMessage(content=_CONDENSE_SYSTEM_PROMPT),
                    HumanMessage(
                        content="\n".join(context_lines)
                        + f"\n\nFollow-up question: {question}\n\nRewritten question:"
                    ),
                ]
            )
            candidate = " ".join(str(response.content).split()).strip().strip('"')
            if candidate and len(candidate) <= _MAX_RESOLVED_LENGTH:
                condensed = candidate
        except Exception:
            logger.exception("Condense call failed — falling back to entity carry-over")

    if condensed:
        # If the rewrite still names no entity but we carry some, merge them —
        # a condense that dropped the subject would otherwise retrieve noise.
        if state.last_entities and not classify_query(condensed).primary_entities:
            condensed = _carry_over(condensed, state.last_entities)
        return ResolvedQuery(
            question=condensed, method="condensed", original_question=question
        )

    # Ship-criterion fallback: carry-over alone, no model involved.
    if state.last_entities:
        return ResolvedQuery(
            question=_carry_over(question, state.last_entities),
            method="carry_over",
            original_question=question,
        )

    return ResolvedQuery(question=question, method="passthrough")
