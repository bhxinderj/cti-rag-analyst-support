"""
Unit tests for the condense-then-retrieve conversation wrapper
(src/cti_rag/rag/conversation.py, Option B2 v1).
"""

from __future__ import annotations

from types import SimpleNamespace

from src.cti_rag.rag.conversation import (
    ConversationState,
    resolve_question,
)


class _FakeCondenseLLM:
    def __init__(self, rewrite: str | None, raise_error: bool = False):
        self._rewrite = rewrite
        self._raise = raise_error
        self.invocations = 0
        self.last_messages = None

    def invoke(self, messages):
        self.invocations += 1
        self.last_messages = list(messages)
        if self._raise:
            raise RuntimeError("condense unavailable")
        return SimpleNamespace(content=self._rewrite)


def _state() -> ConversationState:
    state = ConversationState()
    state.update(
        "What is CVE-2023-4966 (Citrix Bleed) and how severe is it?",
        {"primary_entities": ["CVE-2023-4966"]},
    )
    return state


def test_passthrough_without_history():
    llm = _FakeCondenseLLM("should not be called")
    resolved = resolve_question("And how do I mitigate it?", ConversationState(), llm)
    assert resolved.method == "passthrough"
    assert llm.invocations == 0


def test_passthrough_when_question_names_entities():
    llm = _FakeCondenseLLM("should not be called")
    resolved = resolve_question(
        "Is CVE-2024-3094 in the KEV catalog?", _state(), llm
    )
    assert resolved.method == "passthrough"
    assert resolved.question == "Is CVE-2024-3094 in the KEV catalog?"
    assert llm.invocations == 0


def test_condense_rewrites_elliptical_follow_up():
    llm = _FakeCondenseLLM("What are the recommended mitigations for CVE-2023-4966?")
    resolved = resolve_question("And how do I mitigate it?", _state(), llm)

    assert resolved.method == "condensed"
    assert resolved.question == "What are the recommended mitigations for CVE-2023-4966?"
    assert resolved.original_question == "And how do I mitigate it?"
    # The condense prompt carried the previous question and entities.
    prompt_text = llm.last_messages[1].content
    assert "CVE-2023-4966" in prompt_text
    assert "Citrix Bleed" in prompt_text


def test_condense_dropping_entity_gets_carry_over_merged():
    llm = _FakeCondenseLLM("What are the recommended mitigations?")
    resolved = resolve_question("And how do I mitigate it?", _state(), llm)

    assert resolved.method == "condensed"
    assert "CVE-2023-4966" in resolved.question


def test_condense_failure_falls_back_to_carry_over():
    llm = _FakeCondenseLLM(None, raise_error=True)
    resolved = resolve_question("And how do I mitigate it?", _state(), llm)

    assert resolved.method == "carry_over"
    assert "CVE-2023-4966" in resolved.question
    assert resolved.question.startswith("And how do I mitigate it?")


def test_no_condense_llm_uses_carry_over_directly():
    resolved = resolve_question("What about detection?", _state(), condense_llm=None)
    assert resolved.method == "carry_over"
    assert "CVE-2023-4966" in resolved.question


def test_state_update_keeps_last_known_entities():
    state = _state()
    # A follow-up turn whose routing found no entities must not wipe the
    # carried context.
    state.update("And how do I mitigate it?", {"primary_entities": []})
    assert state.last_entities == ["CVE-2023-4966"]
    assert state.last_question == "And how do I mitigate it?"
