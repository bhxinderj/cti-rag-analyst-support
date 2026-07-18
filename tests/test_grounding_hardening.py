"""
Unit tests for Step 12 fuzzy-alias citation resolution.

Covers the alias expansion and fuzzy-fallback helpers in
``rag/chain.py``:

- :func:`_alias_variants` — pulls out CTI identifiers and bare doc-id
  suffixes so "CVE-2024-3094" resolves back to a canonical label
  "XZ Utils backdoor | nvd_CVE-2024-3094".
- :func:`_fuzzy_resolve_alias` — deterministic fallback when the strict
  alias map misses: substring match on canonical label / doc_id, then
  ≥2-token overlap with the chunk title.
- :func:`_normalize_response_citations` — end-to-end behaviour of the
  two stages combined: bare CVE / title / doc_id citations are
  preserved and normalized to the canonical form; truly unmatched
  citations are still stripped.

An earlier Step-12 revision also shipped a ``_enforce_l2_citations``
line-level replacer; it was reverted after the live smoke showed the
local LLM rarely emits ``[Source: ...]`` at all, so line-level
enforcement replaced every claim with a grounding note and crashed the
Rubric Mean-Total. See ``rag/chain.py`` for the rollback rationale.

Tests follow the repo convention: plain ``def test_*`` functions with
bare ``assert`` statements, discovered by ``tests/run_smoke_checks.py``.
"""

from __future__ import annotations

from src.cti_rag.rag.chain import (
    _alias_variants,
    _build_citation_alias_map,
    _fuzzy_resolve_alias,
    _normalize_response_citations,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _docs() -> list[dict]:
    """Two-chunk fixture mirroring the retriever output shape."""
    return [
        {
            "doc_id": "nvd_CVE-2024-3094",
            "title": "XZ Utils backdoor",
            "citation_label": "XZ Utils backdoor | nvd_CVE-2024-3094",
        },
        {
            "doc_id": "cisa_kev_CVE-2021-44228",
            "title": "Log4Shell in Apache Log4j",
            "citation_label": "Log4Shell in Apache Log4j | cisa_kev_CVE-2021-44228",
        },
    ]


# ---------------------------------------------------------------------------
# _alias_variants + _build_citation_alias_map
# ---------------------------------------------------------------------------


def test_alias_variants_extracts_cve_and_bare_docid():
    variants = _alias_variants("XZ Utils backdoor | nvd_CVE-2024-3094")
    lowered = [v.lower() for v in variants]
    # Full label kept
    assert any("nvd_cve-2024-3094" in v for v in lowered)
    # Bare doc_id suffix after underscore extracted
    assert any(v == "cve-2024-3094" for v in lowered)


def test_alias_variants_handles_attack_technique_id():
    variants = _alias_variants("mitre_T1059.003")
    assert any(v.lower() == "t1059.003" for v in variants)


def test_build_alias_map_includes_bare_cve_when_unambiguous():
    alias_map, _doc_id_map, _position_map = _build_citation_alias_map(_docs())
    assert alias_map.get("cve-2024-3094") == "XZ Utils backdoor | nvd_CVE-2024-3094"
    assert alias_map.get("cve-2021-44228") == "Log4Shell in Apache Log4j | cisa_kev_CVE-2021-44228"


def test_build_alias_map_returns_doc_id_and_position_maps():
    _alias_map, doc_id_map, position_map = _build_citation_alias_map(_docs())
    assert doc_id_map.get("nvd_cve-2024-3094") == "XZ Utils backdoor | nvd_CVE-2024-3094"
    assert position_map.get(1) == "XZ Utils backdoor | nvd_CVE-2024-3094"
    assert position_map.get(2) == "Log4Shell in Apache Log4j | cisa_kev_CVE-2021-44228"


def test_build_alias_map_drops_ambiguous_bare_id_when_two_chunks_share_cve():
    # Two chunks, same CVE — bare CVE can't be uniquely resolved.
    dupes = [
        {
            "doc_id": "nvd_CVE-2024-3094",
            "title": "XZ backdoor NVD entry",
            "citation_label": "XZ backdoor NVD entry | nvd_CVE-2024-3094",
        },
        {
            "doc_id": "cisa_ad_CVE-2024-3094",
            "title": "XZ backdoor CISA advisory",
            "citation_label": "XZ backdoor CISA advisory | cisa_ad_CVE-2024-3094",
        },
    ]
    alias_map, _doc_id_map, _position_map = _build_citation_alias_map(dupes)
    assert "cve-2024-3094" not in alias_map  # ambiguous → removed


# ---------------------------------------------------------------------------
# _fuzzy_resolve_alias
# ---------------------------------------------------------------------------


def test_fuzzy_resolver_matches_title_tokens():
    # Needle has ≥2 substantive tokens shared with the Log4Shell title.
    resolved = _fuzzy_resolve_alias("Apache Log4j vulnerability report", _docs())
    assert resolved == "Log4Shell in Apache Log4j | cisa_kev_CVE-2021-44228"


def test_fuzzy_resolver_returns_none_for_unrelated_text():
    assert _fuzzy_resolve_alias("nothing relevant here", _docs()) is None


def test_fuzzy_resolver_returns_none_on_ambiguous_substring():
    dupes = [
        {
            "doc_id": "src_a",
            "title": "Apache HTTP Server advisory",
            "citation_label": "Apache HTTP Server advisory | src_a",
        },
        {
            "doc_id": "src_b",
            "title": "Apache Tomcat advisory",
            "citation_label": "Apache Tomcat advisory | src_b",
        },
    ]
    # Only one substantive token ("advisory") shared with either title — the
    # fuzzy resolver requires ≥2 tokens, so this must not resolve.
    assert _fuzzy_resolve_alias("advisory something", dupes) is None


# ---------------------------------------------------------------------------
# End-to-end: _normalize_response_citations
# ---------------------------------------------------------------------------


def test_normalize_response_recovers_bare_cve_citations():
    text = "Exploited actively. [Source: CVE-2024-3094]"
    out = _normalize_response_citations(text, _docs())
    assert "[Source: XZ Utils backdoor | nvd_CVE-2024-3094]" in out


def test_normalize_response_recovers_title_citations():
    text = "This bug is catastrophic. [Source: Log4Shell in Apache Log4j]"
    out = _normalize_response_citations(text, _docs())
    assert "[Source: Log4Shell in Apache Log4j | cisa_kev_CVE-2021-44228]" in out


def test_normalize_response_still_strips_truly_invalid_citations():
    text = "Random claim. [Source: some_totally_made_up_label]"
    out = _normalize_response_citations(text, _docs())
    assert "Source" not in out
    assert out.strip() == "Random claim."


def test_normalize_response_multiple_labels_in_one_block():
    text = "Both apply. [Source: CVE-2024-3094; Log4Shell in Apache Log4j]"
    out = _normalize_response_citations(text, _docs())
    assert "[Source: XZ Utils backdoor | nvd_CVE-2024-3094]" in out
    assert "[Source: Log4Shell in Apache Log4j | cisa_kev_CVE-2021-44228]" in out


def test_normalize_response_accepts_short_form_docid_bracket():
    text = "Backdoor confirmed [nvd_CVE-2024-3094]."
    out = _normalize_response_citations(text, _docs())
    assert "[Source: XZ Utils backdoor | nvd_CVE-2024-3094]" in out


def test_normalize_response_resolves_numeric_position_citation():
    # [2] refers to the second chunk in prompt-context order.
    text = "Critical impact [Source: 2]."
    out = _normalize_response_citations(text, _docs())
    assert "[Source: Log4Shell in Apache Log4j | cisa_kev_CVE-2021-44228]" in out


def test_normalize_response_keeps_unknown_short_brackets_untouched():
    text = "Uses scripting [T1059] against servers."
    out = _normalize_response_citations(text, _docs())
    assert "[T1059]" in out
