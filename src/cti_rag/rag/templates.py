"""
Three analyst-triage templates for Phase 2.

Public contract
---------------
Each template exposes two functions:

- ``render_*_l1(fact_bundle)`` — produces the deterministic Layer-1
  markdown block from ``FactBundle`` data. No LLM involvement. This
  block is guaranteed to match the retrieved structured metadata
  verbatim and is the part of the final answer that the analyst can
  trust at a glance.

- ``build_*_prompt(fact_bundle, question, chunks)`` — produces a
  ``[system_msg, user_msg]`` pair (each a ``{"role", "content"}`` dict)
  that asks the LLM to generate only the Layer-2 / Layer-3 sections
  (exploitation context, mitigations, narrative). The pre-rendered L1
  block is included in the user prompt so the model sees the target
  structure and knows which sections it is and is not responsible for.

Separation of L1 and L2/L3
--------------------------
This split is the core design move behind Phase 2: Layer-1 fields
(CVSS, CWE, KEV dates, triage signal, comparison table) are assembled
deterministically from chunk metadata — the LLM never re-states them,
so they cannot be hallucinated. The LLM contributes only narrative
content that must be grounded in the retrieved chunks via inline
citations of the form ``[Source: <citation_label>]``.

See docs/triage-template-spec.md §§ 5, 6, 7.
"""

from __future__ import annotations

from typing import Iterable

from .facts import (
    CrossSourceCompareFacts,
    FactBundle,
    ThreatContextFacts,
    VulnTriageFacts,
)
from .severity import annotate_vuln_triage_bundle


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fmt_bool(value: object) -> str:
    return "yes" if bool(value) else "no"


def _fmt_list(values: Iterable[str], sep: str = ", ") -> str:
    items = [v for v in values if v]
    return sep.join(items) if items else "—"


def _format_chunks_for_prompt(chunks: list[dict]) -> str:
    """
    Format chunks the same way :func:`cti_rag.rag.prompts.format_context`
    does, but locally so this module has no circular dependency on
    ``prompts.py`` during testing.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        label = chunk.get("citation_label") or chunk.get("doc_id") or f"chunk_{i}"
        source = chunk.get("source") or (chunk.get("metadata") or {}).get("source") or "unknown"
        title = chunk.get("title") or ""
        content = chunk.get("content") or ""
        parts.append(
            f"[{i}] Citation Label: {label}\n"
            f"    Source Type: {source}\n"
            f"    Title: {title}\n"
            f"    Content: {content}"
        )
    return "\n\n".join(parts)


def _citation_labels(chunks: list[dict]) -> list[str]:
    labels: list[str] = []
    for chunk in chunks:
        label = chunk.get("citation_label")
        if label and label not in labels:
            labels.append(label)
    return labels


# Shared system prompt. The L2/L3 contract is the same across templates:
# use only the retrieved context, cite every factual claim, one claim per
# line, and do not re-emit fields already given in the L1 block.
_SYSTEM_PROMPT = """You are a Cyber Threat Intelligence (CTI) analyst assistant operating in Triage Card mode.

You will receive:
1. A pre-rendered Layer-1 block that already contains deterministic facts
   (CVE-ID, CVSS, KEV status, severity signal, comparison tables). Do
   NOT repeat, re-state, or modify this block.
2. A list of retrieved context chunks with explicit citation labels.
3. A list of target sections to produce.

Rules:
1. Produce ONLY the target sections listed in the user message, in order.
2. Use ONLY the retrieved context. Do not use prior knowledge.
3. Every factual statement must include an inline citation in the form
   [Source: <citation_label>]. Use only labels present in the retrieved
   context.
4. Keep to one grounded claim per bullet or line.
5. If a section has no support in the retrieved context, write exactly:
   "No grounded information in the retrieved context."
6. Be exact with CTI identifiers (CVE IDs, CVSS scores, ATT&CK techniques,
   malware names, actor names).
"""


# ---------------------------------------------------------------------------
# VulnTriage (§7.1)
# ---------------------------------------------------------------------------


def _vuln_triage_card(facts: VulnTriageFacts) -> list[str]:
    """Render a single VulnTriage card's L1 block as markdown lines."""
    # Ensure the triage signal is populated so every card has an ampel.
    if facts.triage_signal is None:
        annotate_vuln_triage_bundle(facts)
    signal = facts.triage_signal or {}

    header_title = facts.title or facts.cve_id
    lines: list[str] = [
        f"## {facts.cve_id} — {header_title}",
        "",
        f"**Triage Signal:** {signal.get('label', 'UNKNOWN — Manual review required')}",
        f"**Rationale:** {signal.get('rationale', '—')}",
        "",
        "**Vulnerability Facts** (deterministic)",
    ]

    if facts.cvss_score is not None and facts.cvss_vector:
        lines.append(f"- CVSS: {facts.cvss_score} ({facts.cvss_vector})")
    elif facts.cvss_score is not None:
        lines.append(f"- CVSS: {facts.cvss_score}")
    else:
        lines.append("- CVSS: not available in retrieved context")

    if facts.cwe_ids:
        lines.append(f"- CWE: {_fmt_list(facts.cwe_ids)}")
    if facts.severity:
        lines.append(f"- Severity (source-reported): {facts.severity.upper()}")

    lines.append("")
    lines.append("**Affected Products** (deterministic from CPE)")
    if facts.affected_products:
        for product in facts.affected_products:
            lines.append(f"- {product}")
    else:
        lines.append("- No CPE data in retrieved context")

    lines.append("")
    lines.append("**KEV Status** (deterministic)")
    lines.append(f"- Listed in CISA KEV: {_fmt_bool(facts.kev_listed)}")
    if facts.kev_listed:
        if facts.kev_date_added:
            lines.append(f"- Date added: {facts.kev_date_added}")
        if facts.kev_due_date:
            lines.append(f"- Remediation due: {facts.kev_due_date}")
        lines.append(f"- Known ransomware use: {_fmt_bool(facts.ransomware_use)}")
        if facts.kev_required_action:
            lines.append(f"- CISA required action: {facts.kev_required_action}")

    if facts.inconsistencies:
        lines.append("")
        lines.append("**Source Inconsistencies** (deterministic)")
        for issue in facts.inconsistencies:
            lines.append(f"- {issue.get('note', issue.get('field', ''))}")

    return lines


def render_vuln_triage_l1(bundle: FactBundle) -> str:
    """Render the full deterministic L1 block for one or more VulnTriage cards."""
    if bundle.template != "VulnTriage":
        raise ValueError(f"VulnTriage renderer invoked on template={bundle.template!r}")

    card_blocks: list[str] = []
    for facts in bundle.vuln_triage:
        lines = _vuln_triage_card(facts)
        card_blocks.append("\n".join(lines))

    if not card_blocks:
        return "## No CVE entity resolved from retrieved context\n\nSee Evidence section below."

    return "\n\n---\n\n".join(card_blocks)


def build_vuln_triage_prompt(
    bundle: FactBundle,
    question: str,
    chunks: list[dict],
) -> list[dict]:
    """Build the ``[system, user]`` message pair for VulnTriage generation."""
    l1_block = render_vuln_triage_l1(bundle)
    context_str = _format_chunks_for_prompt(chunks)
    allowed_labels = _citation_labels(chunks)
    allowed_str = "\n".join(f"- {label}" for label in allowed_labels) or "(none)"

    cve_list = ", ".join(facts.cve_id for facts in bundle.vuln_triage) or "(none)"

    user_content = (
        f"Question: {question}\n\n"
        f"Target CVE(s): {cve_list}\n\n"
        f"--- Pre-rendered L1 Block (do NOT repeat) ---\n"
        f"{l1_block}\n"
        f"--- End of L1 Block ---\n\n"
        f"--- Retrieved Context ---\n"
        f"{context_str}\n"
        f"--- End of Context ---\n\n"
        f"Allowed citation labels:\n{allowed_str}\n\n"
        "Produce ONLY the following sections, in order, as Markdown:\n\n"
        "**Exploitation Context** (L2)\n"
        "- One grounded claim per bullet about how the vulnerability is\n"
        "  exploited, what it grants, and any active-exploitation evidence.\n\n"
        "**Affected Versions** (L2)\n"
        "- Concrete affected and fixed version strings drawn from the\n"
        "  retrieved context. If versions are not in context, say so.\n\n"
        "**Mitigations** (L2)\n"
        "- Discrete, specific mitigations. If no grounded mitigations\n"
        "  exist, write exactly: No reliable mitigation guidance is\n"
        "  present in the retrieved context.\n\n"
        "**Evidence** (L1)\n"
        "- Bullet list referencing the citation labels you used, each on\n"
        "  its own line with a short (≤ 6-word) description.\n\n"
        "**Gaps** (L2)\n"
        "- What is missing from the retrieved context that would improve\n"
        "  the triage decision? Include at least the absence of IoCs,\n"
        "  CWE-specific mitigations, or detection rules if applicable.\n"
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# ThreatContext (§7.2)
# ---------------------------------------------------------------------------


def render_threat_context_l1(bundle: FactBundle) -> str:
    """Render the deterministic L1 block for a ThreatContext card."""
    if bundle.template != "ThreatContext":
        raise ValueError(
            f"ThreatContext renderer invoked on template={bundle.template!r}"
        )

    tc: ThreatContextFacts | None = bundle.threat_context
    lines: list[str] = []

    if tc is None or not tc.primary_entity:
        lines.append("## MITRE ATT&CK / Threat Context (no specific entity resolved)")
    elif tc.primary_entity_kind == "technique":
        header = tc.primary_entity
        if tc.technique_name:
            header = f"{header} — {tc.technique_name}"
        lines.append(f"## {header}")
    elif tc.primary_entity_kind == "tactic":
        header = tc.primary_entity
        if tc.tactic_name:
            header = f"{header} — {tc.tactic_name}"
        lines.append(f"## {header}")
    elif tc.primary_entity_kind == "ioc":
        lines.append(f"## IoC: {tc.primary_entity}")
    else:
        lines.append(f"## Threat Context: {tc.primary_entity}")

    lines.append("")
    lines.append("**ATT&CK Context** (L1)")
    if tc and tc.attack_technique:
        lines.append(f"- Technique: {tc.attack_technique}")
        if tc.technique_name:
            lines.append(f"- Technique name: {tc.technique_name}")
    if tc and tc.tactic_id:
        tactic_display = tc.tactic_id
        if tc.tactic_name:
            tactic_display = f"{tc.tactic_name} ({tc.tactic_id})"
        lines.append(f"- Tactic: {tactic_display}")
    if tc is None or (not tc.attack_technique and not tc.tactic_id):
        lines.append("- No ATT&CK mapping resolved from retrieved context")

    lines.append("")
    lines.append("**Observed in Retrieved Context** (L1 aggregation)")
    if tc and tc.associated_cves:
        lines.append(
            f"- Associated CVEs ({len(tc.associated_cves)}): {_fmt_list(tc.associated_cves[:10])}"
        )
    else:
        lines.append("- Associated CVEs: none in retrieved context")

    lines.append("")
    lines.append("**IoC Summary** (L1 aggregation)")
    if tc and tc.iocs:
        ip_count = len(tc.iocs.get("ip", []))
        domain_count = len(tc.iocs.get("domain", []))
        hash_count = len(tc.iocs.get("hash", []))
        url_count = len(tc.iocs.get("url", []))
        lines.append(f"- IP addresses: {ip_count} observed")
        lines.append(f"- Domains: {domain_count} observed")
        lines.append(f"- Hashes: {hash_count} observed")
        if url_count:
            lines.append(f"- URLs: {url_count} observed")
    else:
        lines.append("- No structured IoCs in retrieved context")

    return "\n".join(lines)


def build_threat_context_prompt(
    bundle: FactBundle,
    question: str,
    chunks: list[dict],
) -> list[dict]:
    """Build the ``[system, user]`` message pair for ThreatContext generation."""
    l1_block = render_threat_context_l1(bundle)
    context_str = _format_chunks_for_prompt(chunks)
    allowed_labels = _citation_labels(chunks)
    allowed_str = "\n".join(f"- {label}" for label in allowed_labels) or "(none)"

    primary = "(none)"
    if bundle.threat_context and bundle.threat_context.primary_entity:
        primary = (
            f"{bundle.threat_context.primary_entity} "
            f"({bundle.threat_context.primary_entity_kind})"
        )

    user_content = (
        f"Question: {question}\n\n"
        f"Primary entity: {primary}\n\n"
        f"--- Pre-rendered L1 Block (do NOT repeat) ---\n"
        f"{l1_block}\n"
        f"--- End of L1 Block ---\n\n"
        f"--- Retrieved Context ---\n"
        f"{context_str}\n"
        f"--- End of Context ---\n\n"
        f"Allowed citation labels:\n{allowed_str}\n\n"
        "Produce ONLY the following sections, in order, as Markdown:\n\n"
        "**Threat Actors and Malware** (L2)\n"
        "- Named threat groups and malware families attributed in the\n"
        "  retrieved context. If none are named, say so explicitly.\n\n"
        "**Defensive Guidance** (L2)\n"
        "- Concrete detection or mitigation recommendations grounded in\n"
        "  the retrieved context. Prefer actionable items (log signals,\n"
        "  network detections, patches) over generic advice.\n\n"
        "**Evidence** (L1)\n"
        "- Bullet list of the citation labels you used, each with a\n"
        "  short (≤ 6-word) description.\n\n"
        "**Gaps** (L2)\n"
        "- Missing elements such as detection rules (YARA/Sigma),\n"
        "  attribution, or IoC enrichment.\n"
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# CrossSourceCompare (§7.3)
# ---------------------------------------------------------------------------


def _comparison_table_rows(cross: CrossSourceCompareFacts) -> list[str]:
    """Build the deterministic comparison-table rows for the L1 block."""
    rows: list[str] = [
        "| CVE | CVSS | KEV Listed | Ransomware | Vendor |",
        "|---|---|---|---|---|",
    ]
    for facts in cross.entities:
        cvss_display = f"{facts.cvss_score}" if facts.cvss_score is not None else "—"
        vendor_display = _fmt_list(facts.vendors) if facts.vendors else "—"
        rows.append(
            f"| {facts.cve_id} | {cvss_display} | "
            f"{_fmt_bool(facts.kev_listed)} | "
            f"{_fmt_bool(facts.ransomware_use)} | "
            f"{vendor_display} |"
        )
    return rows


def _top_matches_lines(cross: CrossSourceCompareFacts) -> list[str]:
    """Order entities by triage signal + CVSS, render a ranked list."""
    # Ensure every entity carries a triage signal.
    for facts in cross.entities:
        if facts.triage_signal is None:
            annotate_vuln_triage_bundle(facts)

    severity_rank = {"critical": 0, "high": 1, "moderate": 2, "low": 3, "unknown": 4}

    def sort_key(facts: VulnTriageFacts) -> tuple[int, float]:
        sig = facts.triage_signal or {}
        sev = sig.get("severity", "unknown")
        # Higher CVSS should come first within the same severity band;
        # negate so ascending sort gives the right order.
        return (severity_rank.get(sev, 4), -(facts.cvss_score or 0.0))

    sorted_entities = sorted(cross.entities, key=sort_key)

    lines: list[str] = []
    for i, facts in enumerate(sorted_entities, 1):
        sig = facts.triage_signal or {}
        label = sig.get("label", "UNKNOWN — Manual review required")
        rationale = sig.get("rationale", "—")
        lines.append(f"{i}. {facts.cve_id} — {label} ({rationale})")
    return lines


def render_cross_source_l1(bundle: FactBundle) -> str:
    """Render the deterministic L1 block for a CrossSourceCompare card."""
    if bundle.template != "CrossSourceCompare":
        raise ValueError(
            f"CrossSourceCompare renderer invoked on template={bundle.template!r}"
        )

    cross = bundle.cross_source or CrossSourceCompareFacts()

    total = len(cross.entities)
    both_sources = sum(
        1
        for facts in cross.entities
        if {"nvd", "cisa_kev"}.issubset(set(facts.sources))
    )
    single_source = sum(1 for facts in cross.entities if len(facts.sources) == 1)
    no_data = sum(1 for facts in cross.entities if not facts.sources)

    lines: list[str] = [
        "## Cross-Source Comparison",
        "",
        "**Summary** (L1)",
        f"- Entities evaluated: {total}",
        f"- Entities with data from NVD and CISA KEV: {both_sources}",
        f"- Entities with data from only one source: {single_source}",
    ]
    if no_data:
        lines.append(f"- Entities requested but not present in retrieved context: {no_data}")

    if total:
        lines.append("")
        lines.append("**Comparison Table** (L1 deterministic)")
        lines.extend(_comparison_table_rows(cross))

        lines.append("")
        lines.append("**Top Matches by Triage Signal** (L1)")
        lines.extend(_top_matches_lines(cross))

    return "\n".join(lines)


def build_cross_source_prompt(
    bundle: FactBundle,
    question: str,
    chunks: list[dict],
) -> list[dict]:
    """Build the ``[system, user]`` message pair for CrossSourceCompare generation."""
    l1_block = render_cross_source_l1(bundle)
    context_str = _format_chunks_for_prompt(chunks)
    allowed_labels = _citation_labels(chunks)
    allowed_str = "\n".join(f"- {label}" for label in allowed_labels) or "(none)"

    entity_list = (
        ", ".join(facts.cve_id for facts in (bundle.cross_source.entities if bundle.cross_source else []))
        or "(none identified in query — use CVEs from retrieved context)"
    )

    user_content = (
        f"Question: {question}\n\n"
        f"Entities to compare: {entity_list}\n\n"
        f"--- Pre-rendered L1 Block (do NOT repeat) ---\n"
        f"{l1_block}\n"
        f"--- End of L1 Block ---\n\n"
        f"--- Retrieved Context ---\n"
        f"{context_str}\n"
        f"--- End of Context ---\n\n"
        f"Allowed citation labels:\n{allowed_str}\n\n"
        "Produce ONLY the following sections, in order, as Markdown:\n\n"
        "**Source Agreement** (L2)\n"
        "- Where do the sources agree or disagree on CVSS, severity,\n"
        "  exploitation status, or affected products? Cite each claim.\n\n"
        "**Cross-Source Facts** (L2)\n"
        "- Each bullet is one fact that a single source cannot support\n"
        "  alone (e.g., 'NVD records CVSS 9.8 [Source: nvd_X] while CISA\n"
        "  KEV confirms active exploitation [Source: kev_X]').\n\n"
        "**Evidence** (L1)\n"
        "- Bullet list of citation labels used, each with a short\n"
        "  description.\n\n"
        "**Gaps** (L2)\n"
        "- Entities requested but not retrieved, or source combinations\n"
        "  that would improve analyst confidence.\n"
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def render_l1(bundle: FactBundle) -> str:
    """Dispatch to the correct L1 renderer based on ``bundle.template``."""
    if bundle.template == "VulnTriage":
        return render_vuln_triage_l1(bundle)
    if bundle.template == "ThreatContext":
        return render_threat_context_l1(bundle)
    if bundle.template == "CrossSourceCompare":
        return render_cross_source_l1(bundle)
    raise ValueError(f"Unknown template: {bundle.template!r}")


def build_prompt(bundle: FactBundle, question: str, chunks: list[dict]) -> list[dict]:
    """Dispatch to the correct prompt builder based on ``bundle.template``."""
    if bundle.template == "VulnTriage":
        return build_vuln_triage_prompt(bundle, question, chunks)
    if bundle.template == "ThreatContext":
        return build_threat_context_prompt(bundle, question, chunks)
    if bundle.template == "CrossSourceCompare":
        return build_cross_source_prompt(bundle, question, chunks)
    raise ValueError(f"Unknown template: {bundle.template!r}")
