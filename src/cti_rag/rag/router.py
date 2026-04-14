"""
Query Router for Phase 2 templates.

Purpose
-------
Deterministic, table-driven classification of an incoming analyst query
into one of three templates:

- ``VulnTriage`` — CVE-centric single- or multi-CVE questions.
- ``ThreatContext`` — TTP / IoC / MITRE-ATT&CK-concept questions.
- ``CrossSourceCompare`` — multi-entity comparison or explicit
  multi-source questions.

Rule table (see docs/triage-template-spec.md §3.1; first match wins)
--------------------------------------------------------------------
1. ≥ 2 distinct source names mentioned (NVD / CISA KEV / CISA / MISP)
   → ``CrossSourceCompare``
2. Comparison keyword (``compare``, ``vs.``, ``difference between``,
   ``versus``, ``both X and Y``, ``which .* appear``)
   → ``CrossSourceCompare``
3. CVE-ID pattern matches
   → ``VulnTriage``
4. ATT&CK identifier pattern matches (``T\\d{4}`` / ``TA\\d{4}``)
   → ``ThreatContext``
5. IoC pattern matches (IPv4, SHA-1/256, domain-name)
   → ``ThreatContext``
6. ATT&CK concept keywords (``MITRE ATT&CK`` or a tactic name)
   → ``ThreatContext``
7. Fallback
   → ``VulnTriage``

The router returns a :class:`RoutingDecision` with:
- ``template`` — one of the three template names above
- ``primary_entities`` — CVE IDs / technique IDs / tactic IDs / IoC values
  to drive entity-group aggregation downstream
- ``threat_context_kind`` — for ThreatContext only: ``"technique"`` /
  ``"tactic"`` / ``"ioc"`` / ``"concept"``
- ``rule_matched`` — the rule number that fired; recorded for the
  evaluation trace so we can debug routing decisions per query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

TemplateName = Literal["VulnTriage", "ThreatContext", "CrossSourceCompare"]
ThreatContextKind = Literal["technique", "tactic", "ioc", "concept"]


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", flags=re.IGNORECASE)
_ATTACK_TECHNIQUE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_ATTACK_TACTIC_RE = re.compile(r"\bTA\d{4}\b")

_IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")
_SHA1_RE = re.compile(r"\b[a-fA-F0-9]{40}\b")
_SHA256_RE = re.compile(r"\b[a-fA-F0-9]{64}\b")
_MD5_RE = re.compile(r"\b[a-fA-F0-9]{32}\b")
# Domain regex deliberately excludes two-letter TLDs mid-sentence by
# requiring a TLD length of ≥ 3 — this avoids catching things like
# "CVE-2023-1234.is" that can appear as tokenisation artifacts.
_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:com|org|net|io|gov|edu|uk|de|ch|info|biz|xyz|cloud|dev|online|app|co|ru)\b",
    flags=re.IGNORECASE,
)

# Comparison keyword patterns. ``vs.`` and ``vs`` are handled with a
# trailing lookahead (whitespace / end-of-string) instead of ``\b``,
# because ``\b`` cannot match the boundary between a literal period and
# a following whitespace character.
_COMPARISON_RE = re.compile(
    r"\b(?:compare|comparison|difference between|versus|both .* and|which .* appear)\b"
    r"|\bvs\.?(?=\s|$)",
    flags=re.IGNORECASE,
)

_MITRE_ATTACK_RE = re.compile(r"\bMITRE\s+ATT&?CK\b", flags=re.IGNORECASE)

_SOURCE_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("cisa_kev", re.compile(r"\bCISA\s+KEV\b", flags=re.IGNORECASE)),
    ("nvd", re.compile(r"\bNVD\b")),
    ("misp", re.compile(r"\bMISP\b")),
    # "CISA" (without KEV) matches a CISA advisory mention. We strip
    # already-matched CISA KEV spans before checking this pattern so the
    # two do not double-count.
    ("cisa_advisory", re.compile(r"\bCISA\b(?!\s+KEV)", flags=re.IGNORECASE)),
)

# Tactic concept keywords → canonical (TA-id, display name).
# Order matters: multi-word phrases should match before single-word ones
# to avoid e.g. "command and control" being partially matched as
# "command and scripting" or unrelated.
_TACTIC_KEYWORDS: tuple[tuple[re.Pattern, str, str], ...] = (
    (re.compile(r"\bcommand\s+and\s+control\b", flags=re.IGNORECASE), "TA0011", "Command and Control"),
    (re.compile(r"\bcommand\s+and\s+scripting\b", flags=re.IGNORECASE), "TA0002", "Execution"),
    (re.compile(r"\bresource\s+development\b", flags=re.IGNORECASE), "TA0042", "Resource Development"),
    (re.compile(r"\binitial\s+access\b", flags=re.IGNORECASE), "TA0001", "Initial Access"),
    (re.compile(r"\blateral\s+movement\b", flags=re.IGNORECASE), "TA0008", "Lateral Movement"),
    (re.compile(r"\bprivilege\s+escalation\b", flags=re.IGNORECASE), "TA0004", "Privilege Escalation"),
    (re.compile(r"\bdefense\s+evasion\b", flags=re.IGNORECASE), "TA0005", "Defense Evasion"),
    (re.compile(r"\bpersistence\b", flags=re.IGNORECASE), "TA0003", "Persistence"),
    (re.compile(r"\bcredential\s+access\b", flags=re.IGNORECASE), "TA0006", "Credential Access"),
    (re.compile(r"\bdiscovery\b", flags=re.IGNORECASE), "TA0007", "Discovery"),
    (re.compile(r"\bcollection\b", flags=re.IGNORECASE), "TA0009", "Collection"),
    (re.compile(r"\bexfiltration\b", flags=re.IGNORECASE), "TA0010", "Exfiltration"),
    (re.compile(r"\bimpact\b", flags=re.IGNORECASE), "TA0040", "Impact"),
    (re.compile(r"\bexecution\b", flags=re.IGNORECASE), "TA0002", "Execution"),
    (re.compile(r"\breconnaissance\b", flags=re.IGNORECASE), "TA0043", "Reconnaissance"),
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """Deterministic routing output for a single query."""

    template: TemplateName
    primary_entities: list[str] = field(default_factory=list)
    threat_context_kind: ThreatContextKind | None = None
    rule_matched: int = 0
    matched_sources: list[str] = field(default_factory=list)

    def as_trace_dict(self) -> dict:
        """Compact form for inclusion in evaluation traces."""
        return {
            "template": self.template,
            "primary_entities": list(self.primary_entities),
            "threat_context_kind": self.threat_context_kind,
            "rule_matched": self.rule_matched,
            "matched_sources": list(self.matched_sources),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_sources(query: str) -> list[str]:
    """Detect distinct CTI sources named in the query text."""
    sources: list[str] = []
    remaining = query

    for name, pattern in _SOURCE_PATTERNS:
        if pattern.search(remaining):
            if name not in sources:
                sources.append(name)
            # Strip all matches so a later, broader pattern (e.g. bare
            # "CISA" after "CISA KEV") does not double-count.
            remaining = pattern.sub(" ", remaining)

    return sources


def _detect_cves(query: str) -> list[str]:
    """Return unique normalized CVE IDs in order of first appearance."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _CVE_RE.finditer(query):
        cve = match.group(0).upper()
        if cve not in seen:
            seen.add(cve)
            ordered.append(cve)
    return ordered


def _detect_techniques(query: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _ATTACK_TECHNIQUE_RE.finditer(query):
        tid = match.group(0).upper()
        if tid not in seen:
            seen.add(tid)
            ordered.append(tid)
    return ordered


def _detect_tactics(query: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _ATTACK_TACTIC_RE.finditer(query):
        tid = match.group(0).upper()
        if tid not in seen:
            seen.add(tid)
            ordered.append(tid)
    return ordered


def _detect_iocs(query: str) -> list[str]:
    """Surface IOC-shaped tokens that should trigger the ThreatContext path."""
    hits: list[str] = []
    for pat in (_IPV4_RE, _SHA256_RE, _SHA1_RE, _MD5_RE, _DOMAIN_RE):
        hits.extend(match.group(0) for match in pat.finditer(query))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for hit in hits:
        if hit not in seen:
            seen.add(hit)
            ordered.append(hit)
    return ordered


def _detect_tactic_concept(query: str) -> tuple[str, str] | None:
    """Return (TA-id, display name) for the first tactic keyword hit."""
    for pattern, tactic_id, display in _TACTIC_KEYWORDS:
        if pattern.search(query):
            return tactic_id, display
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def classify_query(query: str) -> RoutingDecision:
    """Apply the routing rules and return a :class:`RoutingDecision`."""
    if not query or not query.strip():
        return RoutingDecision(template="VulnTriage", rule_matched=7)

    text = query.strip()

    # Rule 1: ≥ 2 distinct source names.
    sources = _detect_sources(text)
    if len(sources) >= 2:
        return RoutingDecision(
            template="CrossSourceCompare",
            primary_entities=_detect_cves(text),
            rule_matched=1,
            matched_sources=sources,
        )

    # Rule 2: comparison keyword.
    if _COMPARISON_RE.search(text):
        return RoutingDecision(
            template="CrossSourceCompare",
            primary_entities=_detect_cves(text),
            rule_matched=2,
            matched_sources=sources,
        )

    # Rule 3: CVE-ID match.
    cves = _detect_cves(text)
    if cves:
        return RoutingDecision(
            template="VulnTriage",
            primary_entities=cves,
            rule_matched=3,
            matched_sources=sources,
        )

    # Rule 4: ATT&CK identifier.
    techniques = _detect_techniques(text)
    if techniques:
        return RoutingDecision(
            template="ThreatContext",
            primary_entities=techniques,
            threat_context_kind="technique",
            rule_matched=4,
            matched_sources=sources,
        )
    tactics = _detect_tactics(text)
    if tactics:
        return RoutingDecision(
            template="ThreatContext",
            primary_entities=tactics,
            threat_context_kind="tactic",
            rule_matched=4,
            matched_sources=sources,
        )

    # Rule 5: IoC pattern.
    iocs = _detect_iocs(text)
    if iocs:
        return RoutingDecision(
            template="ThreatContext",
            primary_entities=iocs,
            threat_context_kind="ioc",
            rule_matched=5,
            matched_sources=sources,
        )

    # Rule 6: ATT&CK concept keywords (MITRE ATT&CK literal or tactic name).
    has_mitre = bool(_MITRE_ATTACK_RE.search(text))
    concept_hit = _detect_tactic_concept(text)
    if has_mitre or concept_hit is not None:
        if concept_hit is not None:
            tactic_id, _ = concept_hit
            return RoutingDecision(
                template="ThreatContext",
                primary_entities=[tactic_id],
                threat_context_kind="tactic",
                rule_matched=6,
                matched_sources=sources,
            )
        # MITRE ATT&CK mentioned but no specific tactic — concept mode.
        return RoutingDecision(
            template="ThreatContext",
            primary_entities=[],
            threat_context_kind="concept",
            rule_matched=6,
            matched_sources=sources,
        )

    # Rule 7: fallback.
    return RoutingDecision(
        template="VulnTriage",
        primary_entities=[],
        rule_matched=7,
        matched_sources=sources,
    )
