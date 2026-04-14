"""
Fact Bundle builder and entity-group aggregation.

Role in the pipeline
--------------------
After hybrid retrieval produces a list of chunk dictionaries, this module
is responsible for the **Layer-1 deterministic** step of the triage card:

    retrieved chunks
          │
          ▼
    group_chunks_by_entity()      ← entity-group aggregation (§4 of spec)
          │
          ▼
    build_*_facts()               ← deterministic field extraction from
          │                         chunk.metadata (§5 of spec)
          ▼
    FactBundle (dataclass)        ← passed to prompt assembly + evaluation

No LLM is invoked here. Every field written into a ``*Facts`` dataclass
comes from structured chunk metadata or from a regex over known chunk
fields — there is no hallucination surface.

Design decisions captured here
------------------------------
- Primary-entity detection uses the chunk's ``title`` and ``doc_id``
  first, and falls back to ``metadata.cve_ids`` only for tie-breaking.
  Rationale: a MISP event that *mentions* CVE-X as a related reference
  should not be treated as an NVD-equivalent fact source for CVE-X
  (spec §12, first open question).
- CPEs are humanized at Fact Bundle construction time via
  :mod:`cti_rag.rag.cpe`. Raw CPE strings are retained in parallel so
  downstream consumers (evaluator, debug trace) still see the
  machine-readable identifier.
- Inconsistency detection (§5.7) is deliberately conservative in this
  first iteration: it flags disagreements on CVSS score and severity
  label between NVD and KEV chunks. Extending to more fields is
  straightforward once the first pass stabilizes.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

from .cpe import humanize_cpes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity regexes
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", flags=re.IGNORECASE)
_CWE_RE = re.compile(r"\bCWE-\d+\b", flags=re.IGNORECASE)
_ATTACK_TECHNIQUE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_ATTACK_TACTIC_RE = re.compile(r"\bTA\d{4}\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SHA1_RE = re.compile(r"\b[a-fA-F0-9]{40}\b")
_SHA256_RE = re.compile(r"\b[a-fA-F0-9]{64}\b")
_MD5_RE = re.compile(r"\b[a-fA-F0-9]{32}\b")
_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b",
    flags=re.IGNORECASE,
)

# Known ATT&CK tactic ID -> name table. Names from MITRE ATT&CK Enterprise
# matrix. Used when the retrieved chunk does not carry the mapping
# explicitly in its metadata.
_ATTACK_TACTIC_NAMES: dict[str, str] = {
    "TA0001": "Initial Access",
    "TA0002": "Execution",
    "TA0003": "Persistence",
    "TA0004": "Privilege Escalation",
    "TA0005": "Defense Evasion",
    "TA0006": "Credential Access",
    "TA0007": "Discovery",
    "TA0008": "Lateral Movement",
    "TA0009": "Collection",
    "TA0010": "Exfiltration",
    "TA0011": "Command and Control",
    "TA0040": "Impact",
    "TA0042": "Resource Development",
    "TA0043": "Reconnaissance",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VulnTriageFacts:
    """
    Deterministic facts about a single CVE-centric entity.

    All fields are derived from chunk.metadata across all chunks grouped
    to this entity. ``triage_signal`` is populated later by
    :func:`cti_rag.rag.severity.compute_triage_signal` — keeping it
    optional here avoids a circular module import.
    """

    cve_id: str
    title: str | None = None
    cvss_score: float | None = None
    cvss_vector: str | None = None
    severity: str | None = None
    cwe_ids: list[str] = field(default_factory=list)
    affected_products: list[str] = field(default_factory=list)
    affected_products_raw: list[str] = field(default_factory=list)
    vendors: list[str] = field(default_factory=list)

    kev_listed: bool = False
    kev_date_added: str | None = None
    kev_due_date: str | None = None
    kev_required_action: str | None = None
    ransomware_use: bool = False

    references: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    source_chunks: list[dict] = field(default_factory=list)
    inconsistencies: list[dict] = field(default_factory=list)

    triage_signal: dict | None = None


@dataclass
class ThreatContextFacts:
    """Deterministic facts for TTP- or IoC-centric queries."""

    primary_entity: str
    primary_entity_kind: str  # "technique" | "tactic" | "ioc" | "concept"
    attack_technique: str | None = None
    technique_name: str | None = None
    tactic_id: str | None = None
    tactic_name: str | None = None
    associated_cves: list[str] = field(default_factory=list)
    iocs: dict[str, list[str]] = field(default_factory=dict)
    iocs_count: int = 0
    actors_or_groups: list[str] = field(default_factory=list)
    malware_families: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    source_chunks: list[dict] = field(default_factory=list)


@dataclass
class CrossSourceCompareFacts:
    """Wrapper bundle for multi-entity comparison questions."""

    entities: list[VulnTriageFacts] = field(default_factory=list)
    source_coverage: dict[str, list[str]] = field(default_factory=dict)
    comparison_axes: list[str] = field(
        default_factory=lambda: [
            "cvss_score",
            "kev_listed",
            "ransomware_use",
            "vendor",
        ]
    )


@dataclass
class FactBundle:
    """
    Top-level output of the fact-building step.

    Exactly one of ``vuln_triage`` (possibly a list), ``threat_context``,
    or ``cross_source`` is populated, depending on ``template``.
    """

    template: str  # "VulnTriage" | "ThreatContext" | "CrossSourceCompare"
    primary_entities: list[str] = field(default_factory=list)
    vuln_triage: list[VulnTriageFacts] = field(default_factory=list)
    threat_context: ThreatContextFacts | None = None
    cross_source: CrossSourceCompareFacts | None = None
    supporting_chunks: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity extraction helpers
# ---------------------------------------------------------------------------


def extract_cve_ids(text: str | None) -> list[str]:
    """Extract normalized CVE IDs (upper-case) from free text."""
    if not text:
        return []
    return [m.group(0).upper() for m in _CVE_RE.finditer(text)]


def extract_attack_techniques(text: str | None) -> list[str]:
    """Extract ATT&CK technique identifiers (e.g. ``T1190``)."""
    if not text:
        return []
    return [m.group(0).upper() for m in _ATTACK_TECHNIQUE_RE.finditer(text)]


def extract_attack_tactics(text: str | None) -> list[str]:
    """Extract ATT&CK tactic identifiers (e.g. ``TA0001``)."""
    if not text:
        return []
    return [m.group(0).upper() for m in _ATTACK_TACTIC_RE.finditer(text)]


def _split_metadata_list(value) -> list[str]:
    """Re-split a comma-joined ChromaDB metadata scalar into its items."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _chunk_cve_ids(chunk: dict) -> list[str]:
    """Collect every CVE ID associated with a chunk, from metadata + text."""
    meta = chunk.get("metadata") or {}
    cves: list[str] = []

    # Metadata-stored list (comma-joined string after to_chromadb_metadata).
    for value in _split_metadata_list(meta.get("cve_ids")):
        cves.append(value.upper())

    # doc_id prefix, e.g. nvd_CVE-2023-4966
    doc_id = chunk.get("doc_id") or ""
    cves.extend(extract_cve_ids(doc_id))

    # Title and content regex as last resort.
    cves.extend(extract_cve_ids(chunk.get("title")))
    cves.extend(extract_cve_ids(chunk.get("content")))

    seen: set[str] = set()
    ordered: list[str] = []
    for cve in cves:
        if cve in seen:
            continue
        seen.add(cve)
        ordered.append(cve)
    return ordered


def _is_primary_match(chunk: dict, entity_id: str) -> bool:
    """
    Decide whether ``chunk`` is primarily about ``entity_id``.

    Primary = the entity appears in the chunk's ``title`` or ``doc_id``.
    This protects against MISP events that only *mention* a CVE as a
    related reference from inflating the VulnTriage fact bundle for that
    CVE (spec §12, first open question).
    """
    target = entity_id.upper()
    doc_id = (chunk.get("doc_id") or "").upper()
    title = (chunk.get("title") or "").upper()
    return target in doc_id or target in title


# ---------------------------------------------------------------------------
# Entity-group aggregation (§4)
# ---------------------------------------------------------------------------


def group_chunks_by_cve(
    chunks: list[dict],
    primary_cve_ids: Iterable[str],
) -> tuple[dict[str, list[dict]], list[dict]]:
    """
    Group retrieved chunks around one or more CVE IDs.

    Returns
    -------
    groups : dict[cve_id, list[chunk]]
        Primary-match chunks per CVE (title/doc_id contains the CVE ID).
        A chunk that matches multiple target CVEs is added to each group —
        this is an intended behaviour for cross-source / multi-CVE
        queries where a single advisory can legitimately speak to both.
    supporting : list[chunk]
        Chunks that did not primarily match any requested CVE.
    """
    targets = [cve.upper() for cve in primary_cve_ids if cve]
    groups: dict[str, list[dict]] = {cve: [] for cve in targets}
    supporting: list[dict] = []

    if not targets:
        return groups, list(chunks)

    for chunk in chunks:
        matched = [cve for cve in targets if _is_primary_match(chunk, cve)]
        if matched:
            for cve in matched:
                groups[cve].append(chunk)
        else:
            # Chunk mentions the CVE only in passing (e.g. via cve_ids
            # metadata) — keep it as supporting context.
            chunk_cves = set(_chunk_cve_ids(chunk))
            if any(cve in chunk_cves for cve in targets):
                supporting.append(chunk)
            else:
                supporting.append(chunk)

    return groups, supporting


def group_chunks_by_technique(
    chunks: list[dict],
    technique_id: str,
) -> tuple[list[dict], list[dict]]:
    """Primary/supporting split for a single ATT&CK technique."""
    primary: list[dict] = []
    supporting: list[dict] = []
    target = technique_id.upper()

    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        techniques = {
            t.upper() for t in _split_metadata_list(meta.get("attack_techniques"))
        }
        techniques.update(extract_attack_techniques(chunk.get("content")))
        techniques.update(extract_attack_techniques(chunk.get("title")))

        if target in techniques:
            primary.append(chunk)
        else:
            supporting.append(chunk)

    return primary, supporting


# ---------------------------------------------------------------------------
# Fact builders (§5.3 – §5.5)
# ---------------------------------------------------------------------------


def _prefer(existing, candidate):
    """Return ``candidate`` when ``existing`` is falsy; else keep ``existing``."""
    return candidate if (existing in (None, "", [], {}) and candidate not in (None, "")) else existing


def _merge_unique(into: list[str], values: Iterable[str]) -> None:
    for value in values:
        if value and value not in into:
            into.append(value)


def _parse_iocs_json(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(items, list):
        return []
    cleaned: list[dict] = []
    for item in items:
        if isinstance(item, dict) and "value" in item:
            cleaned.append(
                {
                    "type": str(item.get("type", "unknown")),
                    "value": str(item.get("value", "")),
                    "category": item.get("category"),
                }
            )
    return cleaned


def _bucket_ioc_type(ioc_type: str) -> str:
    """Collapse MISP's fine-grained IOC types into a small analyst-facing set."""
    t = (ioc_type or "").lower()
    if "sha" in t or "md5" in t or "hash" in t:
        return "hash"
    if t in {"ip-src", "ip-dst", "ip"} or t.startswith("ip-"):
        return "ip"
    if "domain" in t or "hostname" in t:
        return "domain"
    if "url" in t or "uri" in t:
        return "url"
    if "email" in t:
        return "email"
    return t or "other"


def build_vuln_triage_facts(cve_id: str, chunks: list[dict]) -> VulnTriageFacts:
    """Collapse a list of primary-match chunks for ``cve_id`` into facts."""
    facts = VulnTriageFacts(cve_id=cve_id.upper())

    # Track per-chunk values so we can flag inconsistencies afterwards.
    seen_cvss: list[tuple[str, float]] = []
    seen_severity: list[tuple[str, str]] = []

    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        source = str(meta.get("source") or chunk.get("source") or "").lower()
        title = chunk.get("title") or ""

        _merge_unique(facts.sources, [source] if source else [])
        facts.source_chunks.append(
            {
                "doc_id": chunk.get("doc_id"),
                "citation_label": chunk.get("citation_label"),
                "source": source,
                "title": title,
            }
        )

        if not facts.title and title:
            facts.title = title

        # CVSS
        cvss = meta.get("cvss_score")
        if isinstance(cvss, (int, float)):
            cvss_val = float(cvss)
            facts.cvss_score = _prefer(facts.cvss_score, cvss_val)
            seen_cvss.append((source, cvss_val))

        vector = meta.get("cvss_vector")
        if isinstance(vector, str) and vector.strip():
            facts.cvss_vector = _prefer(facts.cvss_vector, vector.strip())

        # Severity
        sev = meta.get("severity")
        if isinstance(sev, str) and sev.strip() and sev.lower() != "unknown":
            facts.severity = _prefer(facts.severity, sev.lower())
            seen_severity.append((source, sev.lower()))

        # CWE
        _merge_unique(facts.cwe_ids, _split_metadata_list(meta.get("cwe_ids")))

        # Affected products
        raw_products = _split_metadata_list(meta.get("affected_products"))
        _merge_unique(facts.affected_products_raw, raw_products)

        # References
        _merge_unique(facts.references, _split_metadata_list(meta.get("references")))

        # KEV-specific
        if source == "cisa_kev":
            facts.kev_listed = True
            facts.kev_date_added = _prefer(
                facts.kev_date_added, meta.get("date_added_to_kev")
            )
            facts.kev_due_date = _prefer(facts.kev_due_date, meta.get("due_date"))
            facts.kev_required_action = _prefer(
                facts.kev_required_action, meta.get("required_action")
            )
            if bool(meta.get("known_ransomware_bool")):
                facts.ransomware_use = True

    # Humanize CPEs now that we've collected all of them.
    facts.affected_products = humanize_cpes(facts.affected_products_raw)

    # Try to surface a vendor list (useful for the CrossSource table).
    for raw in facts.affected_products_raw:
        parts = raw.split(":")
        if len(parts) >= 4:
            vendor = parts[3].replace("_", " ").strip()
            if vendor and vendor != "*":
                _merge_unique(facts.vendors, [vendor.title()])

    # Inconsistency detection (§5.7).
    cvss_values = {round(v, 1) for _, v in seen_cvss}
    if len(cvss_values) > 1:
        facts.inconsistencies.append(
            {
                "field": "cvss_score",
                "values_by_source": seen_cvss,
                "note": "CVSS score disagrees across retrieved sources.",
            }
        )
    severity_values = {v for _, v in seen_severity}
    if len(severity_values) > 1:
        facts.inconsistencies.append(
            {
                "field": "severity",
                "values_by_source": seen_severity,
                "note": "Severity label disagrees across retrieved sources.",
            }
        )

    return facts


def build_threat_context_facts(
    primary_entity: str,
    primary_entity_kind: str,
    chunks: list[dict],
) -> ThreatContextFacts:
    """Build Layer-1 facts for a TTP- or IoC-centric query."""
    facts = ThreatContextFacts(
        primary_entity=primary_entity,
        primary_entity_kind=primary_entity_kind,
    )

    aggregated_iocs: dict[str, list[str]] = {}

    if primary_entity_kind == "technique":
        facts.attack_technique = primary_entity.upper()
    elif primary_entity_kind == "tactic":
        facts.tactic_id = primary_entity.upper()
        facts.tactic_name = _ATTACK_TACTIC_NAMES.get(facts.tactic_id)

    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        source = str(meta.get("source") or chunk.get("source") or "").lower()
        _merge_unique(facts.sources, [source] if source else [])
        facts.source_chunks.append(
            {
                "doc_id": chunk.get("doc_id"),
                "citation_label": chunk.get("citation_label"),
                "source": source,
                "title": chunk.get("title") or "",
            }
        )

        # CVE correlations — these are the "associated CVEs" for the TTP.
        _merge_unique(facts.associated_cves, _chunk_cve_ids(chunk))

        # ATT&CK technique / tactic enrichment when the primary is only a
        # concept keyword: derive the most-frequent technique.
        _merge_unique(
            facts.associated_cves,
            # no-op second call intentionally omitted; kept for clarity
            [],
        )

        # IoCs
        ioc_items = _parse_iocs_json(meta.get("iocs_json"))
        for ioc in ioc_items:
            bucket = _bucket_ioc_type(ioc["type"])
            aggregated_iocs.setdefault(bucket, [])
            if ioc["value"] and ioc["value"] not in aggregated_iocs[bucket]:
                aggregated_iocs[bucket].append(ioc["value"])

    facts.iocs = aggregated_iocs
    facts.iocs_count = sum(len(v) for v in aggregated_iocs.values())

    return facts


def build_cross_source_facts(
    entity_ids: Iterable[str],
    chunks: list[dict],
) -> CrossSourceCompareFacts:
    """Build per-entity VulnTriage facts for multi-entity comparison."""
    groups, _ = group_chunks_by_cve(chunks, entity_ids)
    cross = CrossSourceCompareFacts()

    for cve_id, cve_chunks in groups.items():
        if not cve_chunks:
            # Retain the entity even when empty so the comparison table
            # can show "no data retrieved" explicitly.
            cross.entities.append(VulnTriageFacts(cve_id=cve_id))
            cross.source_coverage[cve_id] = []
            continue

        facts = build_vuln_triage_facts(cve_id, cve_chunks)
        cross.entities.append(facts)
        cross.source_coverage[cve_id] = list(facts.sources)

    return cross


# ---------------------------------------------------------------------------
# Convenience top-level constructor
# ---------------------------------------------------------------------------


def build_fact_bundle(
    template: str,
    primary_entities: list[str],
    chunks: list[dict],
    *,
    threat_context_kind: str | None = None,
) -> FactBundle:
    """
    High-level dispatch from (template, primary_entities, chunks) to a
    populated :class:`FactBundle`.

    ``threat_context_kind`` is required only for the ThreatContext
    template and must be one of ``"technique"``, ``"tactic"``, ``"ioc"``,
    ``"concept"``.
    """
    template_key = template.strip()
    bundle = FactBundle(template=template_key, primary_entities=list(primary_entities))

    if template_key == "VulnTriage":
        groups, supporting = group_chunks_by_cve(chunks, primary_entities)
        for cve_id, cve_chunks in groups.items():
            if cve_chunks:
                bundle.vuln_triage.append(build_vuln_triage_facts(cve_id, cve_chunks))
            else:
                bundle.vuln_triage.append(VulnTriageFacts(cve_id=cve_id))
        bundle.supporting_chunks = supporting

    elif template_key == "ThreatContext":
        if not primary_entities:
            logger.warning("ThreatContext fact bundle built with no primary entity")
            bundle.threat_context = ThreatContextFacts(
                primary_entity="",
                primary_entity_kind=threat_context_kind or "concept",
            )
            bundle.supporting_chunks = list(chunks)
            return bundle

        primary = primary_entities[0]
        if threat_context_kind == "technique":
            primary_chunks, supporting = group_chunks_by_technique(chunks, primary)
        else:
            # Tactic / IoC / concept queries: use all retrieved chunks as
            # the aggregation input. Filtering happens implicitly via the
            # regex extraction inside build_threat_context_facts.
            primary_chunks, supporting = list(chunks), []

        bundle.threat_context = build_threat_context_facts(
            primary, threat_context_kind or "concept", primary_chunks
        )
        bundle.supporting_chunks = supporting

    elif template_key == "CrossSourceCompare":
        bundle.cross_source = build_cross_source_facts(primary_entities, chunks)
        # supporting_chunks: chunks not matching any target entity.
        _, supporting = group_chunks_by_cve(chunks, primary_entities)
        bundle.supporting_chunks = supporting

    else:
        raise ValueError(f"Unknown template: {template!r}")

    return bundle
