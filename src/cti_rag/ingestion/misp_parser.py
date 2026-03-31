"""
MISP Event Parser.

Parses MISP JSON event exports into unified CTIDocument models.
Each MISP event becomes one CTIDocument.

Data sources:
- CIRCL MISP default feeds (public, free)
- Any MISP-compatible JSON export

MISP events contain attributes (IOCs), galaxies (ATT&CK mappings),
and tags that provide rich context for threat intelligence.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from .models import CTIDocument, CTISourceType, SeverityLevel, IOCEntry

logger = logging.getLogger(__name__)

# MISP threat level mapping
_THREAT_LEVEL_MAP = {
    "1": SeverityLevel.HIGH,
    "2": SeverityLevel.MEDIUM,
    "3": SeverityLevel.LOW,
    "4": SeverityLevel.UNKNOWN,
}

# MISP attribute types that are IOCs
_IOC_TYPES = {
    "ip-src", "ip-dst", "domain", "hostname", "url",
    "md5", "sha1", "sha256", "filename",
    "email-src", "email-dst", "email-subject",
    "mutex", "regkey", "pattern-in-file",
}

_ATTACK_TECHNIQUE_ID_RE = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)
_CVE_ID_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
_IOC_TYPE_PRIORITY = {
    "domain": 0,
    "hostname": 1,
    "url": 2,
    "sha256": 3,
    "sha1": 4,
    "md5": 5,
    "filename": 6,
    "ip-src": 7,
    "ip-dst": 8,
    "email-src": 9,
    "email-dst": 10,
    "email-subject": 11,
    "mutex": 12,
    "regkey": 13,
    "pattern-in-file": 14,
}
_PRODUCT_SUFFIX_RE = re.compile(
    r"\s+(?:"
    r"remote code execution|"
    r"authentication bypass|"
    r"privilege escalation|"
    r"command injection|"
    r"sql injection|"
    r"information disclosure|"
    r"denial of service|"
    r"arbitrary file upload|"
    r"arbitrary file read|"
    r"directory traversal|"
    r"path traversal|"
    r"unsafe flight protocol property access|"
    r"vulnerabilit(?:y|ies)|"
    r"exploit(?:ation)?"
    r")\b.*$",
    re.IGNORECASE,
)
_SIGNAL_TAG_PREFIXES = ("misp-galaxy:", "kill-chain:", "tlp:", "misp:threat-level")


def _iter_event_attributes(event: dict):
    """Yield top-level and object-level MISP attributes."""
    for attr in event.get("Attribute", []):
        yield attr

    for obj in event.get("Object", []):
        for attr in obj.get("Attribute", []):
            yield attr


def _extract_attack_ids_from_value(value: str | list[str] | None) -> list[str]:
    """Extract normalized ATT&CK technique IDs from a text value or string list."""
    if isinstance(value, str):
        return [match.upper() for match in _ATTACK_TECHNIQUE_ID_RE.findall(value)]

    if isinstance(value, list):
        attack_ids = []
        for item in value:
            if isinstance(item, str):
                attack_ids.extend(_extract_attack_ids_from_value(item))
        return attack_ids

    return []


def _extract_attack_techniques(event: dict) -> list[str]:
    """Extract MITRE ATT&CK technique IDs from MISP galaxies and tags."""
    techniques = []

    # From Galaxy clusters
    for galaxy in event.get("Galaxy") or []:
        if "attack" in galaxy.get("type", "").lower():
            for cluster in galaxy.get("GalaxyCluster", []):
                techniques.extend(_extract_attack_ids_from_value(cluster.get("tag_name")))
                techniques.extend(_extract_attack_ids_from_value(cluster.get("value")))

                meta = cluster.get("meta") or {}
                techniques.extend(_extract_attack_ids_from_value(meta.get("external_id")))

    # From tags (e.g., "misp-galaxy:mitre-attack-pattern=\"Exploit Public-Facing Application - T1190\"")
    for tag in event.get("Tag", []):
        tag_name = tag.get("name", "")
        if "attack-pattern" in tag_name.lower():
            techniques.extend(_extract_attack_ids_from_value(tag_name))

    for attr in _iter_event_attributes(event):
        techniques.extend(_extract_attack_ids_from_value(attr.get("comment")))
        for tag in attr.get("Tag", []):
            techniques.extend(_extract_attack_ids_from_value(tag.get("name")))

    for obj in event.get("Object", []):
        techniques.extend(_extract_attack_ids_from_value(obj.get("comment")))

    seen = set()
    deduped = []
    for technique in techniques:
        if technique not in seen:
            seen.add(technique)
            deduped.append(technique)

    return deduped


def _extract_cve_ids(event: dict) -> list[str]:
    """Extract CVE IDs from MISP attributes."""
    candidates = [event.get("info", "")]

    for attr in _iter_event_attributes(event):
        candidates.append(attr.get("value", ""))
        candidates.append(attr.get("comment", ""))

    for obj in event.get("Object", []):
        candidates.append(obj.get("comment", ""))

    seen = set()
    cve_ids = []
    for value in candidates:
        for match in _CVE_ID_RE.findall(value or ""):
            cve_id = match.upper()
            if cve_id not in seen:
                seen.add(cve_id)
                cve_ids.append(cve_id)

    return cve_ids


def _extract_iocs(event: dict) -> list[IOCEntry]:
    """Extract IOC entries from MISP attributes."""
    iocs = []
    for attr in _iter_event_attributes(event):
        attr_type = attr.get("type", "")
        if attr_type in _IOC_TYPES:
            iocs.append(IOCEntry(
                type=attr_type,
                value=attr.get("value", ""),
                category=attr.get("category"),
            ))
    return iocs


def _select_key_iocs(event: dict, iocs: list[IOCEntry], limit: int = 10) -> list[IOCEntry]:
    """Prioritize higher-value, unique IOC entries for analyst-facing summaries."""
    attr_lookup: dict[tuple[str, str], dict] = {}
    for attr in _iter_event_attributes(event):
        key = (attr.get("type", ""), attr.get("value", ""))
        attr_lookup.setdefault(key, attr)

    seen = set()
    ranked = []
    for ioc in iocs:
        key = (ioc.type, ioc.value)
        if key in seen or not ioc.value:
            continue
        seen.add(key)

        attr = attr_lookup.get(key, {})
        ranked.append((
            0 if attr.get("to_ids") else 1,
            _IOC_TYPE_PRIORITY.get(ioc.type, 99),
            ioc.value.lower(),
            ioc,
        ))

    ranked.sort()
    return [item[-1] for item in ranked[:limit]]


def _extract_products_from_snippets(snippets: list[str]) -> list[str]:
    """Infer product names from CVE/vulnerability-bearing snippets."""
    products = []
    seen = set()

    for snippet in snippets:
        if not snippet:
            continue

        candidate = " ".join(snippet.split())
        if not candidate:
            continue

        lowered_candidate = candidate.lower()
        if not (
            _CVE_ID_RE.search(candidate)
            or any(keyword in lowered_candidate for keyword in ("vulnerab", "exploit", "rce", "authentication bypass"))
        ):
            continue

        candidate = re.sub(r"^[A-Z0-9][A-Z0-9_./-]*(?:\s+[A-Z0-9][A-Z0-9_./-]*){0,4}\s+", "", candidate)
        candidate = re.sub(r"\(?(CVE-\d{4}-\d{4,}).*$", "", candidate, flags=re.IGNORECASE)
        candidate = candidate.split(" - ", 1)[0]
        candidate = _PRODUCT_SUFFIX_RE.sub("", candidate).strip(" -:;,.'\"")

        if len(candidate) < 4:
            continue

        if candidate.lower() in {"osint", "threat brief", "security alert advisory"}:
            continue

        normalized = candidate.lower()
        if normalized not in seen:
            seen.add(normalized)
            products.append(candidate)

    return products[:5]


def _extract_context_snippets(event: dict) -> list[str]:
    """Keep a few high-signal analyst context snippets from comments."""
    snippets = []
    seen = set()

    for attr in _iter_event_attributes(event):
        comment = " ".join((attr.get("comment") or "").split())
        if not comment:
            continue

        has_cti_signal = (
            bool(_CVE_ID_RE.search(comment))
            or bool(_ATTACK_TECHNIQUE_ID_RE.search(comment))
            or attr.get("to_ids")
        )
        if not has_cti_signal:
            continue

        lowered = comment.lower()
        if lowered not in seen:
            seen.add(lowered)
            snippets.append(comment)

    return snippets[:5]


def _parse_misp_timestamp(value) -> datetime | None:
    """Parse a MISP unix timestamp into a UTC datetime."""
    if value in (None, "", 0, "0"):
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (ValueError, TypeError, OSError):
        return None


def _parse_misp_date(value: str | None) -> datetime | None:
    """Parse the event-level YYYY-MM-DD date when publish_timestamp is missing."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_misp_event(event: dict) -> CTIDocument | None:
    """
    Parse a single MISP event dict into a CTIDocument.

    Concatenates event info, attribute summaries, and tag labels
    into a readable text block for embedding.
    """
    event_id = event.get("id", event.get("uuid", "unknown"))
    event_info = event.get("info", "")

    if not event_info:
        return None

    # Extract structured data
    threat_level = event.get("threat_level_id", "4")
    severity = _THREAT_LEVEL_MAP.get(str(threat_level), SeverityLevel.UNKNOWN)
    attack_techniques = _extract_attack_techniques(event)
    cve_ids = _extract_cve_ids(event)
    iocs = _extract_iocs(event)
    key_iocs = _select_key_iocs(event, iocs)
    context_snippets = _extract_context_snippets(event)
    affected_products = _extract_products_from_snippets([event_info, *context_snippets])

    # Build content text
    content_parts = [f"Event: {event_info}"]

    if severity != SeverityLevel.UNKNOWN:
        content_parts.append(f"Severity: {severity.value.upper()}")

    # Add tag context
    tags = [
        tag.get("name", "")
        for tag in event.get("Tag", [])
        if tag.get("name") and tag.get("name", "").lower().startswith(_SIGNAL_TAG_PREFIXES)
    ]
    if tags:
        content_parts.append(f"Signal Tags: {', '.join(tags[:12])}")

    # Summarize attributes
    attr_summary = {}
    for attr in _iter_event_attributes(event):
        t = attr.get("type", "other")
        attr_summary[t] = attr_summary.get(t, 0) + 1

    if attr_summary:
        summary_items = sorted(attr_summary.items(), key=lambda item: (-item[1], item[0]))
        summary_str = ", ".join(f"{count}x {atype}" for atype, count in summary_items[:10])
        content_parts.append(f"Indicators: {summary_str}")

    # Add key IOC values (limited to avoid chunk bloat)
    if key_iocs:
        ioc_text = "; ".join(f"{ioc.type}: {ioc.value}" for ioc in key_iocs)
        content_parts.append(f"Key IOCs: {ioc_text}")

    if attack_techniques:
        content_parts.append(f"ATT&CK Techniques: {', '.join(attack_techniques)}")

    if cve_ids:
        content_parts.append(f"Related CVEs: {', '.join(cve_ids)}")

    if affected_products:
        content_parts.append(f"Related Products: {', '.join(affected_products)}")

    if context_snippets:
        content_parts.append(f"Observed Context: {'; '.join(context_snippets[:3])}")

    content = "\n".join(content_parts)

    # Prefer publish_timestamp for publication time; keep timestamp as last modification time.
    pub_date = (
        _parse_misp_timestamp(event.get("publish_timestamp"))
        or _parse_misp_date(event.get("date"))
        or _parse_misp_timestamp(event.get("timestamp"))
    )
    mod_date = _parse_misp_timestamp(event.get("timestamp"))

    return CTIDocument(
        doc_id=f"misp_{event_id}",
        source=CTISourceType.MISP,
        title=f"MISP Event: {event_info[:120]}",
        content=content,
        severity=severity,
        cve_ids=cve_ids,
        attack_techniques=attack_techniques,
        iocs=iocs,
        affected_products=affected_products,
        published_date=pub_date,
        modified_date=mod_date,
        metadata={
            "misp_event_id": event_id,
            "misp_uuid": event.get("uuid", ""),
            "org": event.get("Orgc", {}).get("name", ""),
            "attribute_count": len(event.get("Attribute", [])),
            "publish_timestamp": event.get("publish_timestamp", ""),
            "timestamp": event.get("timestamp", ""),
        },
    )


def parse_misp_file(filepath: Path) -> list[CTIDocument]:
    """Parse a MISP JSON export file (single event or event list)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    # Handle different MISP export formats
    if isinstance(data, dict):
        if "Event" in data:
            event = data["Event"]
            doc = parse_misp_event(event)
            if doc:
                documents.append(doc)
        elif "response" in data:
            for item in data["response"]:
                event = item.get("Event", item)
                doc = parse_misp_event(event)
                if doc:
                    documents.append(doc)
        else:
            doc = parse_misp_event(data)
            if doc:
                documents.append(doc)
    elif isinstance(data, list):
        for item in data:
            event = item.get("Event", item) if isinstance(item, dict) else {}
            doc = parse_misp_event(event)
            if doc:
                documents.append(doc)

    logger.info(f"Parsed {len(documents)} MISP events from {filepath.name}")
    return documents


def parse_misp_directory(directory: Path) -> list[CTIDocument]:
    """Parse all MISP JSON files in a directory."""
    all_docs = []
    json_files = sorted(directory.glob("*.json"))

    for filepath in json_files:
        try:
            docs = parse_misp_file(filepath)
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"Error parsing MISP file {filepath.name}: {e}")

    logger.info(f"Total MISP documents: {len(all_docs)}")
    return all_docs
