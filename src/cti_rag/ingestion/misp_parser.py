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
from datetime import datetime
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


def _extract_attack_techniques(event: dict) -> list[str]:
    """Extract MITRE ATT&CK technique IDs from MISP galaxies and tags."""
    techniques = []

    # From Galaxy clusters
    for galaxy in event.get("Galaxy", []):
        if "attack" in galaxy.get("type", "").lower():
            for cluster in galaxy.get("GalaxyCluster", []):
                # ATT&CK technique IDs are in the tag name or meta
                tag = cluster.get("tag_name", "")
                if "attack-pattern" in tag.lower():
                    # Extract technique ID like T1059
                    meta = cluster.get("meta", {})
                    ext_ids = meta.get("external_id", [])
                    techniques.extend(ext_ids)

    # From tags (e.g., "misp-galaxy:mitre-attack-pattern=\"T1059\"")
    for tag in event.get("Tag", []):
        tag_name = tag.get("name", "")
        if "mitre-attack" in tag_name.lower():
            # Try to extract technique ID
            for part in tag_name.split("\""):
                if part.startswith("T") and len(part) >= 5:
                    techniques.append(part.split(" -")[0].strip())

    return list(set(techniques))


def _extract_cve_ids(event: dict) -> list[str]:
    """Extract CVE IDs from MISP attributes."""
    cve_ids = []
    for attr in event.get("Attribute", []):
        if attr.get("type") == "vulnerability":
            value = attr.get("value", "")
            if value.upper().startswith("CVE-"):
                cve_ids.append(value.upper())
    return list(set(cve_ids))


def _extract_iocs(event: dict) -> list[IOCEntry]:
    """Extract IOC entries from MISP attributes."""
    iocs = []
    for attr in event.get("Attribute", []):
        attr_type = attr.get("type", "")
        if attr_type in _IOC_TYPES:
            iocs.append(IOCEntry(
                type=attr_type,
                value=attr.get("value", ""),
                category=attr.get("category"),
            ))
    # Also check Object attributes
    for obj in event.get("Object", []):
        for attr in obj.get("Attribute", []):
            attr_type = attr.get("type", "")
            if attr_type in _IOC_TYPES:
                iocs.append(IOCEntry(
                    type=attr_type,
                    value=attr.get("value", ""),
                    category=attr.get("category"),
                ))
    return iocs


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

    # Build content text
    content_parts = [f"Event: {event_info}"]

    # Add tag context
    tags = [tag.get("name", "") for tag in event.get("Tag", []) if tag.get("name")]
    if tags:
        content_parts.append(f"Tags: {', '.join(tags[:15])}")

    # Summarize attributes
    attr_summary = {}
    for attr in event.get("Attribute", []):
        t = attr.get("type", "other")
        attr_summary[t] = attr_summary.get(t, 0) + 1

    if attr_summary:
        summary_str = ", ".join(f"{count}x {atype}" for atype, count in attr_summary.items())
        content_parts.append(f"Indicators: {summary_str}")

    # Add key IOC values (limited to avoid chunk bloat)
    key_iocs = iocs[:10]
    if key_iocs:
        ioc_text = "; ".join(f"{ioc.type}: {ioc.value}" for ioc in key_iocs)
        content_parts.append(f"Key IOCs: {ioc_text}")

    if attack_techniques:
        content_parts.append(f"ATT&CK Techniques: {', '.join(attack_techniques)}")

    if cve_ids:
        content_parts.append(f"Related CVEs: {', '.join(cve_ids)}")

    content = "\n".join(content_parts)

    # Parse date
    pub_date = None
    timestamp = event.get("timestamp") or event.get("publish_timestamp")
    if timestamp:
        try:
            pub_date = datetime.fromtimestamp(int(timestamp))
        except (ValueError, TypeError, OSError):
            pass

    return CTIDocument(
        doc_id=f"misp_{event_id}",
        source=CTISourceType.MISP,
        title=f"MISP Event: {event_info[:120]}",
        content=content,
        severity=severity,
        cve_ids=cve_ids,
        attack_techniques=attack_techniques,
        iocs=iocs,
        published_date=pub_date,
        metadata={
            "misp_event_id": event_id,
            "misp_uuid": event.get("uuid", ""),
            "org": event.get("Orgc", {}).get("name", ""),
            "attribute_count": len(event.get("Attribute", [])),
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
