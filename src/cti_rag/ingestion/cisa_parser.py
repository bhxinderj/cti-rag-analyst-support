"""
CISA Data Parsers.

Handles two CISA data sources:
1. Known Exploited Vulnerabilities (KEV) catalog - single JSON file
2. ICS-CERT Advisories - JSON/structured format

KEV catalog: https://www.cisa.gov/known-exploited-vulnerabilities-catalog
(Download: known_exploited_vulnerabilities.json)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from .models import CTIDocument, CTISourceType, SeverityLevel

logger = logging.getLogger(__name__)


def parse_cisa_kev(filepath: Path) -> list[CTIDocument]:
    """
    Parse CISA Known Exploited Vulnerabilities catalog.

    Each KEV entry becomes one CTIDocument. The KEV catalog provides
    critical context that NVD alone doesn't: confirmed active exploitation
    and required remediation dates.

    Args:
        filepath: Path to known_exploited_vulnerabilities.json
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    vulnerabilities = data.get("vulnerabilities", [])

    for vuln in vulnerabilities:
        cve_id = vuln.get("cveID", "")
        if not cve_id:
            continue

        vendor = vuln.get("vendorProject", "Unknown")
        product = vuln.get("product", "Unknown")
        name = vuln.get("vulnerabilityName", "")
        description = vuln.get("shortDescription", "")
        action = vuln.get("requiredAction", "")
        date_added = vuln.get("dateAdded", "")
        due_date = vuln.get("dueDate", "")
        known_ransomware = vuln.get("knownRansomwareCampaignUse", "Unknown")

        # Build rich content text
        content_parts = [
            f"Vulnerability: {name}",
            f"Vendor/Product: {vendor} {product}",
            f"Description: {description}",
        ]
        if action:
            content_parts.append(f"Required Action: {action}")
        if known_ransomware and known_ransomware.lower() == "known":
            content_parts.append("WARNING: Known use in ransomware campaigns.")
        if due_date:
            content_parts.append(f"Remediation Due Date: {due_date}")

        content = "\n".join(content_parts)

        pub_date = None
        if date_added:
            try:
                pub_date = datetime.strptime(date_added, "%Y-%m-%d")
            except ValueError:
                pass

        doc = CTIDocument(
            doc_id=f"cisa_kev_{cve_id}",
            source=CTISourceType.CISA_KEV,
            title=f"CISA KEV: {cve_id} - {name}",
            content=content,
            severity=SeverityLevel.CRITICAL,  # All KEV entries are critical by definition
            cve_ids=[cve_id],
            affected_products=[f"{vendor} {product}"],
            published_date=pub_date,
            metadata={
                "required_action": action,
                "due_date": due_date,
                "known_ransomware": known_ransomware,
                "date_added_to_kev": date_added,
            },
        )
        documents.append(doc)

    logger.info(f"Parsed {len(documents)} KEV entries from {filepath.name}")
    return documents


def parse_cisa_advisory(filepath: Path) -> list[CTIDocument]:
    """
    Parse a single CISA advisory JSON file.

    Advisories are split by section (summary, technical details, mitigations)
    to create appropriately-sized chunks. The advisory title is prepended
    to each section for context preservation.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    advisory_id = data.get("id", filepath.stem)
    title = data.get("title", advisory_id)
    cve_ids = data.get("cve_ids", [])
    published = data.get("published", "")
    last_updated = data.get("last_updated", "")
    raw_metadata = data.get("metadata", {})

    pub_date = None
    if published:
        try:
            pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            pass

    mod_date = None
    if last_updated:
        try:
            mod_date = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
        except ValueError:
            pass

    # Process sections as individual chunks
    sections = data.get("sections", {})
    if not sections:
        # Fallback: treat entire body as one chunk
        body = data.get("body", data.get("content", ""))
        if body:
            sections = {"full_advisory": body}

    for section_name, section_text in sections.items():
        if not section_text or len(section_text.strip()) < 50:
            continue

        doc = CTIDocument(
            doc_id=f"cisa_advisory_{advisory_id}_{section_name}",
            source=CTISourceType.CISA_ADVISORY,
            title=f"CISA Advisory {advisory_id}: {title} [{section_name}]",
            content=f"Advisory: {title}\nSection: {section_name}\n\n{section_text}",
            cve_ids=cve_ids,
            published_date=pub_date,
            modified_date=mod_date,
            metadata={
                **raw_metadata,
                "advisory_id": advisory_id,
                "section": section_name,
            },
        )
        documents.append(doc)

    if not documents:
        logger.debug(f"No valid sections in advisory {advisory_id}")

    return documents


def parse_cisa_advisories_directory(directory: Path) -> list[CTIDocument]:
    """Parse all CISA advisory JSON files in a directory."""
    all_docs = []
    json_files = sorted(directory.glob("*.json"))

    for filepath in json_files:
        try:
            docs = parse_cisa_advisory(filepath)
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"Error parsing advisory {filepath.name}: {e}")

    logger.info(f"Total CISA advisory documents: {len(all_docs)}")
    return all_docs
