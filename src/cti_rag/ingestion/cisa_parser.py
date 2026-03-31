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
import re
from datetime import datetime
from pathlib import Path

from .models import CTIDocument, CTISourceType, IOCEntry, SeverityLevel

logger = logging.getLogger(__name__)

_ATTACK_TECHNIQUE_ID_RE = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)
_CVE_ID_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"]+", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_DOMAIN_RE = re.compile(r"\b(?:(?:[a-z0-9-]+\.)+[a-z]{2,})(?:/[^\s]*)?\b", re.IGNORECASE)
_SHA256_RE = re.compile(r"\b[a-f0-9]{64}\b", re.IGNORECASE)
_SHA1_RE = re.compile(r"\b[a-f0-9]{40}\b", re.IGNORECASE)
_MD5_RE = re.compile(r"\b[a-f0-9]{32}\b", re.IGNORECASE)
_NOISY_SECTION_NAMES = {
    "contact",
    "disclaimer_of_endorsement",
    "purpose",
    "works_cited",
    "references",
}
_PRODUCT_SUFFIX_RE = re.compile(
    r"\s+(?:"
    r"remote code execution|"
    r"authentication bypass|"
    r"privilege escalation|"
    r"command injection|"
    r"sql injection|"
    r"information disclosure|"
    r"arbitrary file upload|"
    r"arbitrary file read|"
    r"directory traversal|"
    r"path traversal|"
    r"denial of service|"
    r"cross-site scripting|"
    r"vulnerabilit(?:y|ies)"
    r")\b.*$",
    re.IGNORECASE,
)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _extract_attack_techniques(text: str) -> list[str]:
    return _dedupe_preserve_order([match.upper() for match in _ATTACK_TECHNIQUE_ID_RE.findall(text or "")])


def _extract_cve_ids(text: str) -> list[str]:
    return _dedupe_preserve_order([match.upper() for match in _CVE_ID_RE.findall(text or "")])


def _extract_iocs(text: str) -> list[IOCEntry]:
    iocs = []
    seen = set()

    def add_ioc(ioc_type: str, value: str):
        key = (ioc_type, value.lower())
        if value and key not in seen:
            seen.add(key)
            iocs.append(IOCEntry(type=ioc_type, value=value))

    for match in _URL_RE.findall(text or ""):
        add_ioc("url", match.rstrip(".,);]"))

    masked_text = _URL_RE.sub(" ", text or "")
    for match in _SHA256_RE.findall(masked_text):
        add_ioc("sha256", match)
    for match in _SHA1_RE.findall(masked_text):
        add_ioc("sha1", match)
    for match in _MD5_RE.findall(masked_text):
        add_ioc("md5", match)
    for match in _IPV4_RE.findall(masked_text):
        add_ioc("ip-dst", match)
    for match in _DOMAIN_RE.findall(masked_text):
        lowered = match.lower()
        if lowered.startswith("http"):
            continue
        if lowered.count(".") < 1:
            continue
        add_ioc("domain", match.rstrip(".,);]"))

    return iocs[:15]


def _extract_products(title: str, section_text: str) -> list[str]:
    products = []

    title_candidate = title or ""
    if "vulnerabil" in title_candidate.lower():
        title_candidate = _PRODUCT_SUFFIX_RE.sub("", title_candidate).strip(" -:;,.'\"")
        if title_candidate and len(title_candidate) > 4:
            products.append(title_candidate)

    text = " ".join((section_text or "").split())
    for match in re.finditer(
        r"\bCVE-\d{4}-\d{4,}\b[^.]{0,160}?\bin\s+([A-Z][A-Za-z0-9.+/_&() -]{3,80})",
        text,
        flags=re.IGNORECASE,
    ):
        candidate = _PRODUCT_SUFFIX_RE.sub("", match.group(1)).strip(" -:;,.'\"")
        if candidate and len(candidate) > 4:
            products.append(candidate)

    cleaned = []
    seen = set()
    for product in products:
        normalized = product.lower()
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(product)
    return cleaned[:5]


def _infer_severity(text: str) -> SeverityLevel:
    matches = set(
        level.lower()
        for level in re.findall(
            r"\b(critical|high|medium|low)(?:[- ]severity)?\s+vulnerab",
            text or "",
            flags=re.IGNORECASE,
        )
    )
    if len(matches) != 1:
        return SeverityLevel.UNKNOWN

    level = matches.pop()
    return SeverityLevel(level)


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
        if section_name.lower() in _NOISY_SECTION_NAMES:
            continue
        if not section_text or len(section_text.strip()) < 50:
            continue

        combined_text = f"{title}\n{section_text}"
        section_cve_ids = _extract_cve_ids(combined_text)
        attack_techniques = _extract_attack_techniques(combined_text)
        iocs = _extract_iocs(section_text)
        affected_products = _extract_products(title, section_text)
        severity = _infer_severity(combined_text)

        content_parts = [
            f"Advisory: {title}",
            f"Section: {section_name}",
        ]
        if severity != SeverityLevel.UNKNOWN:
            content_parts.append(f"Severity: {severity.value.upper()}")
        if section_cve_ids:
            content_parts.append(f"Related CVEs: {', '.join(section_cve_ids)}")
        if attack_techniques:
            content_parts.append(f"ATT&CK Techniques: {', '.join(attack_techniques)}")
        if affected_products:
            content_parts.append(f"Affected Products: {', '.join(affected_products)}")
        if iocs:
            content_parts.append(
                "Key IOCs: " + "; ".join(f"{ioc.type}: {ioc.value}" for ioc in iocs[:8])
            )
        content_parts.append("")
        content_parts.append(section_text)

        doc = CTIDocument(
            doc_id=f"cisa_advisory_{advisory_id}_{section_name}",
            source=CTISourceType.CISA_ADVISORY,
            title=f"CISA Advisory {advisory_id}: {title} [{section_name}]",
            content="\n".join(content_parts),
            severity=severity,
            cve_ids=section_cve_ids or cve_ids,
            attack_techniques=attack_techniques,
            iocs=iocs,
            affected_products=affected_products,
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
