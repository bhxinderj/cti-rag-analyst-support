"""
NVD/CVE Data Parser.

Parses NVD CVE JSON feed files into unified CTIDocument models.
Expects NVD API 2.0 JSON format (bulk download).

Data source: https://nvd.nist.gov/developers/vulnerabilities
Download: Use nvd_downloader.py or manual bulk download.

Each CVE entry becomes exactly one CTIDocument (1 CVE = 1 chunk).
CVE descriptions are typically 100-500 tokens, which is already
the optimal chunk size for embedding.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from .models import CTIDocument, CTISourceType, SeverityLevel

logger = logging.getLogger(__name__)


def _parse_severity(cvss_score: float | None) -> SeverityLevel:
    """Map CVSS score to severity level per NVD specification."""
    if cvss_score is None:
        return SeverityLevel.UNKNOWN
    if cvss_score >= 9.0:
        return SeverityLevel.CRITICAL
    if cvss_score >= 7.0:
        return SeverityLevel.HIGH
    if cvss_score >= 4.0:
        return SeverityLevel.MEDIUM
    return SeverityLevel.LOW


def _extract_cvss_v31(metrics: dict) -> tuple[float | None, str | None]:
    """Extract CVSS v3.1 score and vector string."""
    cvss_v31 = metrics.get("cvssMetricV31", [])
    if not cvss_v31:
        cvss_v31 = metrics.get("cvssMetricV30", [])
    if not cvss_v31:
        return None, None

    # Use primary metric (usually from NVD)
    for metric in cvss_v31:
        if metric.get("type") == "Primary":
            data = metric.get("cvssData", {})
            return data.get("baseScore"), data.get("vectorString")

    # Fallback to first available
    data = cvss_v31[0].get("cvssData", {})
    return data.get("baseScore"), data.get("vectorString")


def _extract_cwe_ids(weaknesses: list) -> list[str]:
    """Extract CWE identifiers from NVD weakness data."""
    cwe_ids = []
    for weakness in weaknesses:
        for desc in weakness.get("description", []):
            value = desc.get("value", "")
            if value.startswith("CWE-"):
                cwe_ids.append(value)
    return cwe_ids


def _extract_affected_products(configurations: list) -> list[str]:
    """Extract CPE strings for affected products."""
    products = []
    for config in configurations:
        for node in config.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                if cpe_match.get("vulnerable", False):
                    cpe = cpe_match.get("criteria", "")
                    if cpe:
                        products.append(cpe)
    return products


def parse_nvd_file(filepath: Path, min_cvss: float = 7.0) -> list[CTIDocument]:
    """
    Parse a single NVD JSON file into CTIDocument list.

    Args:
        filepath: Path to NVD JSON file (API 2.0 format)
        min_cvss: Minimum CVSS score filter (default: 7.0 for HIGH+CRITICAL)

    Returns:
        List of CTIDocument instances
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    vulnerabilities = data.get("vulnerabilities", [])

    for vuln_wrapper in vulnerabilities:
        cve = vuln_wrapper.get("cve", {})
        cve_id = cve.get("id", "UNKNOWN")

        # Extract CVSS score
        metrics = cve.get("metrics", {})
        cvss_score, cvss_vector = _extract_cvss_v31(metrics)

        # Apply CVSS filter
        if min_cvss > 0 and (cvss_score is None or cvss_score < min_cvss):
            continue

        # Extract English description
        descriptions = cve.get("descriptions", [])
        description = ""
        for desc in descriptions:
            if desc.get("lang") == "en":
                description = desc.get("value", "")
                break

        if not description:
            logger.debug(f"Skipping {cve_id}: no English description")
            continue

        # Extract dates
        published = cve.get("published")
        modified = cve.get("lastModified")

        pub_date = datetime.fromisoformat(published.replace("Z", "+00:00")) if published else None
        mod_date = datetime.fromisoformat(modified.replace("Z", "+00:00")) if modified else None

        # Extract CWEs and affected products
        cwe_ids = _extract_cwe_ids(cve.get("weaknesses", []))
        affected = _extract_affected_products(cve.get("configurations", []))

        # Extract references
        refs = [ref.get("url", "") for ref in cve.get("references", [])[:5]]

        doc = CTIDocument(
            doc_id=f"nvd_{cve_id}",
            source=CTISourceType.NVD,
            title=cve_id,
            content=description,
            severity=_parse_severity(cvss_score),
            cvss_score=cvss_score,
            cvss_vector=cvss_vector,
            cve_ids=[cve_id],
            cwe_ids=cwe_ids,
            affected_products=affected,
            published_date=pub_date,
            modified_date=mod_date,
            metadata={"references": refs},
        )
        documents.append(doc)

    logger.info(f"Parsed {len(documents)} CVEs from {filepath.name} (CVSS >= {min_cvss})")
    return documents


def parse_nvd_directory(directory: Path, min_cvss: float = 7.0) -> list[CTIDocument]:
    """Parse all NVD JSON files in a directory."""
    all_docs = []
    json_files = sorted(directory.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {directory}")
        return all_docs

    for filepath in json_files:
        try:
            docs = parse_nvd_file(filepath, min_cvss=min_cvss)
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"Error parsing {filepath.name}: {e}")

    logger.info(f"Total NVD documents parsed: {len(all_docs)}")
    return all_docs
