"""
Unified CTI Document Schema.

All data sources (NVD, CISA KEV, CISA Advisories, MISP) are normalized
into this common Pydantic model before chunking and indexing.

Design rationale:
- Single schema enables cross-source correlation queries
- `source` field allows metadata filtering in ChromaDB
- `cve_ids` field enables foreign-key-style joins across sources
- `attack_techniques` captures MITRE ATT&CK mappings from MISP galaxies
- `content` is the primary text field used for embedding generation
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CTISourceType(str, Enum):
    """Enumeration of supported CTI data sources."""
    NVD = "nvd"
    CISA_KEV = "cisa_kev"
    CISA_ADVISORY = "cisa_advisory"
    MISP = "misp"


class SeverityLevel(str, Enum):
    """Standardized severity levels across sources."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class IOCEntry(BaseModel):
    """Individual Indicator of Compromise."""
    type: str = Field(..., description="IOC type: ip, domain, hash-md5, hash-sha256, url, email, etc.")
    value: str = Field(..., description="The actual indicator value")
    category: Optional[str] = Field(None, description="MISP category if available")


class CTIDocument(BaseModel):
    """
    Unified document model for all CTI sources.

    Each instance represents one retrievable unit (chunk) in the RAG pipeline.
    For CVEs: one CVE = one document.
    For advisories: one section = one document.
    For MISP events: one event = one document.
    """
    doc_id: str = Field(..., description="Unique identifier: {source}_{original_id}")
    source: CTISourceType = Field(..., description="Origin data source")
    title: str = Field(..., description="CVE-ID, advisory title, or MISP event info")
    content: str = Field(..., description="Primary text for embedding and retrieval")

    # Severity & scoring
    severity: SeverityLevel = Field(default=SeverityLevel.UNKNOWN)
    cvss_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    cvss_vector: Optional[str] = Field(None)

    # Cross-references
    cve_ids: list[str] = Field(default_factory=list, description="Associated CVE identifiers")
    cwe_ids: list[str] = Field(default_factory=list, description="Associated CWE identifiers")
    attack_techniques: list[str] = Field(default_factory=list, description="MITRE ATT&CK technique IDs")

    # Indicators
    iocs: list[IOCEntry] = Field(default_factory=list)

    # Affected products
    affected_products: list[str] = Field(default_factory=list, description="CPE strings or product names")

    # Temporal
    published_date: Optional[datetime] = Field(None)
    modified_date: Optional[datetime] = Field(None)

    # Source-specific metadata (flexible)
    metadata: dict = Field(default_factory=dict, description="Additional source-specific fields")

    def to_embedding_text(self) -> str:
        """
        Generate the text representation used for embedding.

        Includes title in the content to ensure the embedding captures
        document identity (e.g., CVE-ID), improving retrieval precision
        for identifier-based queries.
        """
        parts = [f"Title: {self.title}"]

        if self.severity != SeverityLevel.UNKNOWN:
            parts.append(f"Severity: {self.severity.value.upper()}")

        if self.cvss_score is not None:
            parts.append(f"CVSS: {self.cvss_score}")

        parts.append(f"Content: {self.content}")

        if self.cve_ids:
            parts.append(f"Related CVEs: {', '.join(self.cve_ids)}")

        if self.attack_techniques:
            parts.append(f"ATT&CK Techniques: {', '.join(self.attack_techniques)}")

        if self.affected_products:
            parts.append(f"Affected Products: {', '.join(self.affected_products[:10])}")

        return "\n".join(parts)

    def to_chromadb_metadata(self) -> dict:
        """
        Extract flat metadata dict for ChromaDB storage.
        ChromaDB metadata values must be str, int, float, or bool.
        """
        meta = {
            "source": self.source.value,
            "title": self.title,
            "severity": self.severity.value,
            "published_date": self.published_date.isoformat() if self.published_date else "",
            "modified_date": self.modified_date.isoformat() if self.modified_date else "",
        }

        if self.cvss_score is not None:
            meta["cvss_score"] = self.cvss_score

        if self.cve_ids:
            meta["cve_ids"] = ",".join(self.cve_ids)

        if self.cwe_ids:
            meta["cwe_ids"] = ",".join(self.cwe_ids)

        if self.attack_techniques:
            meta["attack_techniques"] = ",".join(self.attack_techniques)

        if self.affected_products:
            meta["affected_products"] = ",".join(self.affected_products[:5])

        return meta
