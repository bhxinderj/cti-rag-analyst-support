import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.cti_rag.ingestion.cisa_parser import parse_cisa_advisory
from src.cti_rag.ingestion.models import CTIDocument, CTISourceType, IOCEntry


class TestCTIRepresentation(unittest.TestCase):
    def test_parse_cisa_advisory_extracts_cti_signals_and_skips_noise_sections(self):
        with TemporaryDirectory() as tmp_dir:
            advisory_file = Path(tmp_dir) / "aa26-002a.json"
            advisory_file.write_text(
                json.dumps(
                    {
                        "id": "AA26-002A",
                        "title": "Cisco IOS XE Software Web UI Privilege Escalation Vulnerability",
                        "published": "2026-03-20T12:00:00+00:00",
                        "last_updated": "2026-03-27T18:30:00+00:00",
                        "sections": {
                            "summary": (
                                "This critical vulnerability affects Cisco IOS XE Software Web UI. "
                                "CVE-2023-20273 maps to Exploit Public-Facing Application [T1190]. "
                                "Operators observed 198.51.100.10 contacting vpn.example.org and "
                                "retrieving https://example.org/payload."
                            ),
                            "works_cited": "https://example.org/reference " * 10,
                        },
                        "metadata": {
                            "source_url": "https://example.test/advisory",
                            "archive_year": 2026,
                        },
                    }
                ),
                encoding="utf-8",
            )

            docs = parse_cisa_advisory(advisory_file)

        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc.metadata["section"], "summary")
        self.assertEqual(doc.severity.value, "critical")
        self.assertEqual(doc.cve_ids, ["CVE-2023-20273"])
        self.assertEqual(doc.attack_techniques, ["T1190"])
        self.assertIn("Cisco IOS XE Software Web UI", doc.affected_products)
        self.assertEqual(
            [(ioc.type, ioc.value) for ioc in doc.iocs[:3]],
            [
                ("url", "https://example.org/payload"),
                ("ip-dst", "198.51.100.10"),
                ("domain", "vpn.example.org"),
            ],
        )
        self.assertIn("Affected Products: Cisco IOS XE Software Web UI", doc.content)
        self.assertNotIn("works_cited", doc.doc_id)

    def test_embedding_text_includes_structured_iocs(self):
        doc = CTIDocument(
            doc_id="cisa_advisory_test_summary",
            source=CTISourceType.CISA_ADVISORY,
            title="Example advisory",
            content="Example content",
            cve_ids=["CVE-2026-0001"],
            attack_techniques=["T1190"],
            iocs=[IOCEntry(type="domain", value="example.org")],
            affected_products=["Example Product"],
        )

        text = doc.to_embedding_text()

        self.assertIn("IOCs: domain: example.org", text)
        self.assertIn("Affected Products: Example Product", text)

    def test_signal_metadata_scores_richer_cti_documents_higher(self):
        rich_doc = CTIDocument(
            doc_id="misp_event_rich",
            source=CTISourceType.MISP,
            title="Rich event",
            content="Example content",
            severity="high",
            cve_ids=["CVE-2026-0001"],
            attack_techniques=["T1190"],
            iocs=[
                IOCEntry(type="domain", value="example.org"),
                IOCEntry(type="url", value="https://example.org"),
                IOCEntry(type="ip-dst", value="198.51.100.10"),
            ],
            affected_products=["Example Product"],
        )
        weak_doc = CTIDocument(
            doc_id="misp_event_weak",
            source=CTISourceType.MISP,
            title="Weak event",
            content="Example content",
            iocs=[IOCEntry(type="ip-src", value="203.0.113.10")],
        )

        rich_meta = rich_doc.to_signal_metadata()
        weak_meta = weak_doc.to_signal_metadata()

        self.assertEqual(rich_meta["cti_signal_score"], 6.0)
        self.assertEqual(weak_meta["cti_signal_score"], 0.5)
        self.assertTrue(rich_meta["has_cve"])
        self.assertFalse(weak_meta["has_cve"])


if __name__ == "__main__":
    unittest.main()
