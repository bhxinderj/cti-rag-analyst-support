import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from main import _filter_documents_by_snapshot
from src.cti_rag.ingestion.cisa_parser import parse_cisa_advisory
from src.cti_rag.ingestion.misp_parser import parse_misp_event
from src.cti_rag.ingestion.models import CTIDocument, CTISourceType


class TestSnapshotLogic(unittest.TestCase):
    def test_filter_documents_by_snapshot_drops_documents_modified_after_cutoff(self):
        stale_snapshot_doc = CTIDocument(
            doc_id="nvd_CVE-2024-0001",
            source=CTISourceType.NVD,
            title="CVE-2024-0001",
            content="example",
            published_date=datetime(2026, 3, 20, tzinfo=timezone.utc),
            modified_date=datetime(2026, 3, 29, tzinfo=timezone.utc),
        )
        valid_snapshot_doc = CTIDocument(
            doc_id="nvd_CVE-2024-0002",
            source=CTISourceType.NVD,
            title="CVE-2024-0002",
            content="example",
            published_date=datetime(2026, 3, 20, tzinfo=timezone.utc),
            modified_date=datetime(2026, 3, 28, tzinfo=timezone.utc),
        )

        kept, dropped = _filter_documents_by_snapshot(
            [stale_snapshot_doc, valid_snapshot_doc],
            "2026-03-28",
        )

        self.assertEqual([doc.doc_id for doc in kept], ["nvd_CVE-2024-0002"])
        self.assertEqual([doc.doc_id for doc in dropped], ["nvd_CVE-2024-0001"])

    def test_parse_cisa_advisory_separates_release_and_last_updated(self):
        with TemporaryDirectory() as tmp_dir:
            advisory_file = Path(tmp_dir) / "aa26-001a.json"
            advisory_file.write_text(
                json.dumps(
                    {
                        "id": "AA26-001A",
                        "title": "Example Advisory",
                        "published": "2026-03-20T12:00:00+00:00",
                        "last_updated": "2026-03-27T18:30:00+00:00",
                        "sections": {
                            "summary": "A" * 80,
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
        self.assertEqual(docs[0].published_date.isoformat(), "2026-03-20T12:00:00+00:00")
        self.assertEqual(docs[0].modified_date.isoformat(), "2026-03-27T18:30:00+00:00")
        self.assertEqual(docs[0].metadata["source_url"], "https://example.test/advisory")
        self.assertEqual(docs[0].metadata["section"], "summary")

    def test_parse_misp_event_uses_publish_timestamp_for_publication_and_timestamp_for_modification(self):
        event = {
            "id": "event-1",
            "info": "Example MISP event",
            "publish_timestamp": "1710892800",  # 2024-03-20T00:00:00Z
            "timestamp": "1711324800",  # 2024-03-25T00:00:00Z
            "Attribute": [],
        }

        doc = parse_misp_event(event)

        self.assertIsNotNone(doc)
        self.assertEqual(doc.published_date.isoformat(), "2024-03-20T00:00:00+00:00")
        self.assertEqual(doc.modified_date.isoformat(), "2024-03-25T00:00:00+00:00")
        self.assertEqual(doc.metadata["publish_timestamp"], "1710892800")
        self.assertEqual(doc.metadata["timestamp"], "1711324800")


if __name__ == "__main__":
    unittest.main()
