import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from src.cti_rag.ingestion.models import CTIDocument, CTISourceType
from src.cti_rag.retrieval.hybrid_retriever import HybridRetriever
from src.cti_rag.retrieval.indexer import _read_bm25_artifact, _write_bm25_artifact


def _minimal_doc(doc_id: str, source: CTISourceType) -> CTIDocument:
    return CTIDocument(doc_id=doc_id, source=source, title=doc_id, content="example")


class TestSetupAwareIndexing(unittest.TestCase):
    def test_cmd_index_skips_cisa_advisories_for_setup_a(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "data/raw/nvd").mkdir(parents=True)
            (root / "data/raw/nvd/example.json").write_text("{}", encoding="utf-8")
            (root / "data/raw/cisa_kev").mkdir(parents=True)
            (root / "data/raw/cisa_kev/known_exploited_vulnerabilities.json").write_text("{}", encoding="utf-8")
            (root / "data/raw/cisa_advisories").mkdir(parents=True)
            (root / "data/raw/cisa_advisories/example.json").write_text("{}", encoding="utf-8")
            (root / "data/raw/misp").mkdir(parents=True)
            (root / "data/raw/misp/example.json").write_text("{}", encoding="utf-8")

            config = {
                "data": {
                    "active_setup": "a",
                    "include_cisa_advisories": False,
                    "processed_path": "data/processed/setup_a/all_documents.json",
                    "sources": {
                        "nvd": {"raw_dir": "data/raw/nvd", "min_cvss": 7.0},
                        "cisa_kev": {"raw_dir": "data/raw/cisa_kev"},
                        "cisa_advisories": {"raw_dir": "data/raw/cisa_advisories"},
                        "misp": {"raw_dir": "data/raw/misp"},
                    },
                }
            }
            fake_indexer = Mock()
            fake_indexer.get_stats.return_value = {"chromadb_count": 1, "bm25_exists": True}

            with patch("main.load_config", return_value=config), patch(
                "main.get_project_root", return_value=root
            ), patch(
                "src.cti_rag.ingestion.nvd_parser.parse_nvd_directory",
                return_value=[_minimal_doc("nvd_test", CTISourceType.NVD)],
            ), patch(
                "src.cti_rag.ingestion.cisa_parser.parse_cisa_kev",
                return_value=[_minimal_doc("cisa_kev_test", CTISourceType.CISA_KEV)],
            ), patch(
                "src.cti_rag.ingestion.cisa_parser.parse_cisa_advisories_directory",
                return_value=[_minimal_doc("cisa_advisory_test", CTISourceType.CISA_ADVISORY)],
            ) as advisory_patch, patch(
                "src.cti_rag.ingestion.misp_parser.parse_misp_directory",
                return_value=[_minimal_doc("misp_test", CTISourceType.MISP)],
            ), patch(
                "src.cti_rag.retrieval.indexer.CTIIndexer",
                return_value=fake_indexer,
            ):
                main.cmd_index(SimpleNamespace(clear=True))

            self.assertFalse(advisory_patch.called)
            fake_indexer.clear_indexes.assert_called_once()
            indexed_docs = fake_indexer.index_documents.call_args.args[0]
            self.assertEqual([doc.doc_id for doc in indexed_docs], ["nvd_test", "cisa_kev_test", "misp_test"])
            self.assertTrue((root / "data/processed/setup_a/all_documents.json").exists())

    def test_cmd_index_includes_cisa_advisories_for_setup_b(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "data/raw/nvd").mkdir(parents=True)
            (root / "data/raw/nvd/example.json").write_text("{}", encoding="utf-8")
            (root / "data/raw/cisa_kev").mkdir(parents=True)
            (root / "data/raw/cisa_kev/known_exploited_vulnerabilities.json").write_text("{}", encoding="utf-8")
            (root / "data/raw/cisa_advisories").mkdir(parents=True)
            (root / "data/raw/cisa_advisories/example.json").write_text("{}", encoding="utf-8")
            (root / "data/raw/misp").mkdir(parents=True)
            (root / "data/raw/misp/example.json").write_text("{}", encoding="utf-8")

            config = {
                "data": {
                    "active_setup": "b",
                    "include_cisa_advisories": True,
                    "processed_path": "data/processed/setup_b/all_documents.json",
                    "sources": {
                        "nvd": {"raw_dir": "data/raw/nvd", "min_cvss": 7.0},
                        "cisa_kev": {"raw_dir": "data/raw/cisa_kev"},
                        "cisa_advisories": {"raw_dir": "data/raw/cisa_advisories"},
                        "misp": {"raw_dir": "data/raw/misp"},
                    },
                }
            }
            fake_indexer = Mock()
            fake_indexer.get_stats.return_value = {"chromadb_count": 1, "bm25_exists": True}

            with patch("main.load_config", return_value=config), patch(
                "main.get_project_root", return_value=root
            ), patch(
                "src.cti_rag.ingestion.nvd_parser.parse_nvd_directory",
                return_value=[_minimal_doc("nvd_test", CTISourceType.NVD)],
            ), patch(
                "src.cti_rag.ingestion.cisa_parser.parse_cisa_kev",
                return_value=[],
            ), patch(
                "src.cti_rag.ingestion.cisa_parser.parse_cisa_advisories_directory",
                return_value=[_minimal_doc("cisa_advisory_test", CTISourceType.CISA_ADVISORY)],
            ) as advisory_patch, patch(
                "src.cti_rag.ingestion.misp_parser.parse_misp_directory",
                return_value=[],
            ), patch(
                "src.cti_rag.retrieval.indexer.CTIIndexer",
                return_value=fake_indexer,
            ):
                main.cmd_index(SimpleNamespace(clear=False))

            advisory_patch.assert_called_once()
            indexed_docs = fake_indexer.index_documents.call_args.args[0]
            self.assertEqual([doc.doc_id for doc in indexed_docs], ["nvd_test", "cisa_advisory_test"])
            self.assertTrue((root / "data/processed/setup_b/all_documents.json").exists())


class TestBM25Artifacts(unittest.TestCase):
    def test_bm25_artifact_round_trips_as_json(self):
        with TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "bm25_index.json"
            _write_bm25_artifact(
                artifact_path,
                doc_ids=["doc-1"],
                corpus_texts=["Title: Example\nContent: Example content"],
                tokenized_corpus=[["title", "example", "content", "example", "content"]],
                metadatas=[{"source": "nvd", "cti_signal_score": 2.0}],
            )

            loaded = _read_bm25_artifact(artifact_path)
            raw = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertEqual(raw["doc_ids"], ["doc-1"])
        self.assertEqual(loaded["metadatas"][0]["cti_signal_score"], 2.0)
        self.assertEqual(loaded["tokenized_corpus"][0][0], "title")

    def test_retriever_load_bm25_artifact_backfills_missing_tokenized_corpus(self):
        with TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "bm25_index.json"
            artifact_path.write_text(
                json.dumps(
                    {
                        "doc_ids": ["doc-1"],
                        "corpus_texts": ["Title: CVE-2024-3094\nContent: Example content"],
                        "metadatas": [{"source": "nvd"}],
                    }
                ),
                encoding="utf-8",
            )

            loaded = HybridRetriever._load_bm25_artifact(artifact_path)

        self.assertEqual(loaded["doc_ids"], ["doc-1"])
        self.assertIn("cve-2024-3094", loaded["tokenized_corpus"][0])


if __name__ == "__main__":
    unittest.main()
