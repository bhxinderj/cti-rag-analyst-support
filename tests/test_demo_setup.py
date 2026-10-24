"""
Demo setup (CTI_RAG_SETUP=demo): config derivation, CVSS v4.0 fallback,
BM25 append merge, supplement parsing and the index-supplement command.

None of this touches the evaluated setups; the tests assert that the demo
paths are disjoint from Setup B and that the v4.0 fallback never overrides
an existing v3.x score.
"""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from src.cti_rag.ingestion.models import CTIDocument, CTISourceType
from src.cti_rag.ingestion.nvd_parser import _extract_cvss, parse_nvd_file
from src.cti_rag.retrieval.indexer import _merge_bm25_artifact, _tokenize_cti
from src.cti_rag.utils import config as config_module


def _metric(version_key: str, score: float, metric_type: str = "Primary") -> dict:
    return {
        version_key: [
            {
                "type": metric_type,
                "cvssData": {"baseScore": score, "vectorString": f"CVSS:{version_key[-2]}.{version_key[-1]}/AV:N"},
            }
        ]
    }


def _nvd_payload(cve_id: str, metrics: dict, published: str = "2026-10-21T14:17:09.357") -> dict:
    return {
        "vulnerabilities": [
            {
                "cve": {
                    "id": cve_id,
                    "published": published,
                    "lastModified": published,
                    "descriptions": [{"lang": "en", "value": "PHP remote code execution via stale global state."}],
                    "metrics": metrics,
                    "weaknesses": [{"description": [{"lang": "en", "value": "CWE-416"}]}],
                    "configurations": [],
                    "references": [],
                }
            }
        ]
    }


class TestCvssFallback(unittest.TestCase):
    def test_v31_preferred_over_v40(self):
        metrics = {**_metric("cvssMetricV31", 7.5), **_metric("cvssMetricV40", 9.2)}
        score, vector = _extract_cvss(metrics)
        self.assertEqual(score, 7.5)
        self.assertTrue(vector.startswith("CVSS:3.1"))

    def test_v30_preferred_over_v40(self):
        metrics = {**_metric("cvssMetricV30", 8.1), **_metric("cvssMetricV40", 9.2)}
        self.assertEqual(_extract_cvss(metrics)[0], 8.1)

    def test_v40_used_when_only_v40_present(self):
        score, vector = _extract_cvss(_metric("cvssMetricV40", 9.2, metric_type="Secondary"))
        self.assertEqual(score, 9.2)
        self.assertTrue(vector.startswith("CVSS:4.0"))

    def test_no_metrics_returns_none(self):
        self.assertEqual(_extract_cvss({}), (None, None))

    def test_parse_nvd_file_keeps_v40_only_cve_above_min_cvss(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nvd_cves_supplement.json"
            path.write_text(
                json.dumps(_nvd_payload("CVE-2026-109619", _metric("cvssMetricV40", 9.2, "Secondary"))),
                encoding="utf-8",
            )
            docs = parse_nvd_file(path, min_cvss=7.0)
        self.assertEqual([d.doc_id for d in docs], ["nvd_CVE-2026-109619"])
        self.assertEqual(docs[0].cvss_score, 9.2)
        self.assertEqual(docs[0].severity.value, "critical")


class TestDemoConfig(unittest.TestCase):
    def test_demo_setup_derives_disjoint_paths_and_snapshot(self):
        demo = config_module.load_config(setup="demo")
        base = config_module.load_config(setup="b")

        self.assertEqual(demo["data"]["active_setup"], "demo")
        self.assertEqual(demo["data"]["base_setup"], "b")
        self.assertTrue(demo["data"]["include_cisa_advisories"])
        self.assertEqual(demo["data"]["supplement_dir"], "data/raw_demo")
        self.assertEqual(demo["data"]["snapshot_date"], "2026-10-22")
        self.assertEqual(base["data"]["snapshot_date"], "2025-12-31")

        for getter in (
            lambda c: c["retrieval"]["bm25"]["index_path"],
            lambda c: c["chromadb"]["persist_directory"],
            lambda c: c["retrieval"]["vector"]["collection_name"],
            lambda c: c["data"]["processed_path"],
            lambda c: c["evaluation"]["results_dir"],
        ):
            self.assertNotEqual(getter(demo), getter(base))
            self.assertIn("setup_demo", getter(demo))

        # Raw source dirs are shared (read-only); the supplement lives elsewhere.
        self.assertEqual(demo["data"]["sources"]["nvd"]["raw_dir"], base["data"]["sources"]["nvd"]["raw_dir"])

    def test_env_variable_still_selects_setup(self):
        with patch.dict(os.environ, {"CTI_RAG_SETUP": "demo"}):
            self.assertEqual(config_module.load_config()["data"]["active_setup"], "demo")
        with patch.dict(os.environ, {"CTI_RAG_SETUP": "b"}):
            self.assertNotIn("supplement_dir", config_module.load_config()["data"])

    def test_unknown_setup_rejected(self):
        with self.assertRaises(ValueError):
            config_module.load_config(setup="c")


class TestBm25Merge(unittest.TestCase):
    def _doc(self, doc_id: str, content: str) -> CTIDocument:
        return CTIDocument(doc_id=doc_id, source=CTISourceType.NVD, title=doc_id, content=content)

    def test_append_keeps_base_order_and_replaces_duplicates(self):
        base_docs = [self._doc("nvd_A", "alpha"), self._doc("nvd_B", "beta")]
        existing = {
            "doc_ids": [d.doc_id for d in base_docs],
            "corpus_texts": [d.to_embedding_text() for d in base_docs],
            "tokenized_corpus": [_tokenize_cti(d.to_embedding_text()) for d in base_docs],
            "metadatas": [d.to_chromadb_metadata() for d in base_docs],
        }
        merged = _merge_bm25_artifact(existing, [self._doc("nvd_B", "beta updated"), self._doc("nvd_C", "gamma")])

        self.assertEqual(merged["doc_ids"], ["nvd_A", "nvd_B", "nvd_C"])
        self.assertIn("beta updated", merged["corpus_texts"][1])
        self.assertEqual(len(merged["tokenized_corpus"]), 3)
        self.assertEqual(len(merged["metadatas"]), 3)
        # Input artifact is not mutated.
        self.assertEqual(existing["doc_ids"], ["nvd_A", "nvd_B"])


class TestIndexSupplementCommand(unittest.TestCase):
    def _config(self, root: Path) -> dict:
        return {
            "data": {
                "active_setup": "demo",
                "base_setup": "b",
                "include_cisa_advisories": True,
                "snapshot_date": "2026-10-22",
                "supplement_dir": "data/raw_demo",
                "processed_path": "data/processed/setup_demo/all_documents.json",
                "sources": {
                    "nvd": {"raw_dir": "data/raw/nvd", "min_cvss": 7.0},
                    "cisa_kev": {"raw_dir": "data/raw/cisa_kev"},
                    "cisa_advisories": {"raw_dir": "data/raw/cisa_advisories"},
                    "misp": {"raw_dir": "data/raw/misp"},
                },
            },
            "retrieval": {"bm25": {"index_path": "data/indexes/setup_demo/bm25_index.json"}, "vector": {"collection_name": "cti_documents_setup_demo"}},
            "chromadb": {"persist_directory": "data/indexes/setup_demo/chromadb"},
        }

    def test_supplement_is_parsed_filtered_and_appended(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            nvd_dir = root / "data/raw_demo/nvd"
            nvd_dir.mkdir(parents=True)
            payload = _nvd_payload("CVE-2026-109618", _metric("cvssMetricV40", 9.2, "Secondary"))
            payload["vulnerabilities"].extend(
                _nvd_payload("CVE-2026-999999", _metric("cvssMetricV40", 9.9, "Secondary"), published="2026-11-01T00:00:00.000")["vulnerabilities"]
            )
            (nvd_dir / "nvd_cves_supplement.json").write_text(json.dumps(payload), encoding="utf-8")
            base_processed = root / "data/processed/setup_b/all_documents.json"
            base_processed.parent.mkdir(parents=True)
            base_processed.write_text(json.dumps([{"doc_id": "nvd_CVE-2024-3094"}]), encoding="utf-8")

            demo_config = self._config(root)
            base_config = {
                "data": {"processed_path": "data/processed/setup_b/all_documents.json"},
                "retrieval": {"bm25": {"index_path": "data/indexes/setup_b/bm25_index.json"}, "vector": {"collection_name": "cti_documents_setup_b"}},
                "chromadb": {"persist_directory": "data/indexes/setup_b/chromadb"},
            }
            fake_indexer = Mock()
            fake_indexer.get_stats.return_value = {"chromadb_count": 2, "bm25_exists": True, "bm25_count": 2}

            def fake_load_config(setup=None):
                return base_config if setup == "b" else demo_config

            with patch("main.load_config", side_effect=fake_load_config), patch(
                "main.get_project_root", return_value=root
            ), patch("src.cti_rag.retrieval.indexer.CTIIndexer", return_value=fake_indexer), patch(
                "src.cti_rag.retrieval.indexer.clone_setup_indexes", return_value=True
            ) as clone_patch:
                main.cmd_index_supplement(SimpleNamespace())

            clone_patch.assert_called_once()
            appended = fake_indexer.append_documents.call_args.args[0]
            # The post-snapshot CVE (published 2026-11-01) is dropped by the demo cutoff.
            self.assertEqual([d.doc_id for d in appended], ["nvd_CVE-2026-109618"])
            fake_indexer.index_documents.assert_not_called()
            fake_indexer.clear_indexes.assert_not_called()

            processed = json.loads((root / "data/processed/setup_demo/all_documents.json").read_text())
            self.assertEqual([d["doc_id"] for d in processed], ["nvd_CVE-2024-3094", "nvd_CVE-2026-109618"])

    def test_refuses_outside_demo_setup(self):
        config = {"data": {"active_setup": "b"}}
        with patch("main.load_config", return_value=config), self.assertRaises(SystemExit):
            main.cmd_index_supplement(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()


class TestSupplementAlertSections(unittest.TestCase):
    """The fetch script lives in scripts/; load it by path."""

    @classmethod
    def setUpClass(cls):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "fetch_demo_supplement", Path(__file__).resolve().parent.parent / "scripts" / "fetch_demo_supplement.py"
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_alert_preamble_is_split_into_summary_and_mitigations(self):
        body = (
            '<div class="c-page-title"><h1>Title</h1><time>October 21, 2026</time></div>'
            "<p>CISA urges patching of CVE-2026-109618 and CVE-2026-109619.</p>"
            "<p>CISA urges organizations to take the following actions:</p>"
            "<ul><li>Apply the patch immediately.</li><li>Monitor for suspicious activity.</li></ul>"
            "<h2>Disclaimer</h2><p>As is.</p>"
        )
        sections = self.mod.alert_sections(body)
        self.assertEqual(list(sections), ["summary", "mitigations"])
        self.assertNotIn("October 21", sections["summary"])
        self.assertIn("Apply the patch", sections["mitigations"])

    def test_sections_without_cve_get_context_header(self):
        sections = {"summary": "Patch CVE-2026-109618 now.", "mitigations": "Apply the patch."}
        out = self.mod.add_section_context(sections, "PHP alert", ["CVE-2026-109618", "CVE-2026-109619"])
        self.assertEqual(out["summary"], sections["summary"])
        self.assertTrue(out["mitigations"].startswith("Context: PHP alert (CVE-2026-109618, CVE-2026-109619)."))
