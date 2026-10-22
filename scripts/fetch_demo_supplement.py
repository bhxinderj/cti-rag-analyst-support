#!/usr/bin/env python3
"""
Fetch the demo supplement: a handful of post-snapshot documents that the
demo setup (CTI_RAG_SETUP=demo) appends to a clone of the Setup-B indexes.

The evaluated raw snapshot in data/raw is never touched. Everything lands in
data.demo.supplement_dir (default data/raw_demo) using the same file layout
and formats as data/raw, so the regular parsers ingest it unchanged:

  <supplement_dir>/nvd/nvd_cves_supplement.json       NVD API 2.0 response shape
  <supplement_dir>/cisa_advisories/<id>.json          parse_cisa_advisory() shape
  <supplement_dir>/{nvd,cisa_advisories}/manifest.json  sha256 + fetch date

Defaults fetch the "php2shell" pair (CVE-2026-109618, CVE-2026-109619) and
CISA's alert of 2026-10-21. Other CVEs/alerts can be passed on the CLI.

Usage:
    .venv/bin/python scripts/fetch_demo_supplement.py
    .venv/bin/python scripts/fetch_demo_supplement.py --cve CVE-2026-109618 --cve CVE-2026-109619 \
        --alert-url https://www.cisa.gov/news-events/alerts/2026/10/21/critical-vulnerabilities-php-require-immediate-patching \
        --alert-id CISA-ALERT-2026-10-21-PHP2SHELL
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cti_rag.ingestion import downloader as dl  # noqa: E402
from src.cti_rag.utils.config import get_project_root, load_config  # noqa: E402

logger = logging.getLogger("fetch_demo_supplement")

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
DEFAULT_CVES = ["CVE-2026-109618", "CVE-2026-109619"]
DEFAULT_ALERT_URL = (
    "https://www.cisa.gov/news-events/alerts/2026/10/21/"
    "critical-vulnerabilities-php-require-immediate-patching"
)
DEFAULT_ALERT_ID = "CISA-ALERT-2026-10-21-PHP2SHELL"
_ACTIONS_PARAGRAPH_RE = re.compile(
    r"<p>[^<]*following\s+(actions|steps|mitigations)[^<]*</p>", re.IGNORECASE
)


def fetch_nvd_cves(cve_ids: list[str], api_key: str | None) -> list[dict]:
    """Fetch each CVE record from the NVD API (one request per CVE)."""
    headers = {"apiKey": api_key} if api_key else {}
    # Public rate limit without a key: 5 requests per rolling 30 seconds.
    delay = 0.6 if api_key else 6.5
    records: list[dict] = []
    for i, cve_id in enumerate(cve_ids):
        if i:
            time.sleep(delay)
        logger.info("NVD: fetching %s", cve_id)
        resp = requests.get(NVD_API_URL, params={"cveId": cve_id}, headers=headers, timeout=30)
        resp.raise_for_status()
        vulns = resp.json().get("vulnerabilities", [])
        if not vulns:
            logger.warning("NVD: %s not found", cve_id)
            continue
        records.extend(vulns)
    return records


def write_nvd_supplement(records: list[dict], out_dir: Path, snapshot_date: str | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "nvd_cves_supplement.json"
    payload = {
        "resultsPerPage": len(records),
        "totalResults": len(records),
        "vulnerabilities": records,
        "download_date": datetime.now().isoformat(),
    }
    if snapshot_date:
        payload["snapshot_date"] = snapshot_date
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    dl._save_manifest(
        out_dir,
        [
            {
                "filename": out_file.name,
                "cve_ids": [v["cve"]["id"] for v in records],
                "cve_count": len(records),
                "sha256": dl._sha256_file(out_file),
            }
        ],
        snapshot_date=snapshot_date,
        extra_metadata={"kind": "demo_supplement", "source": NVD_API_URL},
    )
    return out_file


def alert_sections(body_html: str) -> dict[str, str]:
    """
    Split a CISA *alert* body into sections.

    Alerts (unlike AA advisories) carry their substance before the first
    heading, which the advisory section splitter would discard. The preamble
    is split into "summary" and "mitigations" at the "...following actions:"
    paragraph when present; headed sections follow as usual, minus the
    boilerplate disclaimer.
    """
    normalized = dl.html.unescape(body_html)
    # The extracted <main> block starts with the page header (type label,
    # h1 title, release-date field). The alert text begins at the first <p>.
    first_paragraph = re.search(r"<p[\s>]", normalized, re.IGNORECASE)
    if first_paragraph:
        normalized = normalized[first_paragraph.start():]
    first_heading = re.search(r"<h[2-6][^>]*>", normalized, re.IGNORECASE)
    preamble = normalized[: first_heading.start()] if first_heading else normalized
    rest = normalized[first_heading.start():] if first_heading else ""

    sections: dict[str, str] = {}
    actions = _ACTIONS_PARAGRAPH_RE.search(preamble)
    if actions:
        summary = dl._strip_html(preamble[: actions.start()])
        mitigations = dl._strip_html(preamble[actions.start():])
        if summary:
            sections["summary"] = summary
        if mitigations:
            sections["mitigations"] = mitigations
    else:
        summary = dl._strip_html(preamble)
        if summary:
            sections["summary"] = summary

    for key, text in dl._extract_html_sections(rest).items() if rest else []:
        if key in {"disclaimer", "full_advisory"}:
            continue
        sections[key] = text
    return sections


def add_section_context(sections: dict[str, str], title: str, cve_ids: list[str]) -> dict[str, str]:
    """
    Prefix sections that do not name a CVE with a one-line context header.

    AA advisories repeat their CVE ids in most sections; alerts name them
    once in the opening paragraph. Without the header, the "mitigations"
    chunk of an alert carries no CVE token, so an entity-centric follow-up
    ("How do I mitigate CVE-2026-109618?") cannot reach it lexically and the
    parser cannot attach the section-level CVE list either.
    """
    if not cve_ids:
        return sections
    header = f"Context: {title} ({', '.join(cve_ids)})."
    out: dict[str, str] = {}
    for key, text in sections.items():
        if dl._extract_cve_ids_from_text(text):
            out[key] = text
        else:
            out[key] = f"{header}\n\n{text}"
    return out


def fetch_cisa_alert(url: str, alert_id: str) -> dict:
    logger.info("CISA: fetching %s", url)
    page_html = dl._fetch_cisa_url_text(url)
    release_dt = dl._extract_cisa_temporal_field(page_html, "field-release-date")
    last_updated_dt = dl._extract_cisa_temporal_field(page_html, "field-last-updated") or release_dt
    if release_dt is None:
        raise RuntimeError(f"Could not parse the release date of {url}")
    body_html = dl._extract_cisa_body_html(page_html)
    title = dl._extract_cisa_page_title(page_html)
    cve_ids = dl._extract_cve_ids_from_text(body_html)
    sections = add_section_context(alert_sections(body_html), title, cve_ids)
    if not sections:
        raise RuntimeError(f"No sections extracted from {url}")
    return {
        "id": alert_id,
        "title": title,
        "published": release_dt.isoformat(),
        "last_updated": last_updated_dt.isoformat() if last_updated_dt else "",
        "cve_ids": cve_ids,
        "sections": sections,
        "metadata": {
            "source_url": url,
            "document_type": "alert",
            "release_date": release_dt.isoformat(),
            "last_updated": last_updated_dt.isoformat() if last_updated_dt else "",
        },
    }


def write_alert(alert_doc: dict, out_dir: Path, snapshot_date: str | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{alert_doc['id'].lower()}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(alert_doc, f, indent=2)
    dl._save_manifest(
        out_dir,
        [
            {
                "filename": out_file.name,
                "advisory_id": alert_doc["id"],
                "published": alert_doc["published"],
                "source_url": alert_doc["metadata"]["source_url"],
                "sha256": dl._sha256_file(out_file),
            }
        ],
        snapshot_date=snapshot_date,
        extra_metadata={"kind": "demo_supplement"},
    )
    return out_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cve", action="append", dest="cves", help="CVE id to fetch (repeatable)")
    parser.add_argument("--alert-url", default=DEFAULT_ALERT_URL, help="CISA alert page URL ('' to skip)")
    parser.add_argument("--alert-id", default=DEFAULT_ALERT_ID, help="Identifier used for the alert file/doc ids")
    parser.add_argument("--nvd-api-key", default=None)
    parser.add_argument("--supplement-dir", default=None, help="Override data.demo.supplement_dir")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    config = load_config(setup="demo")
    root = get_project_root()
    supplement_dir = root / (args.supplement_dir or config["data"]["supplement_dir"])
    snapshot_date = config["data"].get("snapshot_date")
    cve_ids = args.cves or DEFAULT_CVES

    print(f"Supplement dir: {supplement_dir}")
    print(f"Demo snapshot_date: {snapshot_date}")

    records = fetch_nvd_cves(cve_ids, args.nvd_api_key)
    nvd_file = write_nvd_supplement(records, supplement_dir / "nvd", snapshot_date)
    print(f"  NVD: {len(records)} record(s) -> {nvd_file}")
    for record in records:
        cve = record["cve"]
        metrics = cve.get("metrics", {})
        keys = [k for k in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV40") if metrics.get(k)]
        print(f"    {cve['id']}  published={cve.get('published')}  metrics={keys or 'none'}")

    if args.alert_url:
        alert_doc = fetch_cisa_alert(args.alert_url, args.alert_id)
        alert_file = write_alert(alert_doc, supplement_dir / "cisa_advisories", snapshot_date)
        print(f"  CISA alert: {alert_doc['title']!r} -> {alert_file}")
        print(f"    published={alert_doc['published']}  cves={alert_doc['cve_ids']}")
        print(f"    sections={ {k: len(v) for k, v in alert_doc['sections'].items()} }")

    print("\nNext: CTI_RAG_SETUP=demo .venv/bin/python main.py index-supplement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
