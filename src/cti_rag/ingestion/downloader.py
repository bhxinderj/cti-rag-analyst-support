"""
CTI Data Downloader.

Downloads static snapshots of CTI data sources for reproducible evaluation.
All downloads are timestamped and checksummed.

Usage:
    python -m src.ingestion.downloader --all
    python -m src.ingestion.downloader --source nvd
"""

import hashlib
import html
import json
import logging
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm

from ..utils.config import load_config, get_project_root

logger = logging.getLogger(__name__)

# NVD API 2.0 endpoint
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# CISA KEV catalog
CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# CISA cybersecurity advisories RSS feed
CISA_ADVISORIES_FEED_URL = "https://www.cisa.gov/cybersecurity-advisories/all.xml"
CISA_ADVISORY_ARCHIVE_URL = (
    "https://www.cisa.gov/news-events/cybersecurity-advisories"
    "?f%5B0%5D=advisory_type%3A94&f%5B1%5D=release_date_year%3A{year}&items_per_page=All"
)
CISA_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CTI-RAG-Prototype/1.0)"}
CISA_ADVISORY_START_HEADING_RE = re.compile(
    r"<h([2-4])[^>]*>\s*(?:<strong>)?\s*"
    r"(executive\s+summary|summary|background|technical\s+details|mitigations?|"
    r"recommendations?|overview|initial\s+access|persistence|detection|contact\s+information|"
    r"appendix\s+[a-z0-9:\-\s&]+)"
    r"\s*(?:</strong>)?\s*</h\1>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _strip_html(raw_html: str) -> str:
    """Convert simple HTML fragments into plain text."""
    text = re.sub(r"<\s*br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"</\s*(p|div|li|ul|ol|table|tr|h2|h3|h4|h5|h6)\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_html_sections(raw_html: str) -> dict[str, str]:
    """Split advisory HTML into coarse sections based on headings."""
    normalized_html = html.unescape(raw_html)
    heading_pattern = re.compile(r"<h([2-6])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
    matches = list(heading_pattern.finditer(normalized_html))

    if not matches:
        body = _strip_html(normalized_html)
        return {"full_advisory": body} if body else {}

    sections = {}
    for idx, match in enumerate(matches):
        heading = _strip_html(match.group(2)).lower()
        section_key = re.sub(r"[^a-z0-9]+", "_", heading).strip("_") or f"section_{idx + 1}"
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized_html)
        section_text = _strip_html(normalized_html[start:end])
        if section_text:
            sections[section_key] = section_text

    if not sections:
        body = _strip_html(normalized_html)
        return {"full_advisory": body} if body else {}

    return sections


def _extract_cve_ids_from_text(text: str) -> list[str]:
    """Extract CVE identifiers from advisory HTML/text."""
    return sorted(set(re.findall(r"\bCVE-\d{4}-\d{4,}\b", text, flags=re.IGNORECASE)))


def _fetch_cisa_url_text(url: str) -> str:
    """Fetch CISA content, falling back to curl if requests is blocked."""
    try:
        response = requests.get(
            url,
            headers=CISA_REQUEST_HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        return response.text
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 403:
            raise

    result = subprocess.run(
        [
            "curl",
            "-L",
            "--max-time",
            "30",
            "-A",
            CISA_REQUEST_HEADERS["User-Agent"],
            url,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _extract_cisa_archive_links(listing_html: str) -> list[str]:
    """Extract AA-style cybersecurity advisory links from the archive page."""
    matches = re.findall(
        r'href="(/news-events/cybersecurity-advisories/aa\d{2}-\d{3}[a-z])"',
        listing_html,
        flags=re.IGNORECASE,
    )
    return sorted(set(urljoin("https://www.cisa.gov", match) for match in matches))


def _extract_cisa_page_title(page_html: str) -> str:
    """Extract advisory title from page metadata."""
    match = re.search(r'<meta property="og:title" content="([^"]+?) \| CISA"', page_html)
    if match:
        return html.unescape(match.group(1)).strip()

    match = re.search(r"<title>([^<]+?) \| CISA</title>", page_html)
    if match:
        return html.unescape(match.group(1)).strip()

    return "Unknown CISA Advisory"


def _extract_cisa_alert_code(page_html: str, fallback_url: str) -> str:
    """Extract the CISA alert/advisory code."""
    match = re.search(
        r'c-field--name-field-alert-code.*?<div class="c-field__content">([^<]+)</div>',
        page_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return html.unescape(match.group(1)).strip().upper()
    return fallback_url.rstrip("/").split("/")[-1].upper()


def _extract_cisa_body_html(page_html: str) -> str:
    """Extract the main advisory body content from the page."""
    main_match = re.search(r"<main\b.*?</main>", page_html, flags=re.IGNORECASE | re.DOTALL)
    if not main_match:
        return page_html

    main_html = main_match.group(0)
    start_idx = 0

    content_start = CISA_ADVISORY_START_HEADING_RE.search(main_html)
    if content_start:
        start_idx = content_start.start()
    else:
        alert_code_match = re.search(
            r'c-field--name-field-alert-code.*?</div>\s*</div>',
            main_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if alert_code_match:
            after_alert = main_html[alert_code_match.end():]
            first_heading = re.search(r"<h[2-4][^>]*>.*?</h[2-4]>", after_alert, flags=re.IGNORECASE | re.DOTALL)
            if first_heading:
                start_idx = alert_code_match.end() + first_heading.start()

    advisory_html = main_html[start_idx:]
    end_markers = [
        r"<h[2-4][^>]*>\s*(?:<strong>)?\s*Please share your thoughts",
        r"<h[2-4][^>]*>\s*(?:<strong>)?\s*Related Advisories",
        r'c-product-survey',
        r'c-field--name-field-tags',
        r'This product is provided subject to this',
    ]
    end_positions = [
        match.start()
        for pattern in end_markers
        if (match := re.search(pattern, advisory_html, flags=re.IGNORECASE | re.DOTALL))
    ]
    if end_positions:
        advisory_html = advisory_html[:min(end_positions)]

    return advisory_html


def _sha256_file(filepath: Path) -> str:
    """Calculate SHA256 checksum of a file for reproducibility."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _parse_snapshot_cutoff(snapshot_date: str | None):
    """Parse a YYYY-MM-DD snapshot date into a date object."""
    if not snapshot_date:
        return None
    return datetime.strptime(snapshot_date, "%Y-%m-%d").date()


def _parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse ISO 8601 timestamps used by upstream CTI sources."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_cisa_temporal_field(page_html: str, field_name: str) -> datetime | None:
    """Extract a CISA advisory datetime field from the HTML page."""
    match = re.search(
        rf'c-field--name-{re.escape(field_name)}.*?<time datetime="([^"]+)"',
        page_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return _parse_iso_datetime(match.group(1))


def _save_manifest(
    directory: Path,
    files: list[dict],
    snapshot_date: str | None = None,
    extra_metadata: dict | None = None,
):
    """Save download manifest with checksums for reproducibility."""
    manifest = {
        "download_date": datetime.now().isoformat(),
        "files": files,
    }
    if snapshot_date:
        manifest["snapshot_date"] = snapshot_date
    if extra_metadata:
        manifest.update(extra_metadata)
    manifest_path = directory / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved: {manifest_path}")


def download_nvd(
    output_dir: Path,
    min_cvss: float = 7.0,
    year_start: int = 2020,
    year_end: int = 2025,
    api_key: str | None = None,
    snapshot_date: str | None = None,
):
    """
    Download CVE data from NVD API 2.0.

    Downloads in batches by year and CVSS severity to stay within
    API rate limits. Without API key: 5 requests per 30 seconds.
    With API key: 50 requests per 30 seconds.

    Args:
        output_dir: Directory to save JSON files
        min_cvss: Minimum CVSS v3 base score
        year_start: Start year for CVE date range
        year_end: End year for CVE date range
        api_key: Optional NVD API key for higher rate limits
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    headers = {}
    snapshot_cutoff = _parse_snapshot_cutoff(snapshot_date)
    if api_key:
        headers["apiKey"] = api_key

    # Rate limiting
    delay = 6.0 if not api_key else 0.6  # seconds between requests

    for year in range(year_start, year_end + 1):
        start_index = 0
        results_per_page = 2000
        year_docs = []
        max_retries = 3

        # Split year into quarters to stay under NVD's 10K result limit per query
        quarters = [
            (f"{year}-01-01T00:00:00.000", f"{year}-03-31T23:59:59.999"),
            (f"{year}-04-01T00:00:00.000", f"{year}-06-30T23:59:59.999"),
            (f"{year}-07-01T00:00:00.000", f"{year}-09-30T23:59:59.999"),
            (f"{year}-10-01T00:00:00.000", f"{year}-12-31T23:59:59.999"),
        ]

        logger.info(f"Downloading NVD CVEs for {year} (will filter CVSS >= {min_cvss} locally)...")

        for q_start, q_end in quarters:
            start_index = 0

            while True:
                params = {
                    "pubStartDate": q_start,
                    "pubEndDate": q_end,
                    "startIndex": start_index,
                    "resultsPerPage": results_per_page,
                }

                success = False
                for attempt in range(max_retries):
                    try:
                        response = requests.get(
                            NVD_API_URL, params=params, headers=headers, timeout=30
                        )
                        response.raise_for_status()
                        data = response.json()
                        success = True
                        break
                    except requests.RequestException as e:
                        logger.warning(f"NVD API attempt {attempt+1}/{max_retries} failed: {e}")
                        time.sleep(delay * (attempt + 1))

                if not success:
                    logger.error(f"Skipping {q_start} to {q_end} at index {start_index} after {max_retries} retries")
                    break

                vulns = data.get("vulnerabilities", [])
                total = data.get("totalResults", 0)
                year_docs.extend(vulns)

                logger.info(f"  {year} ({q_start[:10]} to {q_end[:10]}): fetched {start_index + len(vulns)}/{total}")

                if start_index + results_per_page >= total:
                    break

                start_index += results_per_page
                time.sleep(delay)

        # Deduplicate by CVE ID
        seen = set()
        unique_docs = []
        for v in year_docs:
            cve_id = v.get("cve", {}).get("id", "")
            if cve_id not in seen:
                seen.add(cve_id)
                unique_docs.append(v)

        # Filter by CVSS v3 score locally (more reliable than API param)
        filtered_docs = []
        for v in unique_docs:
            cve = v.get("cve", {})
            metrics = cve.get("metrics", {})
            published_dt = _parse_iso_datetime(cve.get("published"))
            modified_dt = _parse_iso_datetime(cve.get("lastModified"))

            if snapshot_cutoff:
                if published_dt and published_dt.date() > snapshot_cutoff:
                    continue
                if modified_dt and modified_dt.date() > snapshot_cutoff:
                    continue

            # Check cvssMetricV31, then cvssMetricV30
            cvss_score = 0.0
            for key in ("cvssMetricV31", "cvssMetricV30"):
                metric_list = metrics.get(key, [])
                if metric_list:
                    cvss_score = metric_list[0].get("cvssData", {}).get("baseScore", 0.0)
                    break
            if cvss_score >= min_cvss:
                filtered_docs.append(v)

        unique_docs = filtered_docs
        logger.info(f"  {year}: {len(unique_docs)} CVEs after CVSS >= {min_cvss} filter")

        # Save year file
        output_file = output_dir / f"nvd_cves_{year}.json"
        output_data = {
            "resultsPerPage": len(unique_docs),
            "totalResults": len(unique_docs),
            "vulnerabilities": unique_docs,
            "download_date": datetime.now().isoformat(),
        }
        if snapshot_date:
            output_data["snapshot_date"] = snapshot_date

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f)

        checksum = _sha256_file(output_file)
        manifest_files.append({
            "filename": output_file.name,
            "year": year,
            "cve_count": len(unique_docs),
            "sha256": checksum,
        })

        logger.info(f"Saved {len(unique_docs)} CVEs for {year} -> {output_file.name}")
        time.sleep(delay)

    _save_manifest(
        output_dir,
        manifest_files,
        snapshot_date=snapshot_date,
        extra_metadata={"min_cvss": min_cvss, "year_range": [year_start, year_end]},
    )


def download_cisa_kev(output_dir: Path, snapshot_date: str | None = None):
    """
    Download CISA Known Exploited Vulnerabilities catalog.
    Single JSON file, straightforward download.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_cutoff = _parse_snapshot_cutoff(snapshot_date)

    logger.info("Downloading CISA KEV catalog...")
    response = requests.get(CISA_KEV_URL, timeout=30)
    response.raise_for_status()
    kev_data = response.json()

    if snapshot_cutoff:
        filtered_vulnerabilities = []
        for vuln in kev_data.get("vulnerabilities", []):
            date_added = vuln.get("dateAdded")
            if not date_added:
                continue
            try:
                added_date = datetime.strptime(date_added, "%Y-%m-%d").date()
            except ValueError:
                continue
            if added_date <= snapshot_cutoff:
                filtered_vulnerabilities.append(vuln)

        kev_data["vulnerabilities"] = filtered_vulnerabilities
        kev_data["snapshot_date"] = snapshot_date

    output_file = output_dir / "known_exploited_vulnerabilities.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(kev_data, f)

    checksum = _sha256_file(output_file)
    count = len(kev_data.get("vulnerabilities", []))

    _save_manifest(
        output_dir,
        [{
            "filename": output_file.name,
            "vulnerability_count": count,
            "sha256": checksum,
            "catalog_version": kev_data.get("catalogVersion", ""),
        }],
        snapshot_date=snapshot_date,
    )

    logger.info(f"CISA KEV: {count} vulnerabilities saved to {output_file.name}")


def download_cisa_advisories(
    output_dir: Path,
    year_start: int = 2020,
    year_end: int = 2026,
    snapshot_date: str | None = None,
):
    """
    Download CISA Cybersecurity Advisories from the official RSS feed.

    The feed is filtered to CISA's /news-events/cybersecurity-advisories/ entries
    and normalized into the JSON structure expected by parse_cisa_advisory().
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob("*.json"):
        stale_file.unlink()

    snapshot_cutoff = datetime.strptime(snapshot_date, "%Y-%m-%d").date() if snapshot_date else None

    manifest_files = []
    advisory_count = 0

    for year in range(year_start, year_end + 1):
        logger.info(f"Downloading CISA Cybersecurity Advisories archive for {year}...")
        listing_html = _fetch_cisa_url_text(CISA_ADVISORY_ARCHIVE_URL.format(year=year))
        advisory_links = _extract_cisa_archive_links(listing_html)

        for link in advisory_links:
            page_html = _fetch_cisa_url_text(link)
            release_dt = _extract_cisa_temporal_field(page_html, "field-release-date")
            last_updated_dt = _extract_cisa_temporal_field(page_html, "field-last-updated") or release_dt
            if not release_dt and not last_updated_dt:
                logger.warning(f"Skipping advisory without parsable release/update date: {link}")
                continue
            if release_dt is None:
                release_dt = last_updated_dt

            release_date = release_dt.date()
            if snapshot_cutoff and release_date > snapshot_cutoff:
                continue
            if snapshot_cutoff and last_updated_dt and last_updated_dt.date() > snapshot_cutoff:
                continue

            advisory_id = _extract_cisa_alert_code(page_html, link)
            title = _extract_cisa_page_title(page_html)
            advisory_body_html = _extract_cisa_body_html(page_html)
            sections = _extract_html_sections(advisory_body_html)
            if not sections:
                continue

            advisory_doc = {
                "id": advisory_id,
                "title": title,
                "published": release_dt.isoformat(),
                "last_updated": last_updated_dt.isoformat() if last_updated_dt else "",
                "cve_ids": _extract_cve_ids_from_text(advisory_body_html),
                "sections": sections,
                "metadata": {
                    "source_url": link,
                    "archive_year": year,
                    "release_date": release_dt.isoformat(),
                    "last_updated": last_updated_dt.isoformat() if last_updated_dt else "",
                },
            }

            output_file = output_dir / f"{advisory_id.lower()}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(advisory_doc, f, indent=2)

            manifest_files.append(
                {
                    "filename": output_file.name,
                    "advisory_id": advisory_id,
                    "published": release_dt.isoformat(),
                    "last_updated": last_updated_dt.isoformat() if last_updated_dt else "",
                    "source_url": link,
                    "archive_year": year,
                    "sha256": _sha256_file(output_file),
                }
            )
            advisory_count += 1

    _save_manifest(
        output_dir,
        manifest_files,
        snapshot_date=snapshot_date,
        extra_metadata={"year_range": [year_start, year_end]},
    )
    logger.info(f"CISA Cybersecurity Advisories: {advisory_count} advisories saved to {output_dir}")


def download_misp_feeds(output_dir: Path, max_events: int = 500, snapshot_date: str | None = None):
    """
    Download public MISP feeds from CIRCL.

    Uses the CIRCL OSINT MISP feed (public, no authentication required).
    Downloads the manifest first, then fetches individual event JSON files.

    Args:
        output_dir: Directory to save JSON files
        max_events: Maximum number of events to download (most recent first)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob("*.json"):
        stale_file.unlink()

    snapshot_cutoff = _parse_snapshot_cutoff(snapshot_date)
    manifest_url = "https://www.circl.lu/doc/misp/feed-osint/manifest.json"

    logger.info("Downloading CIRCL OSINT MISP feed manifest...")
    try:
        resp = requests.get(manifest_url, timeout=30)
        resp.raise_for_status()
        manifest = resp.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch MISP manifest: {e}")
        return

    candidate_event_ids = []
    for event_id, event_meta in manifest.items():
        event_timestamp = event_meta.get("timestamp")
        if snapshot_cutoff and event_timestamp:
            try:
                event_date = datetime.fromtimestamp(int(event_timestamp)).date()
            except (ValueError, TypeError, OSError):
                continue
            if event_date > snapshot_cutoff:
                continue
        candidate_event_ids.append(event_id)

    # Sort events by timestamp (most recent first), take max_events
    event_ids = sorted(
        candidate_event_ids,
        key=lambda eid: manifest[eid].get("timestamp", "0"),
        reverse=True,
    )[:max_events]

    logger.info(f"MISP feed: {len(manifest)} events available, downloading {len(event_ids)}...")

    base_url = "https://www.circl.lu/doc/misp/feed-osint/"
    downloaded = 0
    errors = 0
    download_manifest = []

    for i, event_id in enumerate(tqdm(event_ids, desc="MISP events")):
        event_file = output_dir / f"{event_id}.json"

        try:
            event_url = f"{base_url}{event_id}.json"
            resp = requests.get(event_url, timeout=15)
            resp.raise_for_status()

            with open(event_file, "w", encoding="utf-8") as f:
                json.dump(resp.json(), f)

            downloaded += 1
            download_manifest.append({
                "filename": f"{event_id}.json",
                "event_info": manifest[event_id].get("info", "")[:100],
                "timestamp": manifest[event_id].get("timestamp", ""),
                "sha256": _sha256_file(event_file),
            })

            # Be polite to CIRCL's server
            time.sleep(0.5)

        except requests.RequestException as e:
            errors += 1
            logger.warning(f"Failed to download MISP event {event_id}: {e}")
            if errors > 20:
                logger.error("Too many errors, stopping MISP download")
                break

    _save_manifest(
        output_dir,
        download_manifest,
        snapshot_date=snapshot_date,
        extra_metadata={
            "max_events": max_events,
            "total_available": len(manifest),
            "selected_events": len(event_ids),
        },
    )
    logger.info(f"MISP download complete: {downloaded} events saved, {errors} errors")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Download CTI data sources")
    parser.add_argument("--source", choices=["nvd", "cisa_kev", "cisa_advisories", "all"], default="all")
    parser.add_argument("--nvd-api-key", type=str, default=None, help="NVD API key for faster downloads")
    args = parser.parse_args()

    config = load_config()
    root = get_project_root()

    if args.source in ("nvd", "all"):
        nvd_config = config["data"]["sources"]["nvd"]
        download_nvd(
            output_dir=root / nvd_config["raw_dir"],
            min_cvss=nvd_config.get("min_cvss", 7.0),
            year_start=nvd_config["year_range"][0],
            year_end=nvd_config["year_range"][1],
            api_key=args.nvd_api_key,
            snapshot_date=config["data"].get("snapshot_date"),
        )

    if args.source in ("cisa_kev", "all"):
        kev_config = config["data"]["sources"]["cisa_kev"]
        download_cisa_kev(
            output_dir=root / kev_config["raw_dir"],
            snapshot_date=config["data"].get("snapshot_date"),
        )

    if args.source in ("cisa_advisories", "all"):
        advisory_config = config["data"]["sources"]["cisa_advisories"]
        download_cisa_advisories(
            output_dir=root / advisory_config["raw_dir"],
            year_start=advisory_config.get("year_range", [2020, datetime.now().year])[0],
            year_end=advisory_config.get("year_range", [2020, datetime.now().year])[1],
            snapshot_date=config["data"].get("snapshot_date"),
        )
