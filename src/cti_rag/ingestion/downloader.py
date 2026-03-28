"""
CTI Data Downloader.

Downloads static snapshots of CTI data sources for reproducible evaluation.
All downloads are timestamped and checksummed.

Usage:
    python -m src.ingestion.downloader --all
    python -m src.ingestion.downloader --source nvd
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

from ..utils.config import load_config, get_project_root

logger = logging.getLogger(__name__)

# NVD API 2.0 endpoint
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# CISA KEV catalog
CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def _sha256_file(filepath: Path) -> str:
    """Calculate SHA256 checksum of a file for reproducibility."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _save_manifest(directory: Path, files: list[dict]):
    """Save download manifest with checksums for reproducibility."""
    manifest = {
        "download_date": datetime.now().isoformat(),
        "files": files,
    }
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

    _save_manifest(output_dir, manifest_files)


def download_cisa_kev(output_dir: Path):
    """
    Download CISA Known Exploited Vulnerabilities catalog.
    Single JSON file, straightforward download.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading CISA KEV catalog...")
    response = requests.get(CISA_KEV_URL, timeout=30)
    response.raise_for_status()

    output_file = output_dir / "known_exploited_vulnerabilities.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(response.json(), f)

    checksum = _sha256_file(output_file)
    kev_data = response.json()
    count = len(kev_data.get("vulnerabilities", []))

    _save_manifest(output_dir, [{
        "filename": output_file.name,
        "vulnerability_count": count,
        "sha256": checksum,
        "catalog_version": kev_data.get("catalogVersion", ""),
    }])

    logger.info(f"CISA KEV: {count} vulnerabilities saved to {output_file.name}")


def download_misp_feeds(output_dir: Path, max_events: int = 500):
    """
    Download public MISP feeds from CIRCL.

    Uses the CIRCL OSINT MISP feed (public, no authentication required).
    Downloads the manifest first, then fetches individual event JSON files.

    Args:
        output_dir: Directory to save JSON files
        max_events: Maximum number of events to download (most recent first)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_url = "https://www.circl.lu/doc/misp/feed-osint/manifest.json"

    logger.info("Downloading CIRCL OSINT MISP feed manifest...")
    try:
        resp = requests.get(manifest_url, timeout=30)
        resp.raise_for_status()
        manifest = resp.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch MISP manifest: {e}")
        return

    # Sort events by timestamp (most recent first), take max_events
    event_ids = sorted(
        manifest.keys(),
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

        # Skip if already downloaded
        if event_file.exists():
            downloaded += 1
            continue

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

    _save_manifest(output_dir, download_manifest)
    logger.info(f"MISP download complete: {downloaded} events saved, {errors} errors")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Download CTI data sources")
    parser.add_argument("--source", choices=["nvd", "cisa_kev", "all"], default="all")
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
        )

    if args.source in ("cisa_kev", "all"):
        kev_config = config["data"]["sources"]["cisa_kev"]
        download_cisa_kev(output_dir=root / kev_config["raw_dir"])
