"""
CPE 2.3 parser utility.

Purpose
-------
Convert Common Platform Enumeration (CPE 2.3) strings — used by NVD to
identify affected products — into human-readable product names that an
analyst can read at a glance inside the triage card.

Example
-------
>>> humanize_cpe("cpe:2.3:a:citrix:netscaler_adc:*:*:*:*:*:*:*:*")
'Citrix NetScaler ADC'

>>> humanize_cpe("not a cpe string")
'<code>not a cpe string</code>'

Design decisions (see docs/triage-template-spec.md §5.6)
-------------------------------------------------------
- Applied on-the-fly at render time, not at ingestion time. No re-index
  cost, and the transformation is co-located with template concerns.
- Parse failures return the raw CPE inside ``<code>`` tags so the analyst
  still sees the verbatim identifier and can look it up directly.
- A small curated vocabulary of well-known acronyms and vendor/product
  overrides lifts the output from mechanical title-casing (``Netscaler
  Adc``) to the conventional brand spelling (``NetScaler ADC``).
  Unknown tokens fall back to title-case so the parser degrades
  gracefully on CPEs we have not seen before.
"""

from __future__ import annotations

import re
from typing import Iterable

# Curated acronyms that should stay fully upper-case after normalization.
# Keep this list short and high-signal: every entry is a domain term that
# analysts read as an acronym, not as a word.
_ACRONYMS: frozenset[str] = frozenset(
    {
        "ADC",
        "API",
        "ASA",
        "BIG",  # part of F5 "BIG-IP" which we also special-case below
        "CMS",
        "CPU",
        "DLL",
        "DNS",
        "EMC",
        "FTP",
        "GPU",
        "HTTP",
        "HTTPS",
        "IBM",
        "IIS",
        "IOS",
        "IP",
        "LDAP",
        "NAS",
        "NAT",
        "OS",
        "PDF",
        "PHP",
        "RCE",
        "RDP",
        "SDK",
        "SMB",
        "SMTP",
        "SQL",
        "SSH",
        "SSL",
        "TLS",
        "URL",
        "USB",
        "VPN",
        "WAF",
        "XML",
        "XSS",
    }
)

# Vendor/product tokens whose canonical branding is not recoverable by
# mechanical casing rules. Keyed on the lowercased underscore-joined
# CPE token; value is the desired display form.
_PRODUCT_OVERRIDES: dict[str, str] = {
    "netscaler": "NetScaler",
    "netscaler_adc": "NetScaler ADC",
    "netscaler_gateway": "NetScaler Gateway",
    "big_ip": "BIG-IP",
    "bigip": "BIG-IP",
    "macos": "macOS",
    "macos_x": "macOS X",
    "ios": "iOS",
    "ipados": "iPadOS",
    "iphone_os": "iOS",
    "watchos": "watchOS",
    "tvos": "tvOS",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "powerpoint": "PowerPoint",
    "onedrive": "OneDrive",
    "sharepoint": "SharePoint",
    "github": "GitHub",
    "gitlab": "GitLab",
    "openssh": "OpenSSH",
    "openssl": "OpenSSL",
    "wordpress": "WordPress",
    "phpmyadmin": "phpMyAdmin",
    "mysql": "MySQL",
    "mariadb": "MariaDB",
    "postgresql": "PostgreSQL",
    "mongodb": "MongoDB",
    "vmware": "VMware",
    "vcenter_server": "vCenter Server",
    "vcenter": "vCenter",
    "vsphere": "vSphere",
    "esxi": "ESXi",
    "exchange_server": "Exchange Server",
    "sharepoint_server": "SharePoint Server",
    "windows_server": "Windows Server",
    "windows_10": "Windows 10",
    "windows_11": "Windows 11",
}

_VENDOR_OVERRIDES: dict[str, str] = {
    "ibm": "IBM",
    "hp": "HP",
    "hpe": "HPE",
    "emc": "EMC",
    "sap": "SAP",
    "f5": "F5",
    "citrix": "Citrix",
    "cisco": "Cisco",
    "microsoft": "Microsoft",
    "apple": "Apple",
    "google": "Google",
    "oracle": "Oracle",
    "redhat": "Red Hat",
    "vmware": "VMware",
    "fortinet": "Fortinet",
    "paloaltonetworks": "Palo Alto Networks",
    "checkpoint": "Check Point",
    "check_point": "Check Point",
    "jetbrains": "JetBrains",
    "pulsesecure": "Pulse Secure",
    "progress": "Progress Software",
    "atlassian": "Atlassian",
}

_CPE_PREFIX_RE = re.compile(r"^cpe:2\.3:[aho]:", flags=re.IGNORECASE)


def _humanize_token(token: str, overrides: dict[str, str]) -> str:
    """Humanize a single CPE token (underscore-joined) into display form."""
    if not token or token in {"*", "-"}:
        return ""

    lowered = token.lower()
    if lowered in overrides:
        return overrides[lowered]

    # Token may itself be composed of multiple words via underscores.
    parts = [p for p in lowered.split("_") if p]
    if not parts:
        return ""

    display_parts: list[str] = []
    for part in parts:
        upper = part.upper()
        if upper in _ACRONYMS:
            display_parts.append(upper)
        elif part in overrides:
            # Per-sub-token override (rarely needed but cheap to honour).
            display_parts.append(overrides[part])
        else:
            display_parts.append(part.capitalize())

    return " ".join(display_parts)


def _split_cpe23(cpe: str) -> list[str] | None:
    """
    Split a CPE 2.3 string into its 13 fields.

    Returns ``None`` if the string does not look like a valid CPE 2.3
    formatted name (prefix check + minimum field count). We intentionally
    do not attempt full unescaping of ``\\:`` — analyst-facing rendering
    does not require it and real CTI feeds virtually never escape.
    """
    if not cpe or not isinstance(cpe, str):
        return None

    stripped = cpe.strip()
    if not _CPE_PREFIX_RE.match(stripped):
        return None

    parts = stripped.split(":")
    # CPE 2.3 formatted-string has 13 components:
    # cpe:2.3:<part>:<vendor>:<product>:<version>:<update>:<edition>:
    # <language>:<sw_edition>:<target_sw>:<target_hw>:<other>
    if len(parts) < 5:
        return None

    return parts


def humanize_cpe(cpe: str) -> str:
    """
    Render a CPE 2.3 string as ``"<Vendor> <Product>"``.

    On parse failure returns the raw input wrapped in ``<code>`` tags so
    the analyst still sees the verbatim identifier.
    """
    parts = _split_cpe23(cpe)
    if parts is None:
        raw = (cpe or "").strip()
        return f"<code>{raw}</code>" if raw else ""

    vendor_token = parts[3] if len(parts) > 3 else ""
    product_token = parts[4] if len(parts) > 4 else ""

    vendor = _humanize_token(vendor_token, _VENDOR_OVERRIDES)
    product = _humanize_token(product_token, _PRODUCT_OVERRIDES)

    if vendor and product:
        # Avoid duplicating the vendor if the product override already
        # contains it (e.g. "BIG-IP" for vendor=f5, product=big_ip — the
        # analyst convention writes "F5 BIG-IP" with vendor prefixed).
        return f"{vendor} {product}"
    if product:
        return product
    if vendor:
        return vendor

    # Degenerate CPE with no vendor/product content — fall back to raw.
    return f"<code>{cpe.strip()}</code>"


def humanize_cpes(cpes: Iterable[str]) -> list[str]:
    """
    Humanize a sequence of CPE strings, de-duplicating while preserving order.

    Empty/parse-failed inputs that collapse to identical ``<code>…</code>``
    wrappers are still kept once so the analyst sees every unparseable
    entry exactly once.
    """
    seen: set[str] = set()
    result: list[str] = []
    for cpe in cpes:
        display = humanize_cpe(cpe)
        if not display:
            continue
        if display in seen:
            continue
        seen.add(display)
        result.append(display)
    return result
