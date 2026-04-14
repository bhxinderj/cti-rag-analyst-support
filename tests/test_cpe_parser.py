"""
Unit tests for the CPE 2.3 humanizer.

Covers the spec §5.6 contract: known product/vendor branding,
acronym preservation, mechanical fallback for unknown tokens, and
graceful degradation on malformed inputs.
"""

from src.cti_rag.rag.cpe import humanize_cpe, humanize_cpes


def test_humanize_cpe_citrix_netscaler_adc_matches_spec_example():
    result = humanize_cpe("cpe:2.3:a:citrix:netscaler_adc:*:*:*:*:*:*:*:*")
    assert result == "Citrix NetScaler ADC"


def test_humanize_cpe_falls_back_to_title_case_for_unknown_vendor_and_product():
    # acme / widget are not in the override tables, so mechanical
    # title-casing applies. This guards against regressions where an
    # unknown product silently produces the wrong case.
    assert humanize_cpe("cpe:2.3:a:acme:widget:1.2.3:*:*:*:*:*:*:*") == "Acme Widget"


def test_humanize_cpe_preserves_acronyms_in_product_tokens():
    # "vpn_client" -> known acronym VPN + capitalized Client.
    assert (
        humanize_cpe("cpe:2.3:a:acme:vpn_client:*:*:*:*:*:*:*:*") == "Acme VPN Client"
    )


def test_humanize_cpe_uses_vendor_override_for_multi_word_vendor():
    # "paloaltonetworks" -> "Palo Alto Networks" via vendor override.
    result = humanize_cpe("cpe:2.3:o:paloaltonetworks:panos:10.2:*:*:*:*:*:*:*")
    assert result.startswith("Palo Alto Networks")


def test_humanize_cpe_ibm_vendor_is_fully_uppercased():
    assert humanize_cpe("cpe:2.3:a:ibm:websphere:9.0:*:*:*:*:*:*:*").startswith("IBM")


def test_humanize_cpe_returns_code_wrapped_raw_on_malformed_input():
    assert humanize_cpe("not a cpe string") == "<code>not a cpe string</code>"


def test_humanize_cpe_returns_code_wrapped_raw_on_missing_colon_prefix():
    # Right prefix shape but wrong version.
    assert humanize_cpe("cpe:/a:citrix:netscaler_adc").startswith("<code>")


def test_humanize_cpe_handles_empty_input():
    assert humanize_cpe("") == ""
    assert humanize_cpe(None) == ""  # type: ignore[arg-type]


def test_humanize_cpes_deduplicates_while_preserving_order():
    cpes = [
        "cpe:2.3:a:citrix:netscaler_adc:*:*:*:*:*:*:*:*",
        "cpe:2.3:a:citrix:netscaler_adc:13.1:*:*:*:*:*:*:*",
        "cpe:2.3:a:citrix:netscaler_gateway:*:*:*:*:*:*:*:*",
    ]
    result = humanize_cpes(cpes)
    assert result == ["Citrix NetScaler ADC", "Citrix NetScaler Gateway"]


def test_humanize_cpes_keeps_unparseable_entries_once():
    cpes = ["cpe:2.3:a:citrix:netscaler_adc:*:*:*:*:*:*:*:*", "garbage", "garbage"]
    result = humanize_cpes(cpes)
    assert result == ["Citrix NetScaler ADC", "<code>garbage</code>"]
