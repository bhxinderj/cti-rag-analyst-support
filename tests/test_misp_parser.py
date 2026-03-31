import unittest

from src.cti_rag.ingestion.misp_parser import _extract_attack_techniques, parse_misp_event


class TestMispAttackTechniqueExtraction(unittest.TestCase):
    def test_extracts_ids_from_tag_format(self):
        event = {
            "Tag": [
                {"name": 'misp-galaxy:mitre-attack-pattern="Exploit Public-Facing Application - T1190"'},
                {"name": 'misp-galaxy:mitre-attack-pattern="Spearphishing Attachment - T1566.001"'},
                {"name": 'misp-galaxy:stix-2.1-attack-pattern="041bc611-87da-4ad4-a46b-b37926180b7d"'},
            ]
        }

        techniques = _extract_attack_techniques(event)

        self.assertEqual(techniques, ["T1190", "T1566.001"])

    def test_extracts_ids_from_galaxy_cluster_meta(self):
        event = {
            "Galaxy": [
                {
                    "type": "mitre-attack-pattern",
                    "GalaxyCluster": [
                        {
                            "tag_name": 'misp-galaxy:mitre-attack-pattern="PowerShell - T1059.001"',
                            "value": "PowerShell",
                            "meta": {
                                "external_id": ["attack-pattern--1", "T1059.001", "T1190"],
                            },
                        }
                    ],
                }
            ]
        }

        techniques = _extract_attack_techniques(event)

        self.assertEqual(techniques, ["T1059.001", "T1190"])

    def test_deduplicates_attack_ids_across_galaxy_and_tags(self):
        event = {
            "Galaxy": [
                {
                    "type": "mitre-attack-pattern",
                    "GalaxyCluster": [
                        {
                            "tag_name": 'misp-galaxy:mitre-attack-pattern="Exploit Public-Facing Application - t1190"',
                            "meta": {"external_id": ["T1190", "T1566.001"]},
                        }
                    ],
                }
            ],
            "Tag": [
                {"name": 'misp-galaxy:mitre-attack-pattern="Exploit Public-Facing Application - T1190"'},
            ],
        }

        techniques = _extract_attack_techniques(event)

        self.assertEqual(techniques, ["T1190", "T1566.001"])

    def test_parse_misp_event_extracts_cve_product_and_prioritized_iocs_from_comments(self):
        event = {
            "id": "event-2",
            "info": "ET EXPLOIT FortiGate VPN Remote Code Execution Vulnerability - CVE-2023-27997",
            "threat_level_id": "1",
            "Tag": [{"name": "tlp:clear"}],
            "Attribute": [
                {
                    "type": "ip-src",
                    "value": "203.0.113.10",
                    "category": "Network activity",
                    "to_ids": True,
                    "comment": "ET EXPLOIT FortiGate VPN - Repeated GET Requests to /remote/logincheck (CVE-2023-27997) [T1190]",
                },
                {
                    "type": "domain",
                    "value": "fortigate-updates.example",
                    "category": "Network activity",
                    "to_ids": True,
                    "comment": "",
                },
            ],
        }

        doc = parse_misp_event(event)

        self.assertIsNotNone(doc)
        self.assertEqual(doc.severity.value, "high")
        self.assertEqual(doc.cve_ids, ["CVE-2023-27997"])
        self.assertEqual(doc.attack_techniques, ["T1190"])
        self.assertIn("FortiGate VPN", doc.affected_products)
        self.assertIn("Related Products: FortiGate VPN", doc.content)
        self.assertIn("Key IOCs: domain: fortigate-updates.example; ip-src: 203.0.113.10", doc.content)
        self.assertIn("Observed Context:", doc.content)


if __name__ == "__main__":
    unittest.main()
