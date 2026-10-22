# Demo-Video: Shot-für-Shot-Skript (Masterprüfung)

Stand 2026-10-22. Alle Queries, Zeiten und Ausgaben in diesem Skript wurden an diesem Tag
gegen den Demo-Index (`CTI_RAG_SETUP=demo`) und den eingefrorenen Setup-B-Index
verifiziert. Sprache: Captions (Lower Thirds) Englisch, passend zur UI; Voiceover
Deutsch. Für eine englische Prüfung die Voiceover-Zeilen 1:1 übersetzen.

Gesamtlänge: ca. 2:30 min. Harte Obergrenze 3:00 min.

## 1. Dramaturgie

Die Lücke CVE-2026-109618 / CVE-2026-109619 („php2shell") wurde am 20.10.2026 geleakt,
am 21.10. gepatcht und von NVD und CISA erfasst. Kein Generationsmodell im Prototyp kann
sie kennen. Jede korrekte Aussage im Video stammt damit nachweislich aus dem Retrieval.

Erzählbogen in drei Akten:

| Akt | Szenen | Aussage |
|---|---|---|
| Vorher | 1, 2 | Das System rät nicht. Ohne Evidenz: Abstention. Modell allein: weiß nichts. |
| Nachher | 3, 4, 5 | Mit frischer Evidenz: deterministische Triage, Quellenvergleich, Konversation. |
| Grenzen | 6, 7, 8 | Integration (MCP), ehrliche Latenz der evaluierten lokalen Konfiguration, Deklaration. |

Pflichtdeklaration (Szene 8 und Voiceover in Szene 3): Evaluiert wurde Setup B mit
lokalem Llama 3.1 8B. Der Demo-Index ist der evaluierte Korpus plus ein Supplement vom
22.10.2026. Extended Analysis ist ein nicht evaluierter Komfortmodus.

Nicht behaupten: dass die FH Technikum Wien betroffen war. Öffentlich belegt sind TU Wien,
Uni Wien, WU, BOKU, MedUni Wien und JKU. Formulierung: „österreichische Hochschulen".

## 2. Pre-Flight

### 2.1 Daten und Index

```bash
make demo-supplement
```

```bash
make demo-index
```

Beides ist idempotent. Vor der Aufnahme einmal laufen lassen, danach prüfen:

- KEV-Status. Steht eine der CVEs inzwischen im CISA-KEV-Katalog, springt die Triage-Ampel
  von „HIGH — Review" auf „CRITICAL — Immediate attention". Die Voiceover-Zeile in Szene 3
  dann anpassen (Variante B unten). Prüfen mit:

```bash
curl -s https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json | grep -c "CVE-2026-10961"
```

  Ist der Zähler größer als 0: den frischen KEV-Export nach
  `data/raw_demo/cisa_kev/known_exploited_vulnerabilities.json` legen und `make demo-index`
  erneut ausführen. Der Parser nimmt das Format unverändert.

### 2.2 Backend und UI

Die UI spricht mit der API auf Port 8000. Setup B und Demo sind getrennte API-Prozesse.
Empfehlung: Szenen 1 und 2 zuerst mit Setup B aufnehmen, dann API neu starten.

Setup B (Szenen 1, 2):

```bash
make api SETUP=b
```

```bash
make ui SETUP=b
```

Demo (Szenen 3 bis 7):

```bash
make api SETUP=demo
```

```bash
make ui SETUP=demo
```

Alternative ohne Neustart: zweite API auf Port 8001 und zweite UI auf Port 8502 mit
`CTI_RAG_API_URL=http://localhost:8001`. Kostet doppelt Speicher für Embedding, Reranker
und Chroma; auf dem M4 mit 24 GB machbar.

Vorwärmen vor jedem Aufnahmeblock: eine beliebige Query einmal durchlaufen lassen, damit
Modelle und Chroma im Speicher sind. Danach in der Sidebar „＋ New chat".

### 2.3 Bild

- Eigenes, leeres Browser-Profil. Keine Lesezeichenleiste, keine Tabs, keine Extensions.
- Fenster 1920×1080, Browser-Zoom 125 %. Dark Theme und `toolbarMode = "minimal"` sind in
  `.streamlit/config.toml` bereits gesetzt.
- Sidebar aufgeklappt lassen: „Setup: b" bzw. „Setup: demo" ist Beweismittel.
- Nicht stören an, Dock aus, Desktop leer.
- Queries per Cmd+V einfügen, nicht tippen.
- Jede Szene als eigener Clip, mindestens zwei Takes.
- Nach jedem Ergebnis zwei Sekunden Ruhe, Cursor aus dem Bild.
- Terminal für Szenen 2 und 6: Schrift 18 pt, Prompt gekürzt, keine Pfade mit Klarnamen,
  kein `env`-Output, keine API-Keys.

### 2.4 Lesen vor dem Weiterklicken

Im Extended-Modus formuliert ein hosted Modell die L2-Abschnitte. In einem Testlauf stand
unter „Source Agreement", NVD und CISA vergäben beide CVSS 9.8, obwohl der CISA-Alert keinen
Score nennt. Der L1-Block ist deterministisch und immer korrekt, L2 kann wackeln. Deshalb:
Take erst freigeben, nachdem du die L2-Sätze gelesen hast. Enthält einer eine Aussage, die
die zitierte Quelle nicht trägt, neuer Take. Im Zweifel die L1-Blöcke länger im Bild lassen.

## 3. Shot-Liste

Notation: **Query** ist der exakte Text zum Einfügen. **Bild** ist das, was zu sehen sein
muss. **Zoom** sind Nachbearbeitungs-Zooms. **Caption** ist die Lower-Third-Einblendung.
**VO** ist der Voiceover-Text. Zeiten sind Zielwerte im fertigen Schnitt.

### Szene 0: Titelkarte (0:00–0:05)

Standbild, kein UI.

Caption:
```
CTI-RAG Analyst Support — prototype demo
Query: php2shell (CVE-2026-109618 / CVE-2026-109619), disclosed 2026-10-20
```

VO: „Ein SOC-Analyst, fünf Quellen, eine neue kritische Lücke. Wir fragen den Prototypen
nach php2shell, zwei Tage nach der Veröffentlichung."

### Szene 1: Abstention auf dem evaluierten Index (0:05–0:25)

Setup: B. Sidebar: Retrieval Mode `hybrid`, Pipeline `Templated (Phase 2)`,
Extended Analysis **aus**.

Query:
```
What is CVE-2026-109618 and how severe is it?
```

Verifiziert: Abstention nach 3,6 s, Generation 0 ms, kein Modellaufruf.

Bild: Antwort „Insufficient evidence in the retrieved context to answer this question.
Missing: The retrieved context does not mention cve-2026-109618." Darunter Timing-Pills
mit `generation 0ms`. Expander „Retrieved context — not cited (5)" öffnen: fünf fremde
CVEs mit niedrigen Scores.

Zoom: Sidebar „Setup: b", dann Pill `generation 0ms`, dann der geöffnete Expander.

Caption:
```
Evaluated corpus (snapshot 2025-12-31): no evidence → explicit abstention, no model call
```

VO: „Der evaluierte Korpus endet am 31. Dezember 2025. Das System findet keine Evidenz und
lehnt ab, bevor überhaupt ein Modell aufgerufen wird. Es zeigt trotzdem, was es gefunden
hat, und markiert es als nicht verwendet."

Risiko: keins. Deterministisch, unter 4 s.

### Szene 2: Modell ohne Retrieval (0:25–0:35)

Terminal, Setup B. Optional, aber stark. Vorher aufgenommen, im Schnitt auf 10 s gekürzt.

```bash
CTI_RAG_SETUP=b .venv/bin/python main.py baseline "What is CVE-2026-109618 and how severe is it?"
```

Verifiziert: 16 s Generierung lokal. Antwort sinngemäß: „There is no information available
on CVE-2026-109618 … knowledge cutoff is 2025-12-31."

Bild: nur die Zeilen „Mode: BASELINE (no retrieval)" und der Summary-Absatz.

Caption:
```
Same local model, no retrieval: nothing to say
```

VO: „Das Sprachmodell allein kennt die Lücke nicht. Alles, was gleich kommt, muss also aus
den Quellen stammen."

Risiko: Die Baseline kann in einem anderen Take halluzinieren statt zuzugeben, dass sie
nichts weiß. Beides ist verwendbar; eine Halluzination wäre sogar das stärkere Argument,
braucht dann aber eine Caption „fabricated, no sources". Take lesen, dann entscheiden.

### Szene 3: VulnTriage auf dem Demo-Index (0:35–1:10)

Setup: demo. Sidebar: `hybrid`, `Templated (Phase 2)`, Extended Analysis **an**
(Sidebar-Hinweis „Data leaves the local machine" sichtbar).

Harter Schnitt von Szene 2, davor Zwischentitel 2 s:
```
Index refreshed: evaluated corpus + NVD records and CISA alert of 2026-10-21
```

Query (identisch zu Szene 1):
```
What is CVE-2026-109618 and how severe is it?
```

Verifiziert: 7 s, Template VulnTriage, zitiert `nvd_CVE-2026-109618` und
`cisa_advisory_CISA-ALERT-2026-10-21-PHP2SHELL_summary`.

Bild, in dieser Reihenfolge:
1. Pills: `extended`-Pill mit Modellname, `template · VulnTriage`, Timings.
2. L1-Block: „Triage Signal: HIGH — Review", „Rationale: CVSS 9.8 + no KEV listing, no
   ransomware evidence", „Listed in CISA KEV: no".
3. Expander „Routing decision" öffnen: `template: VulnTriage`, `primary_entities:
   ['CVE-2026-109618']`.
4. Kurz nach unten scrollen: „Exploitation Context", „Mitigations", „Gaps".
5. Expander „Cited Sources (2)".

Zoom: Sidebar „Setup: demo" und der Hinweis „Data leaves the local machine"; dann Triage
Signal; dann Routing decision.

Caption:
```
L1: deterministic fact bundle (no model) · L2: generated, every sentence cited
```

VO (Variante A, CVEs nicht in KEV): „Mit der frischen Evidenz routet der Prototyp die Frage
auf das VulnTriage-Template. Der obere Block ist deterministisch: CVSS 9,8, kein KEV-Eintrag,
Ampel ‚High, Review'. Critical wäre es erst mit bestätigter Ausnutzung. Darunter formuliert
ein Modell den Kontext, jeder Satz mit Quelle. Hier läuft der Extended-Modus mit einem
gehosteten Modell. Die evaluierte Konfiguration ist lokal; dazu gleich mehr."

VO (Variante B, falls inzwischen in KEV): „… CVSS 9,8 und ein KEV-Eintrag, Ampel ‚Critical,
Immediate attention'. …"

Risiko: Die L2-Sätze lesen (siehe 2.4). Der Abschnitt „Gaps" ist gewollt: er benennt, was
die Quellen nicht hergeben (IoCs, Detection Rules). Im Bild lassen.

### Szene 4: CrossSourceCompare (1:10–1:35)

Setup: demo. Extended **an**. Gleicher Chat oder „New chat", beides funktioniert, weil
die Query eigenständig ist.

Query:
```
What do NVD and CISA say about the php2shell vulnerabilities CVE-2026-109618 and CVE-2026-109619?
```

Verifiziert: 8,4 s, Template CrossSourceCompare, L1-Tabelle mit beiden CVEs, zitiert
NVD-Einträge beider CVEs und den CISA-Alert.

Bild: L1 „Comparison Table" (CVE, CVSS, KEV Listed, Ransomware, Vendor), darunter
„Top Matches by Triage Signal", dann „Source Agreement" und „Cross-Source Facts".

Zoom: Tabelle; dann der Satz in „Cross-Source Facts", der die zwei unterschiedlichen
Root Causes (longjmp/OOM vs. stale global state) nebeneinanderstellt.

Caption:
```
Two CVEs, three sources (NVD, CISA alert, KEV check): one deterministic table
```

VO: „Zwei CVEs, drei Quellen. Die Tabelle ist deterministisch aus den Fakten gebaut: gleiche
Schwere, beide noch nicht im KEV. Das Modell ergänzt, worin sich die Quellen unterscheiden:
NVD beschreibt zwei verschiedene Ursachen, CISA fasst beide unter php2shell zusammen."

Risiko: L2 „Source Agreement" lesen (2.4). Die Zeile „Entities with data from NVD and CISA
KEV: 0" ist korrekt und darf stehen bleiben.

### Szene 5: Konversatives Follow-up (1:35–1:55)

Setup: demo. Extended **an**. Im selben Chat direkt nach Szene 3 aufnehmen (die Szene
braucht den Vorgänger). Praktisch: Szene 3 und 5 in einem Take, Szene 4 separat.

Query:
```
and how do I mitigate it?
```

Verifiziert: Condense in 1,1 s zu „How do I mitigate CVE-2026-109618?", Antwort nach
11 s; `cisa_advisory_…_mitigations` auf Platz 1, Abschnitte „Mitigations" (fünf Punkte) und
„Detection & Hunting" (PHP-Worker, die Shells starten).

Bild: die Caption unter der Frage „🧭 Interpreted as (condensed): How do I mitigate
CVE-2026-109618?"; dann „Mitigations" und „Detection & Hunting".

Zoom: die „Interpreted as"-Zeile; dann „Detection & Hunting".

Caption:
```
Condense-then-retrieve: the follow-up is rewritten, generation never sees chat history
```

VO: „Ein Analyst fragt nach, nicht neu. Der Prototyp schreibt die elliptische Frage in eine
eigenständige um und zeigt das offen an. Erst die umgeschriebene Frage geht in die
unveränderte Pipeline. Die Antwort zieht jetzt die Maßnahmen aus dem CISA-Alert, inklusive
eines Hunting-Hinweises."

Risiko: Falls das Modell „carry-over" statt „condensed" wählt, ändert sich nur das Wort in
der Caption. Beides ist korrekt.

### Szene 6: MCP-Integration (1:55–2:10)

Terminal, Setup demo. Vorher aufgenommen.

```bash
CTI_RAG_SETUP=demo .venv/bin/python -m src.cti_rag.mcp.client_demo "What is CVE-2026-109618 and how severe is it?"
```

Verifiziert: 731 ms, Fußzeile „Template: VulnTriage · deterministic facts, no model
invoked". Log-Zeilen vor „Tools: cti_query_tool" im Schnitt wegschneiden.

Bild: ab „Tools: cti_query_tool", der Tool-Call, der Fact-Block, die Fußzeile.

Caption:
```
Same fact pipeline exposed as an MCP tool: 0.7 s, no LLM
```

VO: „Dieselbe Faktenpipeline steht als MCP-Tool bereit. Ein Analysten-Client bekommt die
Triage in unter einer Sekunde, ohne dass ein Modell beteiligt ist."

### Szene 7: Der Preis der lokalen Konfiguration (2:10–2:25)

Setup: demo. Extended **aus**. „New chat".

Query:
```
What is CVE-2026-109618 and how severe is it?
```

Verifiziert: lokal 46 s Generierung (End-to-End 48 s), gleiche Zitate wie Szene 3.

Aufnahme in Echtzeit, im Schnitt als Zeitraffer (8× bis 10×) mit eingeblendeter
Stoppuhr. Am Ende 2 s Echtzeit auf der Pill `generation 46xxx ms`.

Zoom: Pill `generation …ms` und das Fehlen der `extended`-Pill.

Caption:
```
Evaluated configuration: local Llama 3.1 8B, nothing leaves the machine · ~46 s (sped up)
```

VO: „Das ist die evaluierte Konfiguration: alles lokal, nichts verlässt den Rechner. Der
Preis ist Zeit, hier 46 Sekunden statt 7. Der Extended-Modus ist deshalb eine bewusste
Wahl des Nutzers und klar gekennzeichnet, nicht der Standard."

### Szene 8: Abspann (2:25–2:30)

Standbild.

```
Evaluated: Setup B, local Llama 3.1 8B, 28 queries, n = 3 (see thesis ch. 4)
Demo index: evaluated corpus + supplement fetched 2026-10-22 (not evaluated)
Extended Analysis: hosted model, user opt-in, not evaluated
```

VO: „Evaluiert wurde Setup B mit dem lokalen Modell. Der Demo-Index ist der evaluierte
Korpus plus ein dokumentiertes Supplement vom 22. Oktober. Zahlen dazu in Kapitel 4."

## 4. Schnitt

- Zooms als sanfte Ken-Burns-Bewegungen, 1,3× bis 1,6×, nie mehr. Jeder Zoom zeigt genau
  ein Element.
- Wartezeiten im Extended-Modus (7 bis 11 s) ungeschnitten lassen. Das ist das Argument.
- Zeitraffer nur in Szene 7, mit Stoppuhr-Overlay. Kein versteckter Jump-Cut irgendwo.
- Captions unten links, zwei Zeilen maximal, mindestens 4 s stehen lassen.
- Untertitel für das gesamte Voiceover einbrennen. Beamer-Audio fällt aus.
- Kein Musikbett oder sehr leise unter –30 dB.
- Export: MP4, H.264, 1080p, 30 fps, AAC. Lokal in Keynote/PowerPoint einbetten, nicht
  verlinken. Kopie auf USB-Stick. Backup-Folie pro Szene mit Screenshot.

## 5. Einsatz in der Präsentation

Empfehlung: zwei Clips statt eines Blocks.

- Clip A (Szenen 0 bis 5, ca. 1:55) direkt nach der Architektur-Folie. Vorher die Boxen
  benennen, die gleich im UI auftauchen: Router, Fact Bundle, Template, Reranker.
- Clip B (Szenen 6 bis 8, ca. 0:35) bei der Diskussion von Latenz, Privacy und Future Work.

Für die Fragerunde: Laptop mit laufender Demo-API bereithalten. Eine Live-Frage nach dem
Video ist ein starker Moment; das Risiko trägst du dann bewusst und mit vorgewärmtem System.

## 6. Verifizierte Messwerte (2026-10-22)

| Szene | Konfiguration | Ergebnis | Zeit |
|---|---|---|---|
| 1 | Setup B, lokal, templated | Abstention, kein Modellaufruf | 3,6 s |
| 2 | Setup B, Baseline ohne Retrieval | „no information … cutoff 2025-12-31" | 16 s |
| 3 | Demo, Extended | VulnTriage, HIGH — Review, 2 Zitate | 7,1 s |
| 4 | Demo, Extended | CrossSourceCompare, L1-Tabelle, 3 Zitate | 8,4 s |
| 5 | Demo, Extended, Follow-up | condensed, Mitigations-Chunk Platz 1, Detection & Hunting | 1,1 s + 11,4 s |
| 6 | Demo, MCP | VulnTriage-Fakten, kein LLM | 0,73 s |
| 7 | Demo, lokal, templated | VulnTriage, 2 Zitate | 48,6 s (Generation 46,2 s) |

Supplement-Fakten: NVD führt beide CVEs mit CVSS 3.1 9.8 (Primary, nvd@nist.gov) und
CVSS 4.0 9.2 (CNA php.net); CISA-Alert vom 21.10.2026 mit den Sections `summary` und
`mitigations`; KEV am 22.10. ohne Eintrag; CIRCL-MISP ohne Event. Der eingefrorene Korpus
(82.282 NVD-Dokumente) ist durch den CVSS-v4-Fallback unverändert.
