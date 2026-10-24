# Demo Video: Shot-by-Shot Script (Master's Defense)

Status 2026-10-22. Every query, timing and output in this script was verified on that day
against the demo index (`CTI_RAG_SETUP=demo`) and the frozen Setup-B index. The defense is
held in English (English slides, English thesis), so captions (lower thirds) and voiceover
are both English.

Total length: about 2:30 min. Hard ceiling 3:00 min.

## 1. Narrative

CVE-2026-109618 / CVE-2026-109619 ("php2shell") leaked on 2026-10-20, were patched on
2026-10-21 and were recorded by NVD and CISA the same day. No generation model in the
prototype can know them. Every correct statement in the video therefore demonstrably comes
from retrieval.

Three acts:

| Act | Scenes | Claim |
|---|---|---|
| Before | 1, 2 | The system does not guess. No evidence: abstention. Model alone: knows nothing. |
| After | 3, 4, 5 | With fresh evidence: deterministic triage, cross-source comparison, conversation. |
| Limits | 6, 7, 8 | Integration (MCP), honest latency of the evaluated local configuration, declaration. |

Mandatory declaration (scene 8, and in the voiceover of scene 3): the evaluation covers
Setup B with local Llama 3.1 8B. The demo index is the evaluated corpus plus a supplement
fetched on 2026-10-22. Extended Analysis is a non-evaluated convenience mode.

Do not claim that FH Technikum Wien was affected. Publicly documented are TU Wien,
University of Vienna, WU, BOKU, MedUni Vienna and JKU. Say "Austrian universities".

## 2. Pre-flight

### 2.1 Data and index

```bash
make demo-supplement
```

```bash
make demo-index
```

Both are idempotent. Run once before recording, then check:

- KEV status. If either CVE has entered the CISA KEV catalog, the triage signal moves from
  "HIGH — Review" to "CRITICAL — Immediate attention". Use voiceover variant B in scene 3.
  Check with:

```bash
curl -s https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json | grep -c "CVE-2026-10961"
```

  If the count is above 0: save the fresh KEV export as
  `data/raw_demo/cisa_kev/known_exploited_vulnerabilities.json` and run `make demo-index`
  again. The parser accepts the format unchanged.

### 2.2 Backend and UI

The UI talks to the API on port 8000. Setup B and demo are separate API processes.
Recommendation: record scenes 1 and 2 with Setup B first, then restart the API.

Setup B (scenes 1, 2):

```bash
make api SETUP=b
```

```bash
make ui SETUP=b
```

Demo (scenes 3 to 7):

```bash
make api SETUP=demo
```

```bash
make ui SETUP=demo
```

Alternative without restart: a second API on port 8001 and a second UI on port 8502 with
`CTI_RAG_API_URL=http://localhost:8001`. Doubles memory for embedding model, reranker and
Chroma; feasible on the M4 with 24 GB.

Warm up before each recording block: run any query once so models and Chroma are in memory.
Then click "＋ New chat" in the sidebar.

### 2.3 Picture

- Separate, empty browser profile. No bookmarks bar, no tabs, no extensions.
- Window 1920×1080, browser zoom 125 %. Dark theme and `toolbarMode = "minimal"` are
  already set in `.streamlit/config.toml`.
- Keep the sidebar open: "Setup: b" and "Setup: demo" are evidence.
- Do Not Disturb on, Dock hidden, desktop empty.
- Paste queries with Cmd+V, never type them.
- One clip per scene, at least two takes each.
- Two seconds of stillness after every result, cursor out of frame.
- Terminal for scenes 2 and 6: 18 pt font, shortened prompt, no paths with real names,
  no `env` output, no API keys.

### 2.4 Read before you click on

In Extended mode a hosted model writes the L2 sections. In one test run "Source Agreement"
stated that NVD and CISA both assign CVSS 9.8, although the CISA alert carries no score.
The L1 block is deterministic and was correct in every run; L2 can wobble. So: approve a
take only after reading the L2 sentences. If one makes a claim the cited source does not
support, record again. When in doubt, keep the L1 blocks on screen longer.

## 3. Shot list

Notation: **Query** is the exact text to paste. **Frame** is what must be visible.
**Zoom** are post-production zooms. **Caption** is the lower-third overlay. **VO** is the
voiceover. Times are targets in the final cut.

### Scene 0: Title card (0:00–0:05)

Still frame, no UI.

Caption:
```
CTI-RAG Analyst Support — prototype demo
Query: php2shell (CVE-2026-109618 / CVE-2026-109619), disclosed 2026-10-20
```

VO: "One SOC analyst, five sources, one new critical vulnerability. We ask the prototype
about php2shell, two days after disclosure."

### Scene 1: Abstention on the evaluated index (0:05–0:25)

Setup: B. Sidebar: Retrieval Mode `hybrid`, Pipeline `Templated (Phase 2)`,
Extended Analysis **off**.

Query:
```
What is CVE-2026-109618 and how severe is it?
```

Verified: abstention after 3.6 s, generation 0 ms, no model call.

Frame: answer "Insufficient evidence in the retrieved context to answer this question.
Missing: The retrieved context does not mention cve-2026-109618." Below it the timing pills
with `generation 0ms`. Open the expander "Retrieved context — not cited (5)": five unrelated
CVEs with low scores.

Zoom: sidebar "Setup: b", then the pill `generation 0ms`, then the opened expander.

Caption:
```
Evaluated corpus (snapshot 2025-12-31): no evidence → explicit abstention, no model call
```

VO: "The evaluated corpus ends on 31 December 2025. The system finds no evidence and
declines before any model is called. It still shows what it retrieved, and marks it as
not used."

Risk: none. Deterministic, under 4 s.

### Scene 2: Model without retrieval (0:25–0:35)

Terminal, Setup B. Optional but strong. Recorded beforehand, cut to 10 s.

```bash
CTI_RAG_SETUP=b .venv/bin/python main.py baseline "What is CVE-2026-109618 and how severe is it?"
```

Verified: 16 s of local generation. Answer in substance: "There is no information available
on CVE-2026-109618 … knowledge cutoff is 2025-12-31."

Frame: only the lines "Mode: BASELINE (no retrieval)" and the Summary paragraph.

Caption:
```
Same local model, no retrieval: nothing to say
```

VO: "The language model on its own does not know the vulnerability. Everything that follows
has to come from the sources."

Risk: in another take the baseline may hallucinate instead of admitting ignorance. Both are
usable; a hallucination is actually the stronger argument, but then needs the caption
"fabricated, no sources". Read the take, then decide.

### Scene 3: VulnTriage on the demo index (0:35–1:10)

Setup: demo. Sidebar: `hybrid`, `Templated (Phase 2)`, Extended Analysis **on**
(sidebar notice "Data leaves the local machine" visible).

Hard cut from scene 2, preceded by a 2 s intertitle:
```
Index refreshed: evaluated corpus + NVD records and CISA alert of 2026-10-21
```

Query (identical to scene 1):
```
What is CVE-2026-109618 and how severe is it?
```

Verified: 7 s, template VulnTriage, cites `nvd_CVE-2026-109618` and
`cisa_advisory_CISA-ALERT-2026-10-21-PHP2SHELL_summary`.

Frame, in this order:
1. Pills: `extended` pill with model name, `template · VulnTriage`, timings.
2. L1 block: "Triage Signal: HIGH — Review", "Rationale: CVSS 9.8 + no KEV listing, no
   ransomware evidence", "Listed in CISA KEV: no".
3. Open the expander "Routing decision": `template: VulnTriage`, `primary_entities:
   ['CVE-2026-109618']`.
4. Scroll down briefly: "Exploitation Context", "Mitigations", "Gaps".
5. Expander "Cited Sources (2)".

Zoom: sidebar "Setup: demo" and the notice "Data leaves the local machine"; then the triage
signal; then the routing decision.

Caption:
```
L1: deterministic fact bundle (no model) · L2: generated, every sentence cited
```

VO (variant A, CVEs not in KEV): "With fresh evidence the prototype routes the question to
the VulnTriage template. The upper block is deterministic: CVSS 9.8, no KEV entry, signal
'High, Review'. It would only become Critical with confirmed exploitation. Below, a model
writes the context, every sentence with its source. This run uses Extended mode with a
hosted model. The evaluated configuration is local; more on that in a moment."

VO (variant B, if listed in KEV meanwhile): "… CVSS 9.8 and a KEV entry, signal 'Critical,
Immediate attention'. …"

Risk: read the L2 sentences (see 2.4). The "Gaps" section is intended: it names what the
sources do not provide (IoCs, detection rules). Keep it in frame.

### Scene 4: CrossSourceCompare (1:10–1:35)

Setup: demo. Extended **on**. Same chat or "New chat", both work because the query is
standalone.

Query:
```
What do NVD and CISA say about the php2shell vulnerabilities CVE-2026-109618 and CVE-2026-109619?
```

Verified: 8.4 s, template CrossSourceCompare, L1 table with both CVEs, cites both NVD
records and the CISA alert.

Frame: L1 "Comparison Table" (CVE, CVSS, KEV Listed, Ransomware, Vendor), below it
"Top Matches by Triage Signal", then "Source Agreement" and "Cross-Source Facts".

Zoom: the table; then the sentence in "Cross-Source Facts" that puts the two different root
causes side by side (longjmp/OOM vs. stale global state).

Caption:
```
Two CVEs, three sources (NVD, CISA alert, KEV check): one deterministic table
```

VO: "Two CVEs, three sources. The table is built deterministically from the facts: same
severity, neither in KEV yet. The model adds where the sources differ: NVD describes two
distinct root causes, CISA groups both under php2shell."

Risk: read L2 "Source Agreement" (2.4). The line "Entities with data from NVD and CISA
KEV: 0" is correct and may stay.

### Scene 5: Conversational follow-up (1:35–1:55)

Setup: demo. Extended **on**. Record in the same chat right after scene 3 (the scene needs
its predecessor). Practical: scenes 3 and 5 in one take, scene 4 separately.

Query:
```
and how do I mitigate it?
```

Verified: condensed in 1.1 s to "How do I mitigate CVE-2026-109618?", answer after 11 s;
`cisa_advisory_…_mitigations` ranked first, sections "Mitigations" (five items) and
"Detection & Hunting" (PHP workers launching shells).

Frame: the caption under the question "🧭 Interpreted as (condensed): How do I mitigate
CVE-2026-109618?"; then "Mitigations" and "Detection & Hunting".

Zoom: the "Interpreted as" line; then "Detection & Hunting".

Caption:
```
Condense-then-retrieve: the follow-up is rewritten, generation never sees chat history
```

VO: "An analyst follows up, they do not start over. The prototype rewrites the elliptical
question into a standalone one and shows that openly. Only the rewritten question enters
the unchanged pipeline. The answer now pulls the measures from the CISA alert, including a
hunting hint."

Risk: if the model picks "carry-over" instead of "condensed", only that word in the caption
changes. Both are correct.

### Scene 6: MCP integration (1:55–2:10)

Terminal, Setup demo. Recorded beforehand.

```bash
CTI_RAG_SETUP=demo .venv/bin/python -m src.cti_rag.mcp.client_demo "What is CVE-2026-109618 and how severe is it?"
```

Verified: 731 ms, footer "Template: VulnTriage · deterministic facts, no model invoked".
Cut the log lines before "Tools: cti_query_tool".

Frame: from "Tools: cti_query_tool", the tool call, the fact block, the footer.

Caption:
```
Same fact pipeline exposed as an MCP tool: 0.7 s, no LLM
```

VO: "The same fact pipeline is available as an MCP tool. An analyst client gets the triage
in under a second, with no model involved."

### Scene 7: The price of the local configuration (2:10–2:25)

Setup: demo. Extended **off**. "New chat".

Query:
```
What is CVE-2026-109618 and how severe is it?
```

Verified: 46 s of local generation (48 s end to end), same citations as scene 3.

Record in real time, present as time-lapse (8× to 10×) with a stopwatch overlay. End with
2 s of real time on the pill `generation 46xxx ms`.

Zoom: the pill `generation …ms` and the absence of the `extended` pill.

Caption:
```
Evaluated configuration: local Llama 3.1 8B, nothing leaves the machine · ~46 s (sped up)
```

VO: "This is the evaluated configuration: everything local, nothing leaves the machine. The
price is time, here 46 seconds instead of 7. That is why Extended mode is a deliberate,
clearly labeled choice by the user, not the default."

### Scene 8: End card (2:25–2:30)

Still frame.

```
Evaluated: Setup B, local Llama 3.1 8B, 28 queries, n = 3 (see thesis ch. 4)
Demo index: evaluated corpus + supplement fetched 2026-10-22 (not evaluated)
Extended Analysis: hosted model, user opt-in, not evaluated
```

VO: "The evaluation covers Setup B with the local model. The demo index is the evaluated
corpus plus a documented supplement from 22 October. Numbers are in chapter 4."

## 4. Editing

- Zooms as gentle Ken Burns moves, 1.3× to 1.6×, never more. Each zoom shows exactly one
  element.
- Leave Extended-mode waits (7 to 11 s) uncut. That is the argument.
- Time-lapse only in scene 7, with the stopwatch overlay. No hidden jump cut anywhere.
- Captions bottom left, two lines maximum, on screen for at least 4 s.
- Burn in subtitles for the entire voiceover. Projector audio fails.
- No music bed, or very quiet below –30 dB.
- Export: MP4, H.264, 1080p, 30 fps, AAC. Embed locally in Keynote/PowerPoint, do not link.
  Copy on a USB stick. One backup slide per scene with a screenshot.

## 5. Use in the presentation

Recommendation: two clips instead of one block.

- Clip A (scenes 0 to 5, about 1:55) right after the architecture slide. Name the boxes
  beforehand that will appear in the UI: router, fact bundle, template, reranker.
- Clip B (scenes 6 to 8, about 0:35) when discussing latency, privacy and future work.

For the Q&A: keep the laptop with the demo API running. A live question after the video is
a strong moment; the risk is then deliberate and the system is warm.

## 6. Verified measurements (2026-10-22)

| Scene | Configuration | Result | Time |
|---|---|---|---|
| 1 | Setup B, local, templated | abstention, no model call | 3.6 s |
| 2 | Setup B, baseline without retrieval | "no information … cutoff 2025-12-31" | 16 s |
| 3 | demo, Extended | VulnTriage, HIGH — Review, 2 citations | 7.1 s |
| 4 | demo, Extended | CrossSourceCompare, L1 table, 3 citations | 8.4 s |
| 5 | demo, Extended, follow-up | condensed, mitigations chunk ranked first, Detection & Hunting | 1.1 s + 11.4 s |
| 6 | demo, MCP | VulnTriage facts, no LLM | 0.73 s |
| 7 | demo, local, templated | VulnTriage, 2 citations | 48.6 s (generation 46.2 s) |

Supplement facts: NVD lists both CVEs with CVSS 3.1 9.8 (Primary, nvd@nist.gov) and
CVSS 4.0 9.2 (CNA php.net); CISA alert of 2026-10-21 with sections `summary` and
`mitigations`; KEV without an entry on 2026-10-22; CIRCL MISP without an event. The frozen
corpus (82,282 NVD documents) is unchanged by the CVSS v4 fallback.
