"""Plot the RAGAS generation results as a small-multiples bar figure.

Reads data/interim/final_v1_aggregate.json (the aggregate of the final
evaluation series, same source as thesis/tables/ragas_final_v1.tex) and
writes thesis/images/generation_results.pdf.

    .venv/bin/python scripts/plot_generation_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "interim" / "final_v1_aggregate.json"
DST = ROOT / "thesis" / "images" / "generation_results.pdf"

# Configurations in matrix order (Table tab:configurations). The gray ramp
# follows the ordinal pipeline build-up V0 -> V4; V4 is the primary
# configuration (emphasis); V5 differs on the deployment axis and is
# rendered hatched instead of darker.
CONFIGS = [
    ("baseline", "V0 baseline", "#d9d9d9", None),
    ("bm25_only", "V1 BM25", "#c2c2c2", None),
    ("vector_only", "V2 vector", "#a9a9a9", None),
    ("legacy_hybrid", "V3 hybrid", "#8c8c8c", None),
    ("templated_hybrid", "V4 templated", "#1a1a1a", None),
    ("templated_hostedgen_haiku45", "V5 hosted", "#ffffff", "///"),
]

METRICS = [
    ("faithfulness", "Faithfulness"),
    ("answer_relevancy", "Answer relevancy"),
    ("answer_correctness", "Answer correctness"),
    ("context_precision", "Context precision"),
    ("context_recall", "Context recall"),
]

EDGE = "#4d4d4d"
MUTED = "#8c8c8c"


def main() -> None:
    ragas = json.loads(SRC.read_text())["ragas"]

    plt.rcParams.update(
        {
            "font.family": "Helvetica",
            "font.size": 7.5,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.1, 2.6), sharey=True)
    ypos = range(len(CONFIGS) - 1, -1, -1)  # V0 on top

    for ax, (metric, title) in zip(axes, METRICS):
        values = []
        for (key, _, _, _), y in zip(CONFIGS, ypos):
            entry = ragas[key]["metrics"].get(metric)
            values.append(entry["mean"] if entry else None)

        best = max(v for v in values if v is not None)

        for (key, _, color, hatch), y, value in zip(CONFIGS, ypos, values):
            if value is None:
                ax.text(0.02, y, "n/a", va="center", ha="left",
                        fontsize=6.5, style="italic", color=MUTED)
                continue
            std = ragas[key]["metrics"][metric]["std"]
            ax.barh(y, value, height=0.62, color=color, hatch=hatch,
                    edgecolor=EDGE if hatch else "none", linewidth=0.6,
                    zorder=3)
            ax.errorbar(value, y, xerr=std, fmt="none", ecolor="#333333",
                        elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=4)
            # Selective labels: the primary configuration and the best value.
            if key == "templated_hybrid" or value == best:
                ax.text(value + std + 0.03, y, f"{value:.3f}", va="center",
                        ha="left", fontsize=6.5, color="#1a1a1a")

        ax.set_title(title, fontsize=7.5, pad=4)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.55, len(CONFIGS) - 0.45)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "", "0.5", "", "1"])
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.5, zorder=0)
        ax.tick_params(axis="both", length=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#bfbfbf")

    axes[0].set_yticks(list(ypos))
    axes[0].set_yticklabels([label for _, label, _, _ in CONFIGS])

    fig.subplots_adjust(left=0.115, right=0.985, top=0.88, bottom=0.10,
                        wspace=0.12)
    fig.savefig(DST)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
