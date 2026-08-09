"""Plot the analyst rubric results as a small-multiples bar figure.

Reads data/interim/final_v1_aggregate.json (the aggregate of the final
evaluation series, same source as thesis/tables/rubric_final_v1.tex) and
writes thesis/images/rubric_results.pdf. Visual grammar matches
scripts/plot_generation_results.py so both figures read as one system.

    .venv/bin/python scripts/plot_rubric_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "interim" / "final_v1_aggregate.json"
DST = ROOT / "thesis" / "images" / "rubric_results.pdf"

# Same per-configuration colors as the RAGAS figure (color follows the
# entity across figures); V1 and V2 were not rubric-scored.
CONFIGS = [
    ("baseline", "V0 baseline", "#d9d9d9", None),
    ("legacy_hybrid", "V3 hybrid", "#8c8c8c", None),
    ("templated_hybrid", "V4 templated", "#1a1a1a", None),
    ("templated_hostedgen_haiku45", "V5 hosted", "#ffffff", "///"),
]

DIMENSIONS = [
    ("prioritization", "Prioritization"),
    ("actionability", "Actionability"),
    ("completeness", "Completeness"),
    ("traceability", "Traceability"),
    ("total", "Total"),
]

EDGE = "#4d4d4d"


def main() -> None:
    rubric = json.loads(SRC.read_text())["rubric"]

    plt.rcParams.update(
        {
            "font.family": "Helvetica",
            "font.size": 7.5,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(6.1, 2.0),
                             sharey=True)
    ypos = range(len(CONFIGS) - 1, -1, -1)  # V0 on top

    for ax, (dim, title) in zip(axes, DIMENSIONS):
        values = {key: rubric[key]["dimensions"][dim] for key, _, _, _ in CONFIGS}
        best = max(entry["mean"] for entry in values.values())

        for (key, _, color, hatch), y in zip(CONFIGS, ypos):
            mean, std = values[key]["mean"], values[key]["std"]
            ax.barh(y, mean, height=0.62, color=color, hatch=hatch,
                    edgecolor=EDGE if hatch else "none", linewidth=0.6,
                    zorder=3)
            ax.errorbar(mean, y, xerr=std, fmt="none", ecolor="#333333",
                        elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=4)
            # Labels: primary configuration, baseline (the contrast the
            # text discusses), and the best value. Labels that would
            # collide with the neighboring panel move inside the bar.
            if key in ("templated_hybrid", "baseline") or mean == best:
                if mean + std + 0.09 > 2.95:
                    ax.text(mean - 0.09, y, f"{mean:.2f}", va="center",
                            ha="right", fontsize=6.5, color="#1a1a1a",
                            bbox=dict(facecolor="white", edgecolor="none",
                                      pad=1.0), zorder=5)
                else:
                    ax.text(mean + std + 0.09, y, f"{mean:.2f}",
                            va="center", ha="left", fontsize=6.5,
                            color="#1a1a1a")

        ax.set_title(title, fontsize=7.5, pad=4)
        ax.set_xlim(0, 3.0)
        ax.set_ylim(-0.55, len(CONFIGS) - 0.45)
        ax.set_xticks([0, 1, 2, 3])
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.5, zorder=0)
        ax.tick_params(axis="both", length=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#bfbfbf")

    axes[0].set_yticks(list(ypos))
    axes[0].set_yticklabels([label for _, label, _, _ in CONFIGS])

    fig.subplots_adjust(left=0.115, right=0.985, top=0.85, bottom=0.13,
                        wspace=0.12)
    fig.savefig(DST)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
