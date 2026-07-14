"""Generate a publication-style framework overview figure.

The figure is deterministic because it contains exact method labels. It replaces
the earlier minimal pipeline graphic used by the LaTeX manuscript.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_ROOT = Path("stage1_explanation_suite_20260517/results/e8_paper_figures/main")


def box(ax, xy, wh, text, fc="#ffffff", ec="#2b2b2b", lw=1.1, size=8.5):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color="#111111")
    return patch


def arrow(ax, start, end, color="#3a3a3a", lw=1.25, rad=0.0, style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    return patch


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "Evidence-guided feature-tensor graph analysis for Stage 1 semantic dynamics",
        ha="center",
        va="center",
        fontsize=12.5,
        weight="bold",
    )

    # Column headers.
    ax.text(0.16, 0.875, "Directed feature-tensor graph", ha="center", fontsize=9.5, weight="bold")
    ax.text(0.49, 0.875, "Node-level measurements", ha="center", fontsize=9.5, weight="bold")
    ax.text(0.81, 0.875, "Evidence synthesis", ha="center", fontsize=9.5, weight="bold")

    # Network graph column.
    nodes = {
        "down1": (0.055, 0.68),
        "down2": (0.145, 0.68),
        "down3": (0.235, 0.68),
        "down4": (0.145, 0.51),
        "up1": (0.055, 0.31),
        "up2": (0.145, 0.31),
        "up3": (0.235, 0.31),
        "up4": (0.325, 0.31),
    }
    for name, (x, y) in nodes.items():
        color = "#e8f3ff" if name.startswith("down") else "#fff1dc"
        box(ax, (x, y), (0.072, 0.07), name, fc=color, ec="#426b8a", size=8)
    # Encoder and decoder arrows.
    arrow(ax, (0.127, 0.715), (0.145, 0.715))
    arrow(ax, (0.217, 0.715), (0.235, 0.715))
    arrow(ax, (0.177, 0.68), (0.177, 0.58))
    arrow(ax, (0.145, 0.51), (0.092, 0.38))
    arrow(ax, (0.127, 0.345), (0.145, 0.345))
    arrow(ax, (0.217, 0.345), (0.235, 0.345))
    arrow(ax, (0.307, 0.345), (0.325, 0.345))
    # Skip connections.
    arrow(ax, (0.09, 0.68), (0.09, 0.38), color="#8a5a00", lw=1.0, rad=0.18)
    arrow(ax, (0.18, 0.68), (0.18, 0.38), color="#8a5a00", lw=1.0, rad=0.12)
    arrow(ax, (0.27, 0.68), (0.27, 0.38), color="#8a5a00", lw=1.0, rad=0.08)
    ax.text(0.145, 0.225, "nodes = named tensors\nedges = transforms / skips", ha="center", fontsize=8)

    # Measurement column.
    measurements = [
        ("Semantic\nconsistency", 0.39, 0.68, "#eaf7ea"),
        ("CAM\nresponse", 0.49, 0.68, "#f1ecff"),
        ("Raw feature\nresponse", 0.59, 0.68, "#eaf6f8"),
        ("Same-case\ntransition deltas", 0.44, 0.43, "#fff4e5"),
        ("Response-guided\nperturbation", 0.55, 0.43, "#ffecec"),
        ("Sensitivity and\naggregation robustness", 0.495, 0.22, "#f5f5f5"),
    ]
    for text, x, y, fc in measurements:
        box(ax, (x, y), (0.095, 0.105), text, fc=fc, ec="#4b5563", size=8)

    # From graph to measurements.
    arrow(ax, (0.37, 0.62), (0.39, 0.725), color="#4b5563")
    arrow(ax, (0.37, 0.55), (0.49, 0.725), color="#4b5563")
    arrow(ax, (0.37, 0.48), (0.59, 0.725), color="#4b5563")
    arrow(ax, (0.37, 0.42), (0.44, 0.485), color="#4b5563")
    arrow(ax, (0.37, 0.36), (0.55, 0.485), color="#4b5563")

    # Evidence synthesis column.
    evidence = [
        ("Stage 1 curve\nand increments", 0.69, 0.68, "#e8f3ff"),
        ("Visual trajectories\n+ numeric readouts", 0.835, 0.68, "#f1ecff"),
        ("Case-level\nscatter alignment", 0.69, 0.45, "#fff4e5"),
        ("Task relevance\nunder perturbation", 0.835, 0.45, "#ffecec"),
        ("Timing explanation:\nearly formation\nvs late recovery", 0.755, 0.20, "#eaf7ea"),
    ]
    for text, x, y, fc in evidence:
        width = 0.13 if y > 0.3 else 0.18
        height = 0.12 if y > 0.3 else 0.14
        box(ax, (x, y), (width, height), text, fc=fc, ec="#374151", size=7.6)

    arrow(ax, (0.485, 0.735), (0.69, 0.735), color="#374151")
    arrow(ax, (0.585, 0.735), (0.835, 0.735), color="#374151")
    arrow(ax, (0.535, 0.50), (0.69, 0.505), color="#374151")
    arrow(ax, (0.645, 0.50), (0.835, 0.505), color="#374151")
    arrow(ax, (0.59, 0.275), (0.76, 0.34), color="#374151")
    arrow(ax, (0.755, 0.45), (0.80, 0.34), color="#374151", rad=-0.05)
    arrow(ax, (0.895, 0.45), (0.835, 0.34), color="#374151", rad=0.05)

    # Bottom note.
    ax.text(
        0.5,
        0.055,
        "Output: a timing explanation supported by semantic readout, response alignment, perturbation effects, and robustness checks.",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#333333",
    )

    out = OUT_ROOT / "fig2_evidence_pipeline"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(out.with_suffix(".png"))
    print(out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
