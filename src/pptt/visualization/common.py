from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt


def configure_figure_style(language: str) -> str:
    if language not in {"en", "zh"}:
        raise ValueError("language must be 'en' or 'zh'")
    if language == "zh":
        for font_path in font_manager.findSystemFonts():
            if "NotoSansCJK" in font_path.replace("-", ""):
                font_manager.fontManager.addfont(font_path)
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = (
        (
            "Noto Sans CJK SC",
            "Noto Sans CJK TC",
            "Noto Sans CJK JP",
            "Microsoft YaHei",
            "SimHei",
        )
        if language == "zh"
        else ("Arial", "Helvetica", "DejaVu Sans")
    )
    selected = next((name for name in candidates if name in available), None)
    if language == "zh" and selected is None:
        raise RuntimeError(
            "Chinese figures require an installed CJK font such as Noto Sans CJK SC"
        )
    if selected is None:
        selected = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": selected,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    return selected


def save_publication_figure(
    figure: plt.Figure,
    output_base: str | Path,
    *,
    dpi: int = 400,
) -> tuple[Path, Path]:
    if dpi < 300:
        raise ValueError("publication figures require at least 300 dpi")
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    figure.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={"Creator": "PPTT reproducible figure pipeline"},
    )
    plt.close(figure)
    return png_path, pdf_path
