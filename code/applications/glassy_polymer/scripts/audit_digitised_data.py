"""Create candidate tables and visual review panels for all source curves."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from applications.glassy_polymer.data import classify_curve, load_all_curves
from applications.glassy_polymer.data_cleaning import propose_cleaning


OUTPUT = Path(__file__).resolve().parents[1] / "results" / "data_audit"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    curves = load_all_curves()
    rows = []
    pages = []
    for curve_id, curve in enumerate(curves):
        meta = classify_curve(curve)
        decisions = propose_cleaning(curve)
        for item in decisions:
            rows.append([
                curve_id, meta.material, meta.family, curve.path.name, item.index,
                item.action, item.x_original, item.y_original, item.x_clean,
                item.y_clean, item.score, item.reason,
            ])
        pages.append((curve_id, curve, meta, decisions))
    with (OUTPUT / "candidate_actions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "curve_id", "material", "family", "file", "zero_based_index", "action",
            "x_original", "y_original", "x_clean", "y_clean", "robust_score", "reason",
        ])
        writer.writerows(rows)

    for page_start in range(0, len(pages), 12):
        fig, axes = plt.subplots(4, 3, figsize=(13, 12))
        for axis, (curve_id, curve, meta, decisions) in zip(axes.flat, pages[page_start:page_start + 12]):
            x, y = np.asarray(curve.x), np.asarray(curve.y)
            axis.plot(x, y, "o-", ms=2.2, lw=0.7, color="#2b6ea6")
            for item in decisions:
                color = "#c62828" if item.action == "delete" else "#ef6c00"
                axis.plot(item.x_original, item.y_original, "o", ms=7, mfc="none", mec=color, mew=1.5)
                axis.annotate(str(item.index), (item.x_original, item.y_original), fontsize=6)
            axis.set_title(f"{curve_id}: {meta.material}/{meta.family} n={len(x)}", fontsize=9)
            axis.grid(alpha=0.18)
        for axis in axes.flat[len(pages[page_start:page_start + 12]):]:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(OUTPUT / f"review_{page_start // 12 + 1:02d}.png", dpi=180)
        plt.close(fig)
    print(f"curves={len(curves)} candidates={len(rows)} output={OUTPUT}")


if __name__ == "__main__":
    main()
