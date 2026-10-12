"""Archive pathwise V4 root, control, state-domain, and safeguard diagnostics."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from applications.glassy_polymer.canonical import CANONICAL_OPERATOR_VERSION
from applications.glassy_polymer.constitutive import pc_parameters, pmma_parameters
from applications.glassy_polymer.experiments import build_experiments, predict_with_audit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "v4_all_path_audit"
FULL_WIDTH_IN = 182.0 / 25.4

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 8.0,
})


def scalar(value):
    array = np.asarray(value)
    if array.dtype == np.bool_:
        return bool(array)
    if np.issubdtype(array.dtype, np.integer):
        return int(array)
    return float(array)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for experiment in build_experiments():
        params = pmma_parameters() if experiment.material == "PMMA" else pc_parameters()
        prediction, audit = predict_with_audit(experiment, params)
        response = np.asarray(prediction)
        row = {
            "curve_id": experiment.curve_id,
            "material": experiment.material,
            "family": experiment.family,
            "target": experiment.target,
            "rate": experiment.rate,
            "split": experiment.split,
            "observations": len(experiment.x),
            "prediction_finite": bool(np.all(np.isfinite(response))),
            "prediction_minimum": float(np.nanmin(response)),
            "prediction_maximum": float(np.nanmax(response)),
            **{name: scalar(value) for name, value in audit._asdict().items()},
        }
        rows.append(row)
        print(
            experiment.curve_id, experiment.family,
            "finite=", row["prediction_finite"] and row["finite"],
            "increments=", row["increments"],
            "root=", f"{row['max_abs_root_residual']:.3e}",
            "control=", f"{row['max_control_residual']:.3e}",
            flush=True,
        )

    with (OUTPUT / "all_path_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    finite = [row for row in rows if row["prediction_finite"] and row["finite"]]
    summary = {
        "operator_version": CANONICAL_OPERATOR_VERSION,
        "paths": len(rows),
        "finite_paths": len(finite),
        "all_paths_finite": len(finite) == len(rows),
        "maximum_root_residual": max(row["max_abs_root_residual"] for row in rows),
        "maximum_control_residual": max(row["max_control_residual"] for row in rows),
        "minimum_A_eigenvalue": min(
            min(row["min_eigenvalue_A0"], row["min_eigenvalue_A1"],
                row["min_eigenvalue_A2"])
            for row in rows
        ),
        "minimum_total_resistance": min(row["minimum_total_resistance"] for row in rows),
        "minimum_gent_margin": min(row["minimum_gent_margin"] for row in rows),
        "matrix_log_floor_activations": sum(
            row["matrix_log_floor_activations"] for row in rows
        ),
        "gent_floor_activations": sum(row["gent_floor_activations"] for row in rows),
        "maximum_control_condition_number": max(
            row["max_control_condition_number"] for row in rows
            if np.isfinite(row["max_control_condition_number"])
        ),
        "maximum_root_iterations": max(row["maximum_root_iterations"] for row in rows),
        "root_fallback_activations": sum(
            row["root_fallback_activations"] for row in rows
        ),
    }
    (OUTPUT / "all_path_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_diagnostics(rows, summary)
    print(json.dumps(summary, indent=2))


def plot_diagnostics(rows, summary) -> None:
    """Render the path diagnostics without evaluating a material history."""
    x = np.arange(len(rows))
    labels = [row["curve_id"] for row in rows]
    family_colors = {
        "creep": "#1b9e77", "recovery": "#d95f02",
        "load_unload_recovery": "#7570b3", "load_unload": "#e7298a",
        "reverse": "#66a61e", "cyclic": "#e6ab02", "relaxation": "#1f78b4",
    }
    family_labels = {
        "creep": "creep", "recovery": "recovery",
        "load_unload_recovery": "L/U + recovery",
        "load_unload": "load/unload", "reverse": "reversal",
        "cyclic": "cycling", "relaxation": "relaxation",
    }
    colors = [family_colors[row["family"]] for row in rows]
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 5.35), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(0.23, 1.0, 1.0))
    legend_axis = fig.add_subplot(grid[0, :])
    legend_axis.axis("off")
    axes = np.asarray([[fig.add_subplot(grid[i + 1, j]) for j in range(3)]
                       for i in range(2)])
    axes[0, 0].scatter(x, [max(row["max_abs_root_residual"], 1e-16) for row in rows],
                       s=14, c=colors)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("max root residual\n(branch units)")
    axes[0, 1].scatter(x, [max(row["max_control_residual"], 1e-16) for row in rows],
                       s=14, c=colors, marker="s")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("max control residual (MPa)")
    axes[0, 2].scatter(
        x,
        [row["max_control_condition_number"] if np.isfinite(row["max_control_condition_number"])
         else np.nan for row in rows],
        s=14, c=colors, marker="D",
    )
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_ylabel("max control-Jacobian condition")
    axes[1, 0].scatter(x, [min(row["min_eigenvalue_A0"], row["min_eigenvalue_A1"],
                               row["min_eigenvalue_A2"]) for row in rows],
                       s=14, c=colors, marker="^")
    axes[1, 0].set_ylabel("minimum eigenvalue of $A_i$")
    axes[1, 1].scatter(x, [row["minimum_gent_margin"] for row in rows],
                       s=14, c=colors, marker="d")
    axes[1, 1].set_ylabel("minimum Gent margin")
    safeguard_counts = [
        row["matrix_log_floor_activations"] + row["gent_floor_activations"]
        + row["root_fallback_activations"] for row in rows
    ]
    axes[1, 2].scatter(x, safeguard_counts, s=14, c=colors, marker="v")
    axes[1, 2].set_ylabel("safeguard/fallback activations")
    axes[1, 2].set_ylim(-0.08, max(1.0, max(safeguard_counts) + 0.5))
    axes[1, 2].text(
        0.03, 0.90, f"fixed root cap = {summary['maximum_root_iterations']}\n"
        f"fallbacks = {summary['root_fallback_activations']}",
        transform=axes[1, 2].transAxes, va="top", fontsize=8.0,
    )
    for index, axis in enumerate(axes.flat):
        axis.set_title(f"({chr(97 + index)})", loc="left", fontweight="bold")
        tick_indices = np.unique(np.r_[np.arange(0, len(rows), 4), len(rows) - 1])
        axis.set_xticks(x[tick_indices], [labels[index] for index in tick_indices],
                       rotation=90)
        axis.grid(alpha=0.25)
        axis.set_xlabel("frozen curve identifier")
    handles = [
        mpl.lines.Line2D([], [], marker="o", ls="", color=color,
                         label=family_labels[family])
        for family, color in family_colors.items()
        if any(row["family"] == family for row in rows)
    ]
    legend_axis.legend(
        handles=handles, loc="center", ncol=4, frameon=False,
        fontsize=8.0, columnspacing=1.4,
    )
    fig.savefig(OUTPUT / "v4_all_path_audit.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT / "v4_all_path_audit.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def replot_archived() -> None:
    with (OUTPUT / "all_path_diagnostics.csv").open() as handle:
        source_rows = list(csv.DictReader(handle))
    text_fields = {"curve_id", "material", "family", "split", "prediction_finite", "finite"}
    data = [{key: value if key in text_fields or value == "" else float(value)
             for key, value in row.items()} for row in source_rows]
    summary = json.loads((OUTPUT / "all_path_summary.json").read_text())
    plot_diagnostics(data, summary)
    print(f"Replotted {len(data)} archived path diagnostics", flush=True)


if __name__ == "__main__":
    replot_archived() if "--plot-only" in sys.argv[1:] else main()
