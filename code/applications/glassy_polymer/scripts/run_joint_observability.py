"""Joint local observability audit for the 12-parameter PMMA update block.

The two six-parameter analyses are useful mechanistic conditionals, but they do
not expose cross-block compensation.  This script evaluates both blocks on one
common, protocol-balanced observation vector and exports the joint spectrum,
weakest modes, and normalized Jacobian-column Gram matrix used by the paper.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from applications.glassy_polymer.calibration import replace_log_ratio_block
from applications.glassy_polymer.constitutive import ConstitutiveRegularization, pmma_parameters
from applications.glassy_polymer.experiments import build_experiments, predict
from applications.glassy_polymer.objectives import path_integration_weights


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "joint_observability"
NAMES = (
    "gamma0", "m0", "s0", "h2", "s1", "s2",
    "m_ve", "zeta1", "zeta2", "h3", "b1_sat", "b2_sat",
)
REGULARIZATION = ConstitutiveRegularization(0.02, 4.0e-7)

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
})


def selected_experiments():
    selected = []
    for item in build_experiments("PMMA"):
        include = (
            (item.family == "load_unload" and item.target in {0.2, 1.05})
            or (item.family == "creep" and item.target in {42.0, 74.0, 94.0})
            or (item.family == "recovery" and item.target == 0.2)
            or (item.family in {"reverse", "cyclic"} and item.target in {0.1, 0.165})
        )
        if include:
            selected.append(item)
    return sorted(selected, key=lambda item: (item.family, item.target or 0.0))


def record_balancing_factors(experiments):
    """Equal family and physical-record mass, with observables nested in records."""
    descriptors = []
    for item in experiments:
        family = "load_unload_recovery" if item.compound_test_id else item.family
        record = item.compound_test_id or item.curve_id
        descriptors.append((item.curve_id, family, record))
    families = sorted({family for _, family, _ in descriptors})
    factors = {}
    for family in families:
        records = sorted({record for _, local_family, record in descriptors
                          if local_family == family})
        for record in records:
            curves = [curve for curve, local_family, local_record in descriptors
                      if local_family == family and local_record == record]
            factor = 1.0 / np.sqrt(len(families) * len(records) * len(curves))
            for curve in curves:
                factors[curve] = factor
    return factors


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base = pmma_parameters()  # rejected candidate never enters this audit
    experiments = selected_experiments()
    balance_factors = record_balancing_factors(experiments)

    theta = jnp.zeros(len(NAMES), dtype=jnp.float64)

    def observations(values):
        params = replace_log_ratio_block(base, NAMES, values)
        blocks = []
        for experiment in experiments:
            output_scale = (
                max(float(np.max(np.abs(experiment.y))), 0.02)
                if experiment.family in {"creep", "recovery"}
                else 110.0
            )
            point_weight = jnp.sqrt(jnp.asarray(path_integration_weights(experiment.x)))
            blocks.append(
                balance_factors[experiment.curve_id] * point_weight
                * predict(experiment, params, REGULARIZATION, True) / output_scale
            )
        return jnp.concatenate(blocks)

    jacobian_ad = np.asarray(jax.jit(jax.jacfwd(observations))(theta))
    if not np.all(np.isfinite(jacobian_ad)):
        raise RuntimeError("Non-finite joint AD Jacobian")

    step = 1.0e-5
    observation_jit = jax.jit(observations)
    columns = []
    for column in range(len(NAMES)):
        direction = np.zeros(len(NAMES))
        direction[column] = step
        plus = np.asarray(observation_jit(theta + direction))
        minus = np.asarray(observation_jit(theta - direction))
        columns.append((plus - minus) / (2.0 * step))
    jacobian_fd = np.column_stack(columns)
    relative_error = np.linalg.norm(jacobian_ad - jacobian_fd) / max(
        np.linalg.norm(jacobian_fd), 1.0e-30
    )

    _, singular, vt = np.linalg.svd(jacobian_ad, full_matrices=False)
    norms = np.linalg.norm(jacobian_ad, axis=0)
    gram = jacobian_ad.T @ jacobian_ad
    gram /= np.maximum(np.outer(norms, norms), 1.0e-30)

    np.save(OUTPUT / "joint_12_parameter_jacobian_ad.npy", jacobian_ad)
    np.save(OUTPUT / "joint_12_parameter_jacobian_fd.npy", jacobian_fd)
    with (OUTPUT / "joint_12_parameter_singular_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "singular_value", "relative", "ad_fd_relative_error"))
        for index, value in enumerate(singular):
            writer.writerow((index + 1, value, value / singular[0], relative_error))
    with (OUTPUT / "joint_12_parameter_right_singular_vectors.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", *NAMES))
        for index, row in enumerate(vt):
            writer.writerow((index + 1, *row))
    with (OUTPUT / "joint_12_parameter_column_norms.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("parameter", "column_norm"))
        writer.writerows(zip(NAMES, norms))
    with (OUTPUT / "joint_12_parameter_gram.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("parameter", *NAMES))
        for name, row in zip(NAMES, gram):
            writer.writerow((name, *row))

    labels = [r"$\dot\gamma_0$", r"$m_0$", r"$s_0$", r"$h_2$", r"$s_1$", r"$s_2$",
              r"$m_{ve}$", r"$\zeta_1$", r"$\zeta_2$", r"$h_3$", r"$b_1^{sat}$", r"$b_2^{sat}$"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), constrained_layout=True)
    axes[0].semilogy(np.arange(1, 13), singular, "o-", color="#1f4e79", lw=1.5, ms=4)
    axes[0].set(xlabel="sensitivity mode", ylabel="absolute singular value")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].text(0.02, 0.98, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold")
    axes[0].text(0.96, 0.94, rf"$\kappa={singular[0] / singular[-1]:.1f}$",
                 transform=axes[0].transAxes, ha="right", va="top")

    weakest = vt[-2:]
    x = np.arange(len(NAMES))
    axes[1].bar(x - 0.18, weakest[0], width=0.36, label="mode 11", color="#4c78a8")
    axes[1].bar(x + 0.18, weakest[1], width=0.36, label="mode 12", color="#f58518")
    axes[1].axhline(0.0, color="0.25", lw=0.7)
    axes[1].set_xticks(x, labels, rotation=55, ha="right")
    axes[1].set_ylabel("right-singular-vector coefficient")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].text(0.02, 0.98, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold")

    image = axes[2].imshow(gram, vmin=-1, vmax=1, cmap="coolwarm")
    axes[2].set_xticks(x, labels, rotation=55, ha="right")
    axes[2].set_yticks(x, labels)
    axes[2].set_title("sensitivity-column cosine matrix")
    axes[2].text(0.02, 0.98, "(c)", transform=axes[2].transAxes, va="top", fontweight="bold")
    fig.colorbar(image, ax=axes[2], shrink=0.82, label="column cosine")
    fig.savefig(OUTPUT / "joint_12_parameter_observability.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "joint_12_parameter_observability.png", dpi=450, bbox_inches="tight")
    plt.close(fig)

    print("AD-FD relative error", relative_error)
    print("singular values", singular)
    print("condition", singular[0] / singular[-1])


if __name__ == "__main__":
    main()
