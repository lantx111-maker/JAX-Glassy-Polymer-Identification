"""Auditable cleaning of digitised two-column experimental curves.

No source workbook is modified.  The detector proposes point-level actions;
accepted actions are materialised into a derived dataset with a complete log.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from .data import Curve


@dataclass(frozen=True)
class PointDecision:
    index: int
    action: str
    x_original: float
    y_original: float
    x_clean: float | None
    y_clean: float | None
    score: float
    reason: str


def _robust_scale(values: np.ndarray, floor: float) -> float:
    median = np.nanmedian(values)
    return max(1.4826 * np.nanmedian(np.abs(values - median)), floor)


def _turning_points(x: np.ndarray) -> set[int]:
    """Protect genuine loading reversals from an outlier detector."""
    dx = np.diff(x)
    tolerance = max(np.ptp(x) * 1.0e-5, 1.0e-12)
    signs = np.sign(np.where(np.abs(dx) > tolerance, dx, 0.0))
    turns: set[int] = {0, len(x) - 1}
    last = 0.0
    for i, sign in enumerate(signs):
        if sign and last and sign != last:
            turns.update(range(max(0, i - 2), min(len(x), i + 4)))
        if sign:
            last = sign
    return turns


def propose_cleaning(curve: Curve) -> list[PointDecision]:
    """Flag isolated digitisation slips while preserving physical features.

    Delete: isolated, gross two-coordinate departure (score >= 9), or a local
    coordinate reversal that cannot be a protected loading reversal.
    Adjust: moderate isolated departure (5 <= score < 9); replace only y by a
    local robust polynomial estimate.  Everything else is retained verbatim.
    """
    x = np.asarray(curve.x, dtype=float)
    y = np.asarray(curve.y, dtype=float)
    n = len(x)
    if n < 7:
        return []
    window = min(11, n if n % 2 else n - 1)
    window = max(window, 5)
    y_reference = savgol_filter(y, window, 2, mode="interp")
    residual = y - y_reference
    y_floor = max(np.ptp(y) * 2.0e-3, 1.0e-8)
    scale = _robust_scale(residual, y_floor)
    score = np.abs(residual - np.median(residual)) / scale
    protected = _turning_points(x)
    decisions: list[PointDecision] = []
    for i in range(2, n - 2):
        if i in protected:
            continue
        local_scores = score[i - 2:i + 3]
        isolated = score[i] >= 1.6 * max(np.max(np.delete(local_scores, 2)), 1.0)
        left_dx, right_dx = x[i] - x[i - 1], x[i + 1] - x[i]
        reversal = left_dx * right_dx < 0 and abs(left_dx) + abs(right_dx) < 0.03 * max(np.ptp(x), 1.0e-12)
        if score[i] >= 9.0 and isolated:
            decisions.append(PointDecision(
                i, "delete", x[i], y[i], None, None, float(score[i]),
                "isolated gross digitisation departure (>9 robust scales)",
            ))
        elif reversal and score[i] >= 5.0:
            decisions.append(PointDecision(
                i, "delete", x[i], y[i], None, None, float(score[i]),
                "small unphysical coordinate reversal plus stress/strain departure",
            ))
        elif score[i] >= 5.0 and isolated:
            decisions.append(PointDecision(
                i, "adjust", x[i], y[i], x[i], float(y_reference[i]), float(score[i]),
                "isolated moderate digitisation departure; local quadratic correction",
            ))
    return decisions


def apply_decisions(curve: Curve, decisions: list[PointDecision]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(curve.x, dtype=float).copy()
    y = np.asarray(curve.y, dtype=float).copy()
    keep = np.ones(len(x), dtype=bool)
    for decision in decisions:
        if decision.action == "delete":
            keep[decision.index] = False
        elif decision.action == "adjust":
            x[decision.index] = decision.x_clean
            y[decision.index] = decision.y_clean
    return x[keep], y[keep]


def strictly_increasing_time(x: np.ndarray) -> np.ndarray:
    """Least-squares monotone repair followed by minimal strict separation."""
    x = np.asarray(x, dtype=float)
    levels = list(x)
    weights = [1] * len(x)
    starts = list(range(len(x)))
    ends = list(range(len(x)))
    i = 0
    while i < len(levels) - 1:
        if levels[i] <= levels[i + 1]:
            i += 1
            continue
        total = weights[i] + weights[i + 1]
        pooled = (weights[i] * levels[i] + weights[i + 1] * levels[i + 1]) / total
        levels[i:i + 2] = [pooled]
        weights[i:i + 2] = [total]
        ends[i] = ends[i + 1]
        del starts[i + 1], ends[i + 1]
        i = max(i - 1, 0)
    repaired = np.empty_like(x)
    for level, start, end in zip(levels, starts, ends):
        repaired[start:end + 1] = level
    positive_steps = np.diff(x)[np.diff(x) > 0]
    nominal = np.median(positive_steps) if positive_steps.size else 1.0
    epsilon = max(nominal * 1.0e-3, np.ptp(x) * 1.0e-10, 1.0e-12)
    for i in range(1, len(repaired)):
        repaired[i] = max(repaired[i], repaired[i - 1] + epsilon)
    return repaired
