"""Observation-independent material meshes for prescribed test histories.

Experimental coordinates define where residuals are evaluated.  They do not
define the constitutive time/strain increments.  The helpers below reduce a
piecewise-monotone control trace to its physical turning points, discretize
each segment by a declared maximum increment, and return fixed interpolation
coordinates for mapping simulated endpoints back to observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StrainMesh:
    strain: np.ndarray
    dt: np.ndarray
    path_coordinate: np.ndarray
    observation_coordinate: np.ndarray
    turning_indices: np.ndarray


@dataclass(frozen=True)
class TimeMesh:
    time: np.ndarray
    dt: np.ndarray


def cumulative_absolute_coordinate(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    return np.concatenate(([0.0], np.cumsum(np.abs(np.diff(values)))))


def turning_point_indices(values: np.ndarray, tolerance: float = 1.0e-12) -> np.ndarray:
    """Return endpoints and genuine direction changes of a scalar control."""
    values = np.asarray(values, dtype=float)
    if values.size <= 2:
        return np.arange(values.size, dtype=int)
    difference = np.diff(values)
    scale = max(float(np.ptp(values)), 1.0)
    sign = np.where(np.abs(difference) > tolerance * scale, np.sign(difference), 0.0)
    # Propagate the nearest non-zero direction across repeated coordinates.
    for index in range(1, len(sign)):
        if sign[index] == 0.0:
            sign[index] = sign[index - 1]
    for index in range(len(sign) - 2, -1, -1):
        if sign[index] == 0.0:
            sign[index] = sign[index + 1]
    changes = np.flatnonzero(sign[:-1] * sign[1:] < 0.0) + 1
    return np.unique(np.concatenate(([0], changes, [len(values) - 1]))).astype(int)


def strain_control_mesh(values: np.ndarray, rate: float, engineering_rate: bool,
                        maximum_true_strain_increment: float,
                        initial_dt: float = 1.0e-5) -> StrainMesh:
    """Discretize a piecewise-monotone strain history independently of samples."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("strain control must be a non-empty one-dimensional array")
    turns = turning_point_indices(values)
    vertices = values[turns]
    mesh = [float(vertices[0])]
    for start, end in zip(vertices[:-1], vertices[1:]):
        count = max(1, int(np.ceil(abs(end - start) / maximum_true_strain_increment)))
        mesh.extend(np.linspace(start, end, count + 1, dtype=float)[1:])
    strain = np.asarray(mesh, dtype=float)
    coordinate = 1.0 - np.exp(-strain) if engineering_rate else strain
    dt = np.empty_like(strain)
    # The first node is the initial condition, not a fictitious increment.
    dt[0] = 0.0
    if len(strain) > 1:
        dt[1:] = np.maximum(np.abs(np.diff(coordinate)) / rate, initial_dt)
    return StrainMesh(
        strain=strain,
        dt=dt,
        path_coordinate=cumulative_absolute_coordinate(strain),
        observation_coordinate=cumulative_absolute_coordinate(values),
        turning_indices=turns,
    )


def time_control_mesh(end_time: float, maximum_dt: float,
                      extra_vertices: tuple[float, ...] = (),
                      initial_dt: float = 1.0e-5,
                      growth_factor: float = 2.0) -> TimeMesh:
    """Construct a graded physical-time mesh at declared stage boundaries.

    Each stage begins at ``initial_dt`` and grows geometrically to
    ``maximum_dt``.  This resolves the fast constitutive transient after a
    load/hold transition without borrowing the digitised observation times.
    """
    end_time = float(end_time)
    if maximum_dt <= 0.0 or initial_dt <= 0.0 or growth_factor <= 1.0:
        raise ValueError("time increments must be positive and growth_factor > 1")
    vertices = sorted({0.0, end_time, *(float(value) for value in extra_vertices
                                       if 0.0 < value < end_time)})
    nodes = [0.0]
    for start, end in zip(vertices[:-1], vertices[1:]):
        current = start
        step = min(initial_dt, maximum_dt)
        tolerance = np.finfo(float).eps * max(abs(end), 1.0) * 8.0
        while end - current > tolerance:
            increment = min(step, maximum_dt, end - current)
            current = min(current + increment, end)
            nodes.append(current)
            step = min(step * growth_factor, maximum_dt)
    time = np.asarray(nodes, dtype=float)
    dt = np.empty_like(time)
    dt[0] = 0.0
    if len(time) > 1:
        dt[1:] = np.diff(time)
    return TimeMesh(time=time, dt=dt)
