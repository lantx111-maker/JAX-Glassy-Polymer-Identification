"""Read-only discovery and loading of the glassy-polymer data snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import openpyxl


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "data" / "raw"
DATA_ROOT_ENV = "GLASSY_POLYMER_DATA_ROOT"


def resolve_data_root(root: Path | None = None) -> Path:
    """Resolve an explicit, environment-provided, or packaged data root.

    The repository snapshot is the reproducible default.  The environment
    override supports private replacements without embedding workstation paths
    in source code or generated manifests.
    """
    candidate = Path(root) if root is not None else Path(
        os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT)
    )
    candidate = candidate.expanduser().resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Experimental data root not found: {candidate}. Set "
            f"{DATA_ROOT_ENV} or restore {DEFAULT_DATA_ROOT}."
        )
    return candidate


@dataclass(frozen=True)
class Curve:
    path: Path
    x_label: str
    y_label: str
    x: tuple[float, ...]
    y: tuple[float, ...]


@dataclass(frozen=True)
class CurveMetadata:
    material: str
    family: str
    control: str
    nominal_rate: float | None
    target: float | None


def canonical_workbooks(root: Path | None = None) -> list[Path]:
    """Return individual curves, excluding the eight aggregate duplicate files."""
    data_root = resolve_data_root(root)
    return sorted(
        p for p in data_root.rglob("*.xlsx") if "相关的实验数据" in str(p)
    )


def load_curve(path: Path) -> Curve:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    header = rows[0] if rows and isinstance(rows[0][0], str) else ("x", "y")
    start = 1 if rows and isinstance(rows[0][0], str) else 0
    xy: list[tuple[float, float]] = []
    for row in rows[start:]:
        if len(row) >= 2 and isinstance(row[0], (int, float)) and isinstance(row[1], (int, float)):
            xy.append((float(row[0]), float(row[1])))
    if not xy:
        raise ValueError(f"No numeric curve found in {path}")
    x, y = zip(*xy)
    return Curve(path, str(header[0]), str(header[1]), tuple(x), tuple(y))


def load_all_curves(paths: Iterable[Path] | None = None) -> list[Curve]:
    return [load_curve(p) for p in (canonical_workbooks() if paths is None else paths)]


def classify_curve(curve: Curve) -> CurveMetadata:
    """Classify the supplied corpus without modifying or duplicating raw data."""
    import re

    text = str(curve.path)
    material = "PC" if "聚碳酸酯（PC）" in text else "PMMA"
    if "蠕变" in text:
        family, control = "creep", "stress"
        match = re.search(r"(42|74|94|99|101)MPa", text)
        target = float(match.group(1)) if match else None
    elif "松弛" in text:
        family, control = "relaxation", "strain"
        match = re.search(r"大约为([0-9.]+)", text)
        target = float(match.group(1)) if match else None
    elif "循环" in text:
        family, control = "cyclic", "strain"
        match = re.search(r"最大真实应变([0-9.]+)", text)
        target = float(match.group(1)) if match else None
    elif "应变恢复曲线" in text or "恢复期间" in text:
        family, control = "recovery", "time"
        match = re.search(r"最大真实应变大约为([0-9.]+)", text)
        target = float(match.group(1)) if match else (0.2 if material == "PMMA" else None)
    elif "单调压缩-拉伸" in text:
        family, control = "reverse", "strain"
        match = re.search(r"大约为([0-9.]+)", text)
        target = float(match.group(1)) if match else None
    elif "加载-卸载-应变恢复" in text or "0应力下恢复600s" in text:
        family, control = "load_unload_recovery", "strain"
        match = re.search(r"大约为([0-9.]+)", text)
        target = float(match.group(1)) if match else (0.2 if material == "PMMA" else None)
    else:
        family, control = "load_unload", "strain"
        match = re.search(r"大约为([0-9.]+)", text)
        target = float(match.group(1)) if match else None
    rate_match = re.search(r"([0-9.]+) s-1", text)
    rate = float(rate_match.group(1)) if rate_match else None
    return CurveMetadata(material, family, control, rate, target)
