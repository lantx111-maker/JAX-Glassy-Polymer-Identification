"""Generate the authoritative 44-curve manifest without modifying raw data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from applications.glassy_polymer.data import load_curve, resolve_data_root
from applications.glassy_polymer.data_cleaning import propose_cleaning, strictly_increasing_time
from applications.glassy_polymer.experiments import SPLIT_FILE, build_experiments


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "data_manifest"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data_root = resolve_data_root()
    experiments = build_experiments()
    rows = []
    for experiment in experiments:
        curve = load_curve(experiment.source_path)
        decisions = propose_cleaning(curve)
        delete_count = sum(item.action == "delete" for item in decisions)
        adjust_count = sum(item.action == "adjust" for item in decisions)
        is_time = experiment.control in {"time", "stress"} or experiment.family == "relaxation"
        if is_time:
            repaired = strictly_increasing_time(experiment.x)
            changes = np.abs(repaired - experiment.x)
            changed = changes > 0.0
            max_change = float(np.max(changes))
            median_change = float(np.median(changes[changed])) if np.any(changed) else 0.0
            duration = max(float(np.ptp(experiment.x)), 1.0e-30)
            relative_max_change = max_change / duration
            repaired_points = int(np.count_nonzero(changed))
        else:
            max_change = median_change = relative_max_change = 0.0
            repaired_points = 0
        rows.append({
            "curve_id": experiment.curve_id,
            "material": experiment.material,
            "source_reference": str(experiment.source_path.parent.relative_to(data_root)),
            "source_file": experiment.label,
            "sha256": sha256(experiment.source_path),
            "protocol_family": experiment.family,
            "control_mode": experiment.control,
            "target": "" if experiment.target is None else experiment.target,
            "nominal_rate_per_s": "" if experiment.rate is None else experiment.rate,
            "source_points": len(curve.x),
            "analysis_points": len(experiment.x),
            "split": experiment.split,
            "compound_test_id": experiment.compound_test_id or "",
            "time_repaired_points": repaired_points,
            "time_max_abs_change": max_change,
            "time_median_abs_change": median_change,
            "time_max_change_over_duration": relative_max_change,
            "flagged_delete": delete_count,
            "flagged_adjust": adjust_count,
            "response_values_modified": "no",
        })

    fields = list(rows[0])
    with (OUTPUT / "data_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    unique_records = {
        row["compound_test_id"] or row["curve_id"] for row in rows
    }
    family_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for row in rows:
        key = f'{row["material"]}:{row["protocol_family"]}'
        family_counts[key] = family_counts.get(key, 0) + 1
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    summary = {
        "manifest_version": json.loads(SPLIT_FILE.read_text())["version"],
        "data_root_policy": "packaged snapshot or GLASSY_POLYMER_DATA_ROOT override",
        "response_curves": len(rows),
        "independent_experiment_records": len(unique_records),
        "source_points": sum(row["source_points"] for row in rows),
        "analysis_points": sum(row["analysis_points"] for row in rows),
        "family_counts": family_counts,
        "split_counts": split_counts,
        "flagged_delete": sum(row["flagged_delete"] for row in rows),
        "flagged_adjust": sum(row["flagged_adjust"] for row in rows),
        "qc_status": (
            "response values retained; five delete candidates and 12 adjustment "
            "candidates are flags for sensitivity analysis only"
        ),
        "time_repaired_points": sum(row["time_repaired_points"] for row in rows),
        "raw_files_modified": False,
    }
    (OUTPUT / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
