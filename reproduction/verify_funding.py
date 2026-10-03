#!/usr/bin/env python3
"""Recompute the included funding benchmark offline and verify all public values.

Default: 5,000 calendar-month bootstrap resamples, seed 1729, SciPy statistics.
--refresh writes current reports/references/provenance from an actual computation.
--finalize-existing writes references/provenance from a previously computed full
funding_results.json. This is for a split CI job, not a substitute for computation.
Neither mode downloads data. Actual observation dates remain in the event table;
execution timestamps and machine paths are unnecessary for scientific provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from funding_persistence import (
    SOURCE_COMMIT, SOURCE_REPOSITORY, analyze_events, finite_events,
    public_reference, read_events, write_json,
)

ABS_TOLERANCE = 1e-10
REL_TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_equivalent(expected, actual, key="result") -> None:
    """Compare all scientific fields; software versions are reported, not fitted."""
    if isinstance(expected, dict):
        for name, value in expected.items():
            if name == "software":
                continue
            if name not in actual:
                raise AssertionError(f"Missing result field: {key}.{name}")
            assert_equivalent(value, actual[name], f"{key}.{name}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f"Array length mismatch: {key}")
        for i, (x, y) in enumerate(zip(expected, actual)):
            assert_equivalent(x, y, f"{key}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(expected, actual, rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE):
            raise AssertionError(f"Numerical mismatch: {key}: {expected} != {actual}")
    elif expected != actual:
        raise AssertionError(f"Value mismatch: {key}: {expected!r} != {actual!r}")


def predictor_reference(result: dict) -> dict:
    return {
        "scope": "Predictor comparisons on the same finite event table as the descriptive funding benchmark.",
        "source_repository": result["source_repository"], "source_commit": result["source_commit"],
        "total_events": result["total_events"], "matched_finite_events": result["finite_lifetime_score_events"],
        "series": result["series_count"], "calendar_month_clusters": result["calendar_month_clusters"],
        "event_table_sha256": result["event_table_sha256"],
        "statistics": result["ar1_comparators"], "formulas": result["ar1_formulas"],
        "matched_sample_identity_spearman_lifetime_edge": result["statistics"]["spearman_lifetime_edge"],
        "sample_method": result["extraction_rule"],
        "spearman_method": "SciPy spearmanr; Pearson correlation of average ranks, including ties.",
    }


def make_provenance(root: Path, full, result: dict) -> dict:
    names = [
        "reproduction/funding_persistence.py", "reproduction/export_funding_events.py",
        "reproduction/verify_funding.py", "reproduction/ar1_predictor_audit.py", "reproduction/data/funding_events.csv",
        "reproduction/data/funding_events.export.json", "reproduction/data/funding_input_manifest.json",
        "reproduction/data/funding_results.json", "reproduction/reference_results.json",
        "reproduction/ar1_predictor_reference.json",
    ]
    hashes = {name: sha256(root / name) for name in names}
    return {
        "schema": "cac-funding-provenance-1",
        "description": "Event table derived from the pinned source inputs by the supplied extraction rule.",
        "source_repository": SOURCE_REPOSITORY, "source_commit": SOURCE_COMMIT,
        "files": hashes,
        "input_manifest_sha256": hashes["reproduction/data/funding_input_manifest.json"],
        "script_sha256": hashes["reproduction/funding_persistence.py"],
        "summary_sha256": hashes["reproduction/data/funding_results.json"],
        "event_table": {
            "path": "reproduction/data/funding_events.csv", "rows": len(full),
            "finite_score_rows": len(finite_events(full)), "columns": list(full.columns),
            "sha256": hashes["reproduction/data/funding_events.csv"],
            "observation_timestamp_columns": ["time", "training_start", "training_end", "future_start", "future_end"],
        },
        "computation": {
            "bootstrap": result["calendar_month_cluster_bootstrap"],
            "numerical_tolerance": {"absolute": ABS_TOLERANCE, "relative": REL_TOLERANCE},
            "software": result["software"],
        },
    }


def check_hashes(root: Path, provenance: dict) -> None:
    for name, expected in provenance["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise AssertionError(f"Provenance hash mismatch: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    action = ap.add_mutually_exclusive_group()
    action.add_argument("--refresh", action="store_true")
    action.add_argument("--finalize-existing", action="store_true")
    ap.add_argument("--report", type=Path, help="Optional verification report, outside or inside the release")
    args = ap.parse_args()
    root = args.root.resolve()
    table = root / "reproduction/data/funding_events.csv"
    result_path = root / "reproduction/data/funding_results.json"
    provenance_path = root / "reproduction/data/funding_provenance.json"
    full = read_events(table)
    if args.finalize_existing:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["event_table_sha256"] != sha256(table):
            raise AssertionError("Existing statistics were computed from a different event table")
        if (result["calendar_month_cluster_bootstrap"]["resamples"], result["calendar_month_cluster_bootstrap"]["seed"]) != (5000, 1729):
            raise AssertionError("Existing statistics do not use the fixed bootstrap settings")
    else:
        result = analyze_events(full, 5000, 1729)
        result["event_table_sha256"] = sha256(table)
    if args.refresh or args.finalize_existing:
        if args.refresh:
            write_json(result_path, result)
        write_json(root / "reproduction/reference_results.json", public_reference(result))
        write_json(root / "reproduction/ar1_predictor_reference.json", predictor_reference(result))
        write_json(provenance_path, make_provenance(root, full, result))
    else:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        check_hashes(root, provenance)
        expected = json.loads(result_path.read_text(encoding="utf-8"))
        assert_equivalent(expected, result)
        assert_equivalent(json.loads((root / "reproduction/reference_results.json").read_text(encoding="utf-8")), public_reference(result))
        assert_equivalent(json.loads((root / "reproduction/ar1_predictor_reference.json").read_text(encoding="utf-8")), predictor_reference(result))
    report = {
        "status": "PASS", "operation": "finalize_existing" if args.finalize_existing else "refresh" if args.refresh else "recompute_and_verify",
        "total_events": len(full), "finite_events": len(finite_events(full)),
        "bootstrap_resamples": 5000, "bootstrap_seed": 1729,
        "event_table_sha256": sha256(table), "absolute_tolerance": ABS_TOLERANCE, "relative_tolerance": REL_TOLERANCE,
    }
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
