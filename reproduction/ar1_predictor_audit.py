#!/usr/bin/env python3
"""Recompute three AR(1) predictor comparisons from the included event table.

Fully offline. The exact same finite-event rule is used by the descriptive
funding benchmark. No raw histories are needed, no model is refitted, and no
trading P&L is computed. Event-table inputs are validated by funding_persistence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from funding_persistence import (
    AR1_COLUMNS, SOURCE_COMMIT, SOURCE_REPOSITORY, _rho,
    finite_events, read_events, write_json,
)


def run(events: Path) -> dict:
    full = read_events(events)
    finite = finite_events(full)
    return {
        "scope": "Predictor comparisons on the same finite event table as the descriptive funding benchmark.",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "total_events": len(full),
        "matched_finite_events": len(finite),
        "series": int(finite.series.nunique()),
        "calendar_month_clusters": int(finite.month.nunique()),
        "event_table_sha256": hashlib.sha256(events.read_bytes()).hexdigest(),
        "statistics": {
            name: {"n": len(finite), "spearman_lifetime_score": _rho(finite.lifetime_score, finite[name])}
            for name in AR1_COLUMNS
        },
        "matched_sample_identity_spearman_lifetime_edge": _rho(finite.lifetime_score, finite.edge_z),
        "spearman_method": "SciPy spearmanr; Pearson correlation of average ranks, including ties.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, default=Path(__file__).resolve().parent / "data/funding_events.csv")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--check-reference", action="store_true", help="Compare all recomputed fields with the current predictor reference")
    args = ap.parse_args()
    result = run(args.events)
    if args.check_reference:
        from verify_funding import assert_equivalent
        reference = json.loads((Path(__file__).resolve().parent / "ar1_predictor_reference.json").read_text(encoding="utf-8"))
        assert_equivalent(result, reference)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
