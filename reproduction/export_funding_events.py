#!/usr/bin/env python3
"""Derive the included event table from twelve pinned funding histories.

Requires NumPy and pandas, but not SciPy. Raw files are read locally by default.
--download explicitly fetches only the twelve immutable repository objects in
the manifest and checks both their Git blob SHA1 and SHA256 before extraction.
The event table retains actual observation times, including future-window times.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import pandas as pd

from funding_persistence import build_events, finite_events, write_json


def verify_inputs(data_root: Path, manifest: dict, download: bool = False) -> list[dict]:
    verified = []
    for item in manifest["files"]:
        path = data_root / item["path"]
        if download and not path.exists():
            url = f'https://raw.githubusercontent.com/{manifest["repository"]}/{manifest["commit"]}/{item["path"]}'
            with urllib.request.urlopen(url, timeout=60) as response:
                blob = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        blob = path.read_bytes()
        git_sha1 = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
        sha256 = hashlib.sha256(blob).hexdigest()
        if git_sha1 != item["git_sha1"] or sha256 != item["sha256"]:
            raise ValueError(f"Pinned input hash mismatch: {item['path']}")
        verified.append({**item, "bytes": len(blob)})
    return verified


def main() -> None:
    base = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--input-manifest", type=Path, default=base / "data/funding_input_manifest.json")
    ap.add_argument("--output", type=Path, default=base / "data/funding_events.csv")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    verified = verify_inputs(args.data_root, manifest, args.download)
    full = build_events(args.data_root)
    finite = finite_events(full)
    if (len(full), len(finite), full.series.nunique(), finite.month.nunique()) != (545, 543, 12, 38):
        raise ValueError("Pinned extraction does not reproduce the declared event counts")
    full["finite_analysis_event"] = full.index.isin(finite.index)
    for name in ["time", "training_start", "training_end", "future_start", "future_end"]:
        full[name] = pd.to_datetime(full[name], utc=True).map(lambda x: x.isoformat())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(args.output, index=False, float_format="%.17g", na_rep="", lineterminator="\n")
    result = {
        "schema": "cac-funding-event-export-1",
        "description": "Event table derived from the pinned source inputs by the supplied extraction rule.",
        "total_events": len(full), "finite_analysis_events": len(finite),
        "calendar_month_clusters": int(finite.month.nunique()), "series_count": int(full.series.nunique()),
        "event_table_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "input_manifest_sha256": hashlib.sha256(args.input_manifest.read_bytes()).hexdigest(),
        "source_repository": manifest["repository"], "source_commit": manifest["commit"],
        "columns": list(full.columns), "source_files": verified,
    }
    write_json(args.output.with_suffix(".export.json"), result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
