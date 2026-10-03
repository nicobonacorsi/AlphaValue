#!/usr/bin/env python3
"""Reproduce the funding-persistence illustration in Certified Alpha Capacity.

The included derived event table supports fully offline reproduction. To rebuild
it from the pinned raw inputs, use export_funding_events.py. The extraction rule,
nonintercept AR(1) fit and monthly cluster bootstrap retain their original
definitions. This is a descriptive funding benchmark, not a trading P&L study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_REPOSITORY = "supervik/historical-funding-rates-fetcher"
SOURCE_COMMIT = "66a085bc68147df2dd3360a25ac6e9f38e7077b5"
W, K, COOLDOWN = 540, 21, 21
I_CRIT = 4.281923675333987
GAP = pd.Timedelta(hours=8)
TOL = pd.Timedelta(minutes=5)
PATHS = [
    f"data/{asset}/{asset}_{venue}_{'2020' if venue in {'binance','gate'} else '2021'}-01-01_2024-01-01_funding_history.csv"
    for asset in ("BTC-USDT", "ETH-USDT")
    for venue in ("binance", "bybit", "gate", "htx", "kucoin", "mexc")
]


def _read(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=["Date", "Funding Rate"])
    d["Date"] = pd.to_datetime(d["Date"], utc=True, errors="coerce")
    d["Funding Rate"] = pd.to_numeric(d["Funding Rate"], errors="coerce")
    return d.dropna().sort_values("Date").drop_duplicates("Date", keep="first").reset_index(drop=True)


def _continuous(times: pd.Series, a: int, b: int) -> bool:
    gaps = times.iloc[a + 1 : b + 1].reset_index(drop=True) - times.iloc[a:b].reset_index(drop=True)
    return bool(((gaps - GAP).abs() <= TOL).all())


def _rho(a, b) -> float:
    from scipy.stats import spearmanr
    if len(a) < 3:
        return math.nan
    return float(spearmanr(a, b).statistic)


def _residualized_rank_association(df: pd.DataFrame, controls: list[str]) -> float:
    from scipy.stats import rankdata
    y = rankdata(df["lifetime_score"].to_numpy())
    X = [np.ones(len(df))]
    for col in controls:
        X.append(rankdata(df[col].to_numpy()))
    X = np.column_stack(X)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return _rho(resid, rankdata(df["future_sum_7d"].to_numpy()))


AR1_COLUMNS = [
    "ar1_raw_cumulative_21",
    "ar1_cumulative_forecast_snr",
    "ar1_finite_geometric_information_proxy",
]


def ar1_features(current: float, phi: float, sigma: float) -> dict:
    """Three prespecified comparators from the same event's nonintercept fit."""
    if not (0 < phi < 1 and sigma > 0):
        return dict.fromkeys(AR1_COLUMNS, math.nan)
    powers = phi ** np.arange(1, K + 1)
    innovation_weights = np.cumsum(phi ** np.arange(K))
    forecast = abs(current) * float(powers.sum())
    forecast_sd = sigma * math.sqrt(float(innovation_weights @ innovation_weights))
    return dict(zip(AR1_COLUMNS, [
        forecast,
        forecast / forecast_sd,
        current * current * float(powers @ powers) / (2 * sigma * sigma),
    ]))


def extract_events(d: pd.DataFrame, series: str) -> list[dict]:
    f = d["Funding Rate"].to_numpy(float)
    times = d["Date"]
    events: list[dict] = []
    next_i = W
    for i in range(W, len(d) - K):
        if i < next_i or not _continuous(times, i - W, i + K):
            continue
        hist = f[i - W : i]
        cur = float(f[i])
        if abs(cur) + 1e-18 < float(np.quantile(np.abs(hist), 0.95)):
            continue
        sign = 1.0 if cur >= 0 else -1.0
        x, y = hist[:-1], hist[1:]
        den = float(np.dot(x, x))
        phi = float(np.dot(x, y) / den) if den > 0 else math.nan
        resid = y - phi * x if math.isfinite(phi) else np.array([])
        sig = float(np.std(resid, ddof=1)) if resid.size > 1 else math.nan
        hist_sd = float(np.std(hist, ddof=1))
        edge_z = float(abs(cur) / hist_sd) if hist_sd > 0 else math.nan
        oriented_hist = sign * hist
        t_stat = float(np.mean(oriented_hist) / (np.std(oriented_hist, ddof=1) / math.sqrt(W)))
        lifetime_score = math.nan
        if 0 < phi < 1 and sig > 0:
            lam = -math.log(phi)
            info = cur * cur / (4 * sig * sig * lam)
            lifetime_score = info / I_CRIT
        future = sign * f[i + 1 : i + 1 + K]
        survival = 0
        for value in future:
            if value > 0:
                survival += 1
            else:
                break
        events.append(
            {
                "series": series,
                "time": times.iloc[i],
                "month": times.iloc[i].strftime("%Y-%m"),
                "settlement_index": int(i),
                "training_start": times.iloc[i - W],
                "training_end": times.iloc[i - 1],
                "future_start": times.iloc[i + 1],
                "future_end": times.iloc[i + K],
                "current_funding": cur,
                "phi_hat": phi,
                "residual_sigma": sig,
                "historical_sigma": hist_sd,
                "lifetime_score": lifetime_score,
                "edge_z": edge_z,
                "t_stat": t_stat,
                "future_sum_7d": float(np.sum(future)),
                "future_positive": bool(np.sum(future) > 0),
                "sign_survival_length": int(survival),
                **ar1_features(cur, phi, sig),
            }
        )
        next_i = i + COOLDOWN + 1
    return events


def summarize(df: pd.DataFrame) -> dict:
    top_l = df[df["lifetime_score"] >= df["lifetime_score"].quantile(0.8)]
    top_e = df[df["edge_z"] >= df["edge_z"].quantile(0.8)]
    return {
        "n": int(len(df)),
        "spearman_lifetime_future": _rho(df.lifetime_score, df.future_sum_7d),
        "spearman_edge_future": _rho(df.edge_z, df.future_sum_7d),
        "spearman_t_future": _rho(df.t_stat, df.future_sum_7d),
        "spearman_lifetime_edge": _rho(df.lifetime_score, df.edge_z),
        "incremental_lifetime_given_edge": _residualized_rank_association(df, ["edge_z"]),
        "incremental_lifetime_given_edge_and_t": _residualized_rank_association(df, ["edge_z", "t_stat"]),
        "top_quintile_lifetime": {
            "n": int(len(top_l)),
            "positive_count": int(top_l.future_positive.sum()),
            "positive_share": float(top_l.future_positive.mean()),
            "mean_sign_survival": float(top_l.sign_survival_length.mean()),
        },
        "top_quintile_edge": {
            "n": int(len(top_e)),
            "positive_count": int(top_e.future_positive.sum()),
            "positive_share": float(top_e.future_positive.mean()),
            "mean_sign_survival": float(top_e.sign_survival_length.mean()),
        },
    }


def month_cluster_bootstrap(df: pd.DataFrame, B: int = 5000, seed: int = 1729) -> dict:
    rng = np.random.default_rng(seed)
    months = np.asarray(sorted(df["month"].unique()))
    groups = {m: df.index[df.month.eq(m)].to_numpy() for m in months}
    out = {"lifetime": [], "edge": [], "t_stat": [], "incremental": []}
    for _ in range(B):
        sampled = rng.choice(months, size=len(months), replace=True)
        # Concatenating by position preserves duplicated clusters in the bootstrap sample.
        boot = pd.concat([df.loc[groups[m]] for m in sampled], ignore_index=True)
        out["lifetime"].append(_rho(boot.lifetime_score, boot.future_sum_7d))
        out["edge"].append(_rho(boot.edge_z, boot.future_sum_7d))
        out["t_stat"].append(_rho(boot.t_stat, boot.future_sum_7d))
        out["incremental"].append(_residualized_rank_association(boot, ["edge_z"]))
    return {
        name: [float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))]
        for name, vals in out.items()
    }


def build_events(data_root: Path) -> pd.DataFrame:
    all_events: list[dict] = []
    for rel in PATHS:
        path = data_root / rel
        if not path.exists():
            raise FileNotFoundError(path)
        asset = rel.split("/")[1]
        venue = Path(rel).name[len(asset) + 1 :].split("_")[0]
        rows = extract_events(_read(path), f"{asset}:{venue}")
        for row in rows:
            row["source_path"] = rel
        all_events.extend(rows)
    return pd.DataFrame(all_events)


def finite_events(full: pd.DataFrame) -> pd.DataFrame:
    return full[
        np.isfinite(full.lifetime_score)
        & np.isfinite(full.edge_z)
        & np.isfinite(full.t_stat)
        & np.isfinite(full.future_sum_7d)
    ].copy()


def read_events(path: Path) -> pd.DataFrame:
    # Round-trip parsing preserves exact float64 ties written with %.17g.
    full = pd.read_csv(path, float_precision="round_trip")
    required = {"series", "time", "month", "lifetime_score", "edge_z", "t_stat",
                "future_sum_7d", "future_positive", "sign_survival_length",
                "current_funding", "phi_hat", "residual_sigma", *AR1_COLUMNS}
    missing = required - set(full.columns)
    if missing:
        raise ValueError(f"Event table is missing columns: {sorted(missing)}")
    times = pd.to_datetime(full.time, utc=True, format="ISO8601", errors="raise")
    if not times.dt.strftime("%Y-%m").eq(full.month).all():
        raise ValueError("Event month disagrees with its actual UTC timestamp")
    if full.duplicated(["series", "time"]).any():
        raise ValueError("Duplicate event identifier")
    if not full.future_positive.eq(full.future_sum_7d > 0).all():
        raise ValueError("Future-positive indicator disagrees with its funding sum")
    # Recompute comparators from stored fitted inputs; never fit or select anew.
    recomputed = pd.DataFrame([
        ar1_features(float(row.current_funding), float(row.phi_hat), float(row.residual_sigma))
        for row in full.itertuples(index=False)
    ])
    for name in AR1_COLUMNS:
        if not np.allclose(full[name].to_numpy(), recomputed[name].to_numpy(), rtol=1e-13, atol=0, equal_nan=True):
            raise ValueError(f"Stored comparator disagrees with fitted inputs: {name}")
    if "finite_analysis_event" in full and not np.array_equal(
        full.finite_analysis_event.to_numpy(), full.index.isin(finite_events(full).index)
    ):
        raise ValueError("Stored analysis-sample indicator disagrees with the finite-value rule")
    return full


def analyze_events(full: pd.DataFrame, bootstrap: int, seed: int) -> dict:
    import scipy
    finite = finite_events(full)
    if not np.isfinite(finite[AR1_COLUMNS].to_numpy()).all():
        raise ValueError("AR(1) comparisons must use the entire same finite analysis sample")
    result = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "series_count": len(PATHS),
        "total_events": int(len(full)),
        "finite_lifetime_score_events": int(len(finite)),
        "calendar_month_clusters": int(finite.month.nunique()),
        "series_counts": finite.groupby("series").size().astype(int).to_dict(),
        "event_table_description": "Derived from the pinned source inputs by the supplied extraction rule.",
        "extraction_rule": {
            "training_observations": W, "future_observations": K,
            "absolute_funding_percentile": 0.95, "threshold_tolerance": 1e-18,
            "cooldown_observations": COOLDOWN, "minimum_event_index_spacing": COOLDOWN + 1,
            "settlement_hours": 8, "settlement_tolerance_minutes": 5,
            "fit": "nonintercept AR(1) on 539 adjacent pairs in the preceding 540 observations",
            "residual_sigma_ddof": 1, "historical_sigma_ddof": 1, "Icrit": I_CRIT,
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "statistics": summarize(finite),
        "ar1_comparators": {
            name: {"n": int(len(finite)), "spearman_lifetime_score": _rho(finite.lifetime_score, finite[name])}
            for name in AR1_COLUMNS
        },
        "ar1_formulas": {
            "ar1_raw_cumulative_21": "abs(f_t) * sum_{k=1}^{21} phi_hat^k",
            "ar1_cumulative_forecast_snr": "raw_forecast / (sigma_hat_eps * sqrt(sum_{j=1}^{21} (sum_{r=0}^{21-j} phi_hat^r)^2))",
            "ar1_finite_geometric_information_proxy": "f_t^2/(2 sigma_hat_eps^2) * sum_{k=1}^{21} phi_hat^(2k)",
            "geometric_proxy_scope": "Squared deterministic mean forecasts scaled by innovation variance; not exact joint-path KL for a stochastic AR(1).",
        },
        "leave_one_series_incremental_range": [
            float(min(_residualized_rank_association(finite[finite.series.ne(s)], ["edge_z"]) for s in finite.series.unique())),
            float(max(_residualized_rank_association(finite[finite.series.ne(s)], ["edge_z"]) for s in finite.series.unique())),
        ],
    }
    if bootstrap > 0:
        result["calendar_month_cluster_bootstrap"] = {
            "resamples": int(bootstrap),
            "seed": int(seed),
            "ci95": month_cluster_bootstrap(finite, bootstrap, seed),
        }
    return result


def run(data_root: Path, bootstrap: int, seed: int) -> dict:
    return analyze_events(build_events(data_root), bootstrap, seed)


def public_reference(result: dict) -> dict:
    stats = result["statistics"]
    funding = {"finite_lifetime_score_events": result["finite_lifetime_score_events"], **stats}
    funding["leave_one_series_incremental_range"] = result["leave_one_series_incremental_range"]
    if "calendar_month_cluster_bootstrap" in result:
        intervals = dict(result["calendar_month_cluster_bootstrap"]["ci95"])
        intervals["incremental_lifetime_given_edge"] = intervals.pop("incremental")
        funding["calendar_month_cluster_ci95"] = intervals
    return {"funding_persistence": funding}


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path)
    source.add_argument("--events", type=Path, help="Included derived event CSV; fully offline")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--reference-output", type=Path, help="Write current figure reference values")
    args = ap.parse_args()
    if args.bootstrap < 0:
        ap.error("--bootstrap must be nonnegative")
    full = read_events(args.events) if args.events else build_events(args.data_root)
    result = analyze_events(full, args.bootstrap, args.seed)
    if args.events:
        result["event_table_sha256"] = hashlib.sha256(args.events.read_bytes()).hexdigest()
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        write_json(args.output, result)
    if args.reference_output:
        write_json(args.reference_output, public_reference(result))
    print(text)


if __name__ == "__main__":
    main()
