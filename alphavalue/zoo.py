"""Batch AlphaValue certificates for a predeclared signal zoo.

The batch runner is intentionally data-source agnostic.  It consumes a long CSV
whose rows have already been constructed under a predeclared mapping without using future outcome data.
It does not silently infer transaction costs, choose signals after outcomes, or
turn the canonical lifetime barrier into a sufficient sample-size claim.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .planning import canonical_one_percent_history_floor
from .multiplicity import e_bh, nig_fixed_cut_coefficient_evalue
from .unknown_scale import (
    NIGMixtureTuning,
    UnknownScaleCertificateModel,
    unknown_scale_value_certificate,
)


@dataclass(frozen=True)
class ZooColumns:
    name: str = "name"
    signal: str = "signal"
    next_signal: str = "next_signal"
    next_return: str = "next_return"
    trading_cost: str = "trading_cost"


def _read_rows(path: Path, columns: ZooColumns) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {columns.name, columns.signal, columns.next_signal, columns.next_return}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns {sorted(required)!r}")
        groups: dict[str, list[dict[str, str]]] = {}
        order: list[str] = []
        for row in reader:
            name = str(row[columns.name]).strip()
            if not name:
                raise ValueError("zoo name values must be nonempty")
            if name not in groups:
                groups[name] = []
                order.append(name)
            groups[name].append(row)
    if not groups:
        raise ValueError("zoo CSV contains no observations")
    return order, groups


def _group_cost(rows: list[dict[str, str]], *, cost_column: str, fallback: float | None) -> float:
    values: list[float] = []
    for row in rows:
        raw = row.get(cost_column)
        if raw is not None and str(raw).strip() != "":
            values.append(float(raw))
    if values:
        first = values[0]
        if not np.isfinite(values).all() or first < 0:
            raise ValueError("trading costs must be finite and nonnegative")
        if max(abs(v-first) for v in values) > 1e-12 * max(1.0, abs(first)):
            raise ValueError("trading_cost must be constant within each zoo signal")
        return float(first)
    if fallback is None:
        raise ValueError("each zoo signal needs a trading_cost column or --trading-cost fallback")
    if not np.isfinite(fallback) or fallback < 0:
        raise ValueError("trading_cost fallback must be finite and nonnegative")
    return float(fallback)


def run_zoo(
    path: Path,
    *,
    gamma: float,
    horizon: int,
    alpha: float = 0.05,
    hurdle: float = 0.0,
    efficiency_tolerance: float = 0.10,
    phi_bounds: tuple[float, float] = (-0.999, 0.999),
    trading_cost: float | None = None,
    columns: ZooColumns = ZooColumns(),
    theta_cells: int = 8,
    phi_cells: int = 24,
    period_unit: str = "periods",
    return_tuning: NIGMixtureTuning = NIGMixtureTuning(),
    signal_tuning: NIGMixtureTuning = NIGMixtureTuning(),
) -> dict:
    """Run the unknown-scale certificate signal-by-signal on a predeclared long CSV."""
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be positive and finite")
    order, groups = _read_rows(Path(path), columns)
    results: list[dict] = []
    decision_counts: dict[str, int] = {}
    efficiency_counts: dict[str, int] = {}
    lifetime_counts: dict[str, int] = {}

    for name in order:
        rows = groups[name]
        cost = _group_cost(rows, cost_column=columns.trading_cost, fallback=trading_cost)
        x = np.asarray([float(r[columns.signal]) for r in rows], dtype=float)
        xn = np.asarray([float(r[columns.next_signal]) for r in rows], dtype=float)
        y = np.asarray([float(r[columns.next_return]) for r in rows], dtype=float)
        if not (np.isfinite(x).all() and np.isfinite(xn).all() and np.isfinite(y).all()):
            raise ValueError(f"nonfinite observations in zoo signal {name!r}")
        model = UnknownScaleCertificateModel(
            gamma=float(gamma), trading_cost=cost, horizon=int(horizon), terminal_penalty=0.0
        )
        cert = unknown_scale_value_certificate(
            x, xn, y,
            model=model,
            hurdle=float(hurdle), alpha=float(alpha),
            efficiency_tolerance=float(efficiency_tolerance),
            return_tuning=return_tuning, signal_tuning=signal_tuning,
            phi_bounds=phi_bounds,
            theta_cells=int(theta_cells), phi_cells=int(phi_cells),
        )
        plan = canonical_one_percent_history_floor(
            int(horizon), len(rows), period_unit=period_unit
        )
        decision = cert["decision"]
        efficiency = cert.get("efficiency_decision", "not_certified")
        lifetime = plan["floor_comparison"]
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        efficiency_counts[efficiency] = efficiency_counts.get(efficiency, 0) + 1
        lifetime_counts[lifetime] = lifetime_counts.get(lifetime, 0) + 1
        rectangle = cert.get("parameter_set", {}).get("projection_rectangle")
        results.append({
            "name": name,
            "n_observations": len(rows),
            "trading_cost": cost,
            "decision": decision,
            "efficiency_decision": efficiency,
            "certified_eta_efficient": bool(cert.get("certified_eta_efficient", False)),
            "relative_regret_upper": cert.get("relative_regret_upper"),
            "policy_value_lower": cert.get("policy_value_lower"),
            "oracle_value_upper": cert.get("oracle_value_upper"),
            "theta_interval": None if rectangle is None else rectangle["theta"],
            "phi_interval": None if rectangle is None else rectangle["phi"],
            "canonical_history": plan,
        })

    total = len(results)
    return {
        "n_signals": total,
        "decision_counts": decision_counts,
        "decision_fractions": {k: v / total for k, v in decision_counts.items()},
        "efficiency_counts": efficiency_counts,
        "lifetime_counts": lifetime_counts,
        "results": results,
        "coverage_scope": (
            "Each signal certificate is time-uniform under its declared conditional-Gaussian model. "
            "No familywise guarantee across a data-dependent choice of zoo is created by this batch runner; "
            "the signal universe and multiplicity policy must be declared before outcome evaluation."
        ),
        "lifetime_scope": (
            "The 3.670443... history/deployment comparison is the canonical 1% necessary barrier only; "
            "it is not a sufficient sample-size certificate for an individual signal."
        ),
    }


def write_zoo_csv(report: dict, output: Path) -> None:
    """Write a compact machine-readable signal-level table from ``run_zoo``."""
    rows = report.get("results", [])
    fields = [
        "name", "n_observations", "trading_cost", "decision", "efficiency_decision",
        "certified_eta_efficient", "relative_regret_upper", "policy_value_lower", "oracle_value_upper",
        "theta_lower", "theta_upper", "phi_lower", "phi_upper",
        "canonical_1pct_history_floor", "additional_periods_to_floor", "floor_comparison",
    ]
    with Path(output).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            ti = r.get("theta_interval") or [None, None]
            pi = r.get("phi_interval") or [None, None]
            p = r["canonical_history"]
            writer.writerow({
                "name": r["name"], "n_observations": r["n_observations"],
                "trading_cost": r["trading_cost"], "decision": r["decision"],
                "efficiency_decision": r["efficiency_decision"],
                "certified_eta_efficient": r["certified_eta_efficient"],
                "relative_regret_upper": r["relative_regret_upper"],
                "policy_value_lower": r["policy_value_lower"],
                "oracle_value_upper": r["oracle_value_upper"],
                "theta_lower": ti[0], "theta_upper": ti[1], "phi_lower": pi[0], "phi_upper": pi[1],
                "canonical_1pct_history_floor": p["necessary_history_floor_integer"],
                "additional_periods_to_floor": p["additional_periods_to_floor"],
                "floor_comparison": p["floor_comparison"],
            })



def fixed_cut_loading_discoveries(path: Path, *, alpha: float=0.05,
                                  columns: ZooColumns=ZooColumns()) -> dict:
    """Secondary fixed-cut e-BH screen for H0: theta_j=0 across a predeclared zoo.

    The nuisance return-noise variance is maximized out in the null denominator.
    This is a statistical loading-discovery screen, not an economic DEPLOY label.
    It is valid only at the prespecified cut represented by ``path``; repeated
    monthly stopped use requires an additional global-filtration argument.
    """
    order,groups=_read_rows(Path(path),columns)
    evalues=[]
    for name in order:
        rows=groups[name]
        x=np.asarray([float(r[columns.signal]) for r in rows],dtype=float)
        y=np.asarray([float(r[columns.next_return]) for r in rows],dtype=float)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError(f'nonfinite observations in zoo signal {name!r}')
        evalues.append(nig_fixed_cut_coefficient_evalue(x,y,beta0=0.0))
    bh=e_bh(evalues,alpha=alpha)
    return {
        'analysis':'secondary_fixed_cut_loading_eBH',
        'alpha':float(alpha),'n_signals':len(order),
        'results':[{'name':name,'e_value':float(ev) if np.isfinite(ev) else 'inf',
                    'rejected_theta_zero':bool(rej)}
                   for name,ev,rej in zip(order,evalues,bh['rejected'])],
        'n_discoveries':bh['k'],'threshold':bh['threshold'],
        'scope':('Secondary fixed-cut FDR screen for exact zero loading. It does not replace '
                 'familywise anytime-valid economic labels, and it does not itself certify DEPLOY.'),
    }
