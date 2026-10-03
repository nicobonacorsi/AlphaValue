"""CLI for transparent model-only demonstrations and sequential/no-future-data CSV evaluation."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from . import FiniteWorldModel, adaptive_policy, phase_order_proxy, two_world_fixed_blend
from .certificate import LoadingCertificateModel, loading_certificate
from .joint import JointCertificateModel, joint_value_certificate
from .unknown_scale import (
    NIGMixtureTuning, UnknownScaleCertificateModel, unknown_scale_value_certificate,
)
from .planning import canonical_one_percent_history_floor
from .zoo import ZooColumns, run_zoo, write_zoo_csv, fixed_cut_loading_discoveries
from .survival_report import alpha_survival_report, render_survival_html, write_report_json
from .data_survival import analyze_edge_return_series, load_edge_return_csv, returns_only_identification_report, render_data_survival_html, render_returns_only_survival_html


SCOPE = (
    "Specified finite-world stationary Gaussian model; known loading, quadratic cost, "
    "risk penalty, innovation variance and candidate persistences. No market alpha, "
    "confidence coverage, statistical calibration or minimax guarantee is inferred "
    "from a CSV. Scores below are Bellman residual scores, whose world-wise "
    "expectations equal expected objective regret, not realized-profit regret."
)


def _run_report(signal, model, training_transitions, horizon, prior, objective, policy):
    result = adaptive_policy(signal, model=model, training_transitions=training_transitions,
                             horizon=horizon, prior=prior, objective=objective, policy=policy)
    return {
        "policy": policy, "objective": objective,
        "oracle_expected_objective_by_world": result.oracle_values.tolist(),
        "path_residual_score_by_world": result.residual_score_by_world.tolist(),
        "path_normalized_residual_score_by_world": (result.residual_score_by_world / result.oracle_values).tolist(),
        "first_position": float(result.positions[0]), "last_position": float(result.positions[-1]),
        "initial_deployment_posterior": result.posterior[0].tolist(),
        "last_deployment_posterior": result.posterior[-1].tolist(),
    }, result


def _demo(args):
    model = FiniteWorldModel((1.0 - 1.0 / 32, 1.0 - 2.0 / 32),
                            trading_cost=992.0, innovation_variance=1.0 / 32)
    n, T = 32, 64
    rng = np.random.default_rng(args.seed)
    true_world = 0
    x = np.empty(n + T)
    x[0] = rng.normal(scale=np.sqrt(model.stationary_variances[true_world]))
    for k in range(1, len(x)):
        x[k] = model.persistences[true_world] * x[k - 1] + rng.normal(scale=np.sqrt(model.innovation_variance))
    policies = ["bayes", "frozen_bayes", "posterior_parameter", "zero"]
    if args.objective == "relative":
        policies.append("fixed_minimax_blend")
    reports = [_run_report(x, model, n, T, None, args.objective, policy)[0] for policy in policies]
    return {
        "scope": SCOPE, "demo_type": "one seeded synthetic path; not a Monte Carlo performance estimate",
        "model": asdict(model), "seed": args.seed, "synthetic_true_world": true_world,
        "training_transitions": n, "horizon": T, "policies": reports,
        "phase_order_proxy": phase_order_proxy(horizon=T, training_transitions=n, epsilon=1.0 / 32),
        "phase_order_proxy_status": "Uncalibrated scale diagnostic; not a confidence bound or exact finite-sample rate",
        "fixed_blend_minimax_upper": two_world_fixed_blend(model, T),
    }


def _csv(args):
    with Path(args.input).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or args.signal_column not in reader.fieldnames:
            raise ValueError(f"CSV must contain column {args.signal_column!r}")
        signal = [float(row[args.signal_column]) for row in reader]
    model = FiniteWorldModel(tuple(args.phis), args.gamma, args.trading_cost, args.theta,
                            args.innovation_variance, args.terminal_penalty)
    T = args.horizon if args.horizon is not None else len(signal) - args.training_transitions
    report, result = _run_report(signal, model, args.training_transitions, T,
                                 args.prior, args.objective, args.policy)
    if args.output:
        output = Path(args.output)
        with output.open("w", newline="") as f:
            columns = ["signal_index", "signal", "position", "effective_gain"]
            columns += [f"posterior_world_{i}" for i in range(len(model.persistences))]
            columns += [f"decision_weight_world_{i}" for i in range(len(model.persistences))]
            writer = csv.writer(f)
            writer.writerow(columns)
            for t in range(T):
                k = args.training_transitions + t
                writer.writerow([k, signal[k], result.positions[t], result.effective_gains[t],
                                 *result.posterior[t], *result.decision_weights[t]])
    return {"scope": SCOPE, "model": asdict(model), "training_transitions": args.training_transitions,
            "horizon": T, "prior": args.prior, "result": report,
            "positions_csv": str(args.output) if args.output else None}



def _certificate(args):
    with Path(args.input).open(newline="") as f:
        reader=csv.DictReader(f)
        needed={args.signal_column,args.return_column}
        if reader.fieldnames is None or not needed.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns {sorted(needed)!r}")
        rows=list(reader)
    if not rows:
        raise ValueError("CSV contains no observations")
    x=np.array([float(r[args.signal_column]) for r in rows],dtype=float)
    y=np.array([float(r[args.return_column]) for r in rows],dtype=float)
    model=LoadingCertificateModel(
        phi=args.phi,gamma=args.gamma,trading_cost=args.trading_cost,
        signal_innovation_variance=args.signal_innovation_variance,
        return_noise_sd=args.return_noise_sd,horizon=args.horizon,
        confidence=args.confidence,terminal_penalty=args.terminal_penalty)
    return loading_certificate(x,y,model=model,hurdle=args.hurdle)



def _joint_certificate(args):
    with Path(args.input).open(newline="") as f:
        reader=csv.DictReader(f)
        needed={args.signal_column,args.next_signal_column,args.return_column}
        if reader.fieldnames is None or not needed.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns {sorted(needed)!r}")
        rows=list(reader)
    if not rows:
        raise ValueError("CSV contains no observations")
    x=np.array([float(r[args.signal_column]) for r in rows],dtype=float)
    xn=np.array([float(r[args.next_signal_column]) for r in rows],dtype=float)
    y=np.array([float(r[args.return_column]) for r in rows],dtype=float)
    model=JointCertificateModel(
        gamma=args.gamma,trading_cost=args.trading_cost,
        signal_innovation_sd=args.signal_innovation_sd,return_noise_sd=args.return_noise_sd,
        horizon=args.horizon,terminal_penalty=args.terminal_penalty)
    return joint_value_certificate(
        x,xn,y,model=model,hurdle=args.hurdle,alpha=args.alpha,
        theta_rho=args.theta_rho,phi_rho=args.phi_rho,
        phi_bounds=(args.phi_lower,args.phi_upper),
        theta_cells=args.theta_cells,phi_cells=args.phi_cells,search_seed=args.search_seed)



def _unknown_scale_certificate(args):
    with Path(args.input).open(newline="") as f:
        reader=csv.DictReader(f)
        needed={args.signal_column,args.next_signal_column,args.return_column}
        if reader.fieldnames is None or not needed.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns {sorted(needed)!r}")
        rows=list(reader)
    if not rows:
        raise ValueError("CSV contains no observations")
    x=np.array([float(r[args.signal_column]) for r in rows],dtype=float)
    xn=np.array([float(r[args.next_signal_column]) for r in rows],dtype=float)
    y=np.array([float(r[args.return_column]) for r in rows],dtype=float)
    model=UnknownScaleCertificateModel(
        gamma=args.gamma,trading_cost=args.trading_cost,horizon=args.horizon,
        terminal_penalty=args.terminal_penalty)
    rt=NIGMixtureTuning(mean=args.return_prior_mean,kappa=args.return_prior_kappa,
                        shape=args.return_prior_shape,scale=args.return_prior_scale)
    st=NIGMixtureTuning(mean=args.signal_prior_mean,kappa=args.signal_prior_kappa,
                        shape=args.signal_prior_shape,scale=args.signal_prior_scale)
    return unknown_scale_value_certificate(
        x,xn,y,model=model,hurdle=args.hurdle,alpha=args.alpha,
        efficiency_tolerance=args.efficiency_tolerance,return_tuning=rt,signal_tuning=st,
        phi_bounds=(args.phi_lower,args.phi_upper),
        theta_cells=args.theta_cells,phi_cells=args.phi_cells)


def _certify(args):
    result=_unknown_scale_certificate(args)
    with Path(args.input).open(newline="", encoding="utf-8") as f:
        observed_periods=sum(1 for _ in csv.DictReader(f))
    result["history_planning"] = canonical_one_percent_history_floor(
        args.horizon, observed_periods=observed_periods,
        period_unit=args.period_unit)
    result["history_planning_note"] = (
        "The reported history floor is the declared canonical one-percent necessary barrier; "
        "it is not a sufficient sample-size estimate for this signal.")
    return result


def _zoo(args):
    cols=ZooColumns(name=args.name_column, signal=args.signal_column,
                    next_signal=args.next_signal_column, next_return=args.return_column,
                    trading_cost=args.trading_cost_column)
    report=run_zoo(
        args.input,gamma=args.gamma,horizon=args.horizon,alpha=args.alpha,hurdle=args.hurdle,
        efficiency_tolerance=args.efficiency_tolerance,
        phi_bounds=(args.phi_lower,args.phi_upper),trading_cost=args.trading_cost,
        columns=cols,theta_cells=args.theta_cells,phi_cells=args.phi_cells,
        period_unit=args.period_unit)
    if args.output_csv:
        write_zoo_csv(report,args.output_csv)
        report["output_csv"] = str(args.output_csv)
    return report


def _loading_ebh(args):
    cols=ZooColumns(name=args.name_column,signal=args.signal_column,
                    next_signal=args.next_signal_column,next_return=args.return_column,
                    trading_cost=args.trading_cost_column)
    return fixed_cut_loading_discoveries(args.input,alpha=args.alpha,columns=cols)



def _survival_report(args):
    report=alpha_survival_report(
        sharpe=args.sharpe,half_life=args.half_life,alpha=args.alpha,power=args.power,
        multiplicity=args.trials,correction=args.correction,time_unit=args.time_unit)
    if args.output_json:
        write_report_json(report,args.output_json); report["output_json"]=str(args.output_json)
    if args.output_html:
        render_survival_html(report,args.output_html); report["output_html"]=str(args.output_html)
    return report


def _survival_data(args):
    if args.edge_column:
        r,e=load_edge_return_csv(args.input,return_column=args.return_column,edge_column=args.edge_column)
        report=analyze_edge_return_series(
            r,e,periods_per_unit=args.periods_per_unit,alpha=args.alpha,power=args.power,
            multiplicity=args.trials,correction=args.correction,time_unit=args.time_unit,
            bootstrap_samples=args.bootstrap_samples,bootstrap_level=args.bootstrap_level,
            block_length=args.block_length,seed=args.seed)
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
            report["output_json"]=str(args.output_json)
        if args.output_html:
            render_data_survival_html(report,args.output_html); report["output_html"]=str(args.output_html)
        return report
    # returns-only mode intentionally refuses to infer future lifetime.
    vals=[]
    with Path(args.input).open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None or args.return_column not in reader.fieldnames:
            raise ValueError(f"CSV must contain column {args.return_column!r}")
        for row in reader:
            try: x=float(row[args.return_column])
            except (TypeError,ValueError): continue
            if np.isfinite(x): vals.append(x)
    report=returns_only_identification_report(vals,periods_per_unit=args.periods_per_unit,time_unit=args.time_unit)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        report["output_json"]=str(args.output_json)
    if args.output_html:
        render_returns_only_survival_html(report,args.output_html); report["output_html"]=str(args.output_html)
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="compare four policies on one seeded synthetic path")
    demo.add_argument("--seed", type=int, default=1729)
    demo.add_argument("--objective", choices=("absolute", "relative"), default="absolute")
    csvp = sub.add_parser("csv", help="produce sequential positions under an explicitly supplied model without future data")
    csvp.add_argument("input", type=Path)
    csvp.add_argument("--signal-column", default="signal")
    csvp.add_argument("--phis", type=float, nargs="+", required=True)
    csvp.add_argument("--prior", type=float, nargs="+")
    csvp.add_argument("--gamma", type=float, required=True)
    csvp.add_argument("--trading-cost", type=float, required=True)
    csvp.add_argument("--theta", type=float, required=True)
    csvp.add_argument("--innovation-variance", type=float, required=True)
    csvp.add_argument("--terminal-penalty", type=float, default=0.0)
    csvp.add_argument("--training-transitions", type=int, required=True)
    csvp.add_argument("--horizon", type=int)
    csvp.add_argument("--objective", choices=("absolute", "relative"), default="absolute")
    csvp.add_argument("--policy", choices=("bayes", "frozen_bayes", "posterior_parameter", "fixed_minimax_blend", "zero"), default="bayes")
    csvp.add_argument("--output", type=Path)
    cert = sub.add_parser("certificate", help="three-way finite-sample value certificate for an unknown loading under the declared Gaussian model")
    cert.add_argument("input", type=Path)
    cert.add_argument("--signal-column", default="signal")
    cert.add_argument("--return-column", default="next_return")
    cert.add_argument("--phi", type=float, required=True, help="known AR(1) persistence of the signal")
    cert.add_argument("--gamma", type=float, required=True, help="quadratic risk coefficient")
    cert.add_argument("--trading-cost", type=float, required=True, help="quadratic position-change coefficient")
    cert.add_argument("--signal-innovation-variance", type=float, required=True, help="known AR(1) innovation variance")
    cert.add_argument("--return-noise-sd", type=float, required=True, help="known Gaussian return-noise standard deviation")
    cert.add_argument("--horizon", type=int, required=True, help="future trading periods in the model")
    cert.add_argument("--confidence", type=float, default=.95)
    cert.add_argument("--hurdle", type=float, default=0., help="minimum total finite-horizon model value required to deploy")
    cert.add_argument("--terminal-penalty", type=float, default=0.)
    joint = sub.add_parser("joint-certificate", help="anytime-valid joint strength/persistence value certificate under the declared Gaussian model")
    joint.add_argument("input", type=Path)
    joint.add_argument("--signal-column", default="signal")
    joint.add_argument("--next-signal-column", default="next_signal")
    joint.add_argument("--return-column", default="next_return")
    joint.add_argument("--gamma", type=float, required=True)
    joint.add_argument("--trading-cost", type=float, required=True)
    joint.add_argument("--signal-innovation-sd", type=float, required=True)
    joint.add_argument("--return-noise-sd", type=float, required=True)
    joint.add_argument("--horizon", type=int, required=True)
    joint.add_argument("--alpha", type=float, default=.05, help="joint time-uniform miscoverage budget")
    joint.add_argument("--hurdle", type=float, default=0.)
    joint.add_argument("--phi-lower", type=float, default=-.999)
    joint.add_argument("--phi-upper", type=float, default=.999)
    joint.add_argument("--theta-rho", type=float, default=1.)
    joint.add_argument("--phi-rho", type=float, default=1.)
    joint.add_argument("--theta-cells", type=int, default=24)
    joint.add_argument("--phi-cells", type=int, default=96)
    joint.add_argument("--search-seed", type=int, default=1729)
    joint.add_argument("--terminal-penalty", type=float, default=0.)
    us = sub.add_parser("joint-unknown-scale-certificate", help="anytime-valid joint strength/persistence/noise-scale certificate with an eta-efficiency output")
    us.add_argument("input", type=Path)
    us.add_argument("--signal-column", default="signal")
    us.add_argument("--next-signal-column", default="next_signal")
    us.add_argument("--return-column", default="next_return")
    us.add_argument("--gamma", type=float, required=True)
    us.add_argument("--trading-cost", type=float, required=True)
    us.add_argument("--horizon", type=int, required=True)
    us.add_argument("--alpha", type=float, default=.05)
    us.add_argument("--hurdle", type=float, default=0.)
    us.add_argument("--efficiency-tolerance", type=float, default=.05, help="maximum oracle-relative regret eta")
    us.add_argument("--phi-lower", type=float, default=-.999)
    us.add_argument("--phi-upper", type=float, default=.999)
    us.add_argument("--theta-cells", type=int, default=20)
    us.add_argument("--phi-cells", type=int, default=80)
    us.add_argument("--terminal-penalty", type=float, default=0.)
    us.add_argument("--return-prior-mean", type=float, default=0.)
    us.add_argument("--return-prior-kappa", type=float, default=1.)
    us.add_argument("--return-prior-shape", type=float, default=1.)
    us.add_argument("--return-prior-scale", type=float, default=1.)
    us.add_argument("--signal-prior-mean", type=float, default=0.)
    us.add_argument("--signal-prior-kappa", type=float, default=1.)
    us.add_argument("--signal-prior-shape", type=float, default=1.)
    us.add_argument("--signal-prior-scale", type=float, default=1.)
    certify = sub.add_parser("certify", help="user-facing unknown-scale certificate plus the canonical necessary-history floor")
    certify.add_argument("input", type=Path)
    certify.add_argument("--signal-column", default="signal")
    certify.add_argument("--next-signal-column", default="next_signal")
    certify.add_argument("--return-column", default="next_return")
    certify.add_argument("--gamma", type=float, required=True)
    certify.add_argument("--trading-cost", type=float, required=True)
    certify.add_argument("--horizon", type=int, required=True)
    certify.add_argument("--alpha", type=float, default=.05)
    certify.add_argument("--hurdle", type=float, default=0.)
    certify.add_argument("--efficiency-tolerance", type=float, default=.05)
    certify.add_argument("--phi-lower", type=float, default=-.999)
    certify.add_argument("--phi-upper", type=float, default=.999)
    certify.add_argument("--theta-cells", type=int, default=20)
    certify.add_argument("--phi-cells", type=int, default=80)
    certify.add_argument("--terminal-penalty", type=float, default=0.)
    certify.add_argument("--return-prior-mean", type=float, default=0.)
    certify.add_argument("--return-prior-kappa", type=float, default=1.)
    certify.add_argument("--return-prior-shape", type=float, default=1.)
    certify.add_argument("--return-prior-scale", type=float, default=1.)
    certify.add_argument("--signal-prior-mean", type=float, default=0.)
    certify.add_argument("--signal-prior-kappa", type=float, default=1.)
    certify.add_argument("--signal-prior-shape", type=float, default=1.)
    certify.add_argument("--signal-prior-scale", type=float, default=1.)
    certify.add_argument("--period-unit", default="periods", help="display unit for the canonical history floor")

    zoo = sub.add_parser("zoo", help="batch unknown-scale certificates for a predeclared long-format signal zoo")
    zoo.add_argument("input", type=Path)
    zoo.add_argument("--name-column", default="name")
    zoo.add_argument("--signal-column", default="signal")
    zoo.add_argument("--next-signal-column", default="next_signal")
    zoo.add_argument("--return-column", default="next_return")
    zoo.add_argument("--trading-cost-column", default="trading_cost")
    zoo.add_argument("--trading-cost", type=float, help="fallback if the CSV has no per-signal cost")
    zoo.add_argument("--gamma", type=float, required=True)
    zoo.add_argument("--horizon", type=int, required=True)
    zoo.add_argument("--alpha", type=float, default=.05)
    zoo.add_argument("--hurdle", type=float, default=0.)
    zoo.add_argument("--efficiency-tolerance", type=float, default=.10)
    zoo.add_argument("--phi-lower", type=float, default=-.999)
    zoo.add_argument("--phi-upper", type=float, default=.999)
    zoo.add_argument("--theta-cells", type=int, default=8)
    zoo.add_argument("--phi-cells", type=int, default=24)
    zoo.add_argument("--period-unit", default="periods")
    zoo.add_argument("--output-csv", type=Path)

    ebh = sub.add_parser("e-bh-loading", help="secondary fixed-cut e-BH screen for exact zero loading across a predeclared zoo")
    ebh.add_argument("input", type=Path)
    ebh.add_argument("--name-column", default="name")
    ebh.add_argument("--signal-column", default="signal")
    ebh.add_argument("--next-signal-column", default="next_signal")
    ebh.add_argument("--return-column", default="next_return")
    ebh.add_argument("--trading-cost-column", default="trading_cost")
    ebh.add_argument("--alpha", type=float, default=.05)

    surv = sub.add_parser("survival", help="one-command Alpha Survival analysis from a strategy CSV")
    surv.add_argument("input", type=Path)
    surv.add_argument("--return-column", default="return")
    surv.add_argument("--edge-column", default=None, help="optional pre-outcome expected-return column constructed without future outcome data; omit to get an explicit lifetime-identification result")
    surv.add_argument("--periods-per-unit", type=float, required=True, help="observations per displayed time unit, e.g. 12 per year for monthly data")
    surv.add_argument("--alpha", type=float, default=.05)
    surv.add_argument("--power", type=float, default=.90)
    surv.add_argument("--trials", type=int, default=1)
    surv.add_argument("--correction", choices=("bonferroni","sidak"), default="bonferroni")
    surv.add_argument("--time-unit", default="years")
    surv.add_argument("--bootstrap-samples", type=int, default=1000)
    surv.add_argument("--bootstrap-level", type=float, default=.95)
    surv.add_argument("--block-length", type=int)
    surv.add_argument("--seed", type=int, default=2718)
    surv.add_argument("--output-json", type=Path)
    surv.add_argument("--output-html", type=Path)

    survival = sub.add_parser("survival-report", help="PM-facing Alpha Survival Frontier feasibility report")
    survival.add_argument("--sharpe", type=float, required=True, help="instantaneous Sharpe in units consistent with half-life")
    survival.add_argument("--half-life", type=float, required=True, help="alpha half-life in the declared time unit")
    survival.add_argument("--alpha", type=float, default=.05, help="family false-deployment budget")
    survival.add_argument("--power", type=float, default=.90, help="required deployment power under the alpha world")
    survival.add_argument("--trials", type=int, default=1, help="number of candidate strategies screened")
    survival.add_argument("--correction", choices=("bonferroni","sidak"), default="bonferroni")
    survival.add_argument("--time-unit", default="years")
    survival.add_argument("--output-json", type=Path)
    survival.add_argument("--output-html", type=Path)

    sdata = sub.add_parser("survival-data", help="data-driven Alpha Survival diagnostic from realized returns and an optional pre-outcome edge path")
    sdata.add_argument("input", type=Path)
    sdata.add_argument("--return-column", default="return")
    sdata.add_argument("--edge-column", default=None, help="pre-outcome expected-return column constructed without future outcome data; omit to receive a lifetime-identification warning instead of an invented half-life")
    sdata.add_argument("--periods-per-unit", type=float, required=True, help="observations per displayed time unit, e.g. 12 per year for monthly data")
    sdata.add_argument("--alpha", type=float, default=.05)
    sdata.add_argument("--power", type=float, default=.90)
    sdata.add_argument("--trials", type=int, default=1)
    sdata.add_argument("--correction", choices=("bonferroni","sidak"), default="bonferroni")
    sdata.add_argument("--time-unit", default="years")
    sdata.add_argument("--bootstrap-samples", type=int, default=1000)
    sdata.add_argument("--bootstrap-level", type=float, default=.95)
    sdata.add_argument("--block-length", type=int)
    sdata.add_argument("--seed", type=int, default=2718)
    sdata.add_argument("--output-json", type=Path)
    sdata.add_argument("--output-html", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "demo": output=_demo(args)
        elif args.command == "csv": output=_csv(args)
        elif args.command == "certificate": output=_certificate(args)
        elif args.command == "joint-certificate": output=_joint_certificate(args)
        elif args.command == "joint-unknown-scale-certificate": output=_unknown_scale_certificate(args)
        elif args.command == "certify": output=_certify(args)
        elif args.command == "zoo": output=_zoo(args)
        elif args.command == "e-bh-loading": output=_loading_ebh(args)
        elif args.command == "survival-report": output=_survival_report(args)
        elif args.command in {"survival","survival-data"}: output=_survival_data(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
