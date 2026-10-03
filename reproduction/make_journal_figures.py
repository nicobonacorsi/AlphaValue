#!/usr/bin/env python3
"""Render the four manuscript figures from formulas and preserved reference data.

Run from any directory: python reproduction/make_journal_figures.py
Use --check-only to validate and export numerical content without matplotlib.
Fonts default to Times New Roman when installed, otherwise STIX with a warning.
Use --font times for strict published-font reproduction, or --font stix explicitly.
No raw market data, network access, or economic evaluation is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import warnings
from pathlib import Path
from statistics import NormalDist

import numpy as np

RED = "#C8102E"
PALE_RED = "#F4CDD4"
INK = "#111111"
DARK_GRAY = "#555555"
GRAY = "#949494"
ALPHA = 0.05
POWER = 0.90


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_content(root: Path) -> dict:
    normal = NormalDist()
    za = normal.inv_cdf(1 - ALPHA)
    zb = normal.inv_cdf(POWER)
    a_star = (za + zb) ** 2
    coefficient = a_star / 2 * ((1 + zb**2) * POWER + zb * normal.pdf(zb))
    reference_path = root / "reproduction/reference_results.json"
    two_look_path = root / "research/near_critical/two_look_results.csv"
    funding = json.loads(reference_path.read_text(encoding="utf-8"))["funding_persistence"]
    funding_bytes = json.dumps(funding, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    best: dict[float, float] = {}
    with two_look_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        epsilon, value = float(row["epsilon"]), float(row["policy_capacity"])
        if not (float(row["size"]) <= ALPHA + 1e-13 and float(row["power"]) >= POWER - 1e-13):
            raise ValueError("A preserved two-look policy violates the displayed feasibility constraint.")
        best[epsilon] = max(best.get(epsilon, -math.inf), value)
    if not best:
        raise ValueError("The preserved two-look table is empty.")
    epsilon = np.logspace(-6, -1, 200)
    ell = np.log(1 / epsilon)
    sharpes = np.linspace(0.75, 4.0, 400)
    multiplicities = [1, 10, 100, 1000]
    frontier_coefficients = {
        str(m): 2 * math.log(2) * (normal.inv_cdf(1 - ALPHA / m) + zb) ** 2
        for m in multiplicities
    }
    competitor_counts = np.arange(2, 101)
    cost_ratios = [0.0, 0.1, 0.5]
    funding_keys = ["spearman_lifetime_future", "spearman_edge_future", "spearman_t_future", "incremental_lifetime_given_edge"]
    interval_keys = ["lifetime", "edge", "t_stat", "incremental_lifetime_given_edge"]
    estimates = [funding[k] for k in funding_keys]
    intervals = [funding["calendar_month_cluster_ci95"][k] for k in interval_keys]
    if not all(lo <= point <= hi for point, (lo, hi) in zip(estimates, intervals)):
        raise ValueError("A displayed funding estimate lies outside its computed interval.")
    return {
        "schema": "cac-journal-figure-values-1",
        "inputs": {
            "funding_persistence": {
                "source": "reproduction/reference_results.json",
                "selected_object": "funding_persistence",
                "canonical_encoding": "UTF-8 JSON; sort_keys=True; separators=(',', ':'); ensure_ascii=True",
                "sha256": hashlib.sha256(funding_bytes).hexdigest(),
            },
            "two_look_results": {"source": "research/near_critical/two_look_results.csv", "sha256": sha256(two_look_path)},
        },
        "nearcritical": {
            "alpha": ALPHA, "power": POWER, "A_star": a_star, "K": coefficient,
            "epsilon": epsilon.tolist(),
            "sharp_asymptotic_reference": (coefficient / ell).tolist(),
            "fixed_time_benchmark": (POWER * epsilon).tolist(),
            "best_two_look_values": [{"epsilon": e, "policy_capacity": best[e]} for e in sorted(best)],
            "preserved_policy_rows": len(rows),
            "information_scale": (ell**-1).tolist(),
            "unit_cash_two_thirds_scale": (ell**(-2 / 3)).tolist(),
            "unit_cash_three_fifths_scale": (ell**(-3 / 5)).tolist(),
            "scope": "Asymptotic references and preserved feasible policies; not an unrestricted finite-horizon optimum. Cash exponents require the stated uniform reward-family assumptions.",
        },
        "survival_frontier": {
            "alpha_familywise": ALPHA, "power": POWER,
            "formula": "h_min(S;M)=2*log(2)*(Phi^-1(1-alpha/M)+Phi^-1(power))^2/S^2",
            "sharpe": sharpes.tolist(), "multiplicities": multiplicities,
            "half_life_years": {m: (c / sharpes**2).tolist() for m, c in frontier_coefficients.items()},
            "highlight": {"sharpe": 2.0, "M": 100, "half_life_years": frontier_coefficients["100"] / 4},
            "shading": "Thin pale-red stroke halo is decorative only; it is not a confidence interval or feasible-region boundary.",
        },
        "equilibrium_competition": {
            "formula": "r_N(q)=(q+1/N)/(1+q), q=psi/(a*kappa_crowd)",
            "competitor_counts": competitor_counts.tolist(), "cost_ratios": cost_ratios,
            "residual_fractions": {str(q): ((q + 1 / competitor_counts) / (1 + q)).tolist() for q in cost_ratios},
            "limits": {str(q): q / (1 + q) for q in cost_ratios},
            "shading": "Thin pale-red stroke halo is decorative only; it is not a confidence interval or finite-N contribution area.",
            "scope": "Symmetric positive-rent equilibrium branch of the exact linear-information benchmark.",
        },
        "funding_associations": {
            "finite_lifetime_score_events": funding["finite_lifetime_score_events"],
            "estimate_keys": funding_keys, "estimates": estimates,
            "interval_keys": interval_keys, "calendar_month_cluster_ci95": intervals,
            "scope": "Descriptive signed-funding associations and calendar-month bootstrap intervals computed from the supplied event table; no trading-profit claim.",
        },
    }


def select_figure_font(mode: str = "auto") -> dict:
    """Resolve real font faces; never report a substituted face as Times."""
    from matplotlib import font_manager

    if mode not in {"auto", "times", "stix"}:
        raise ValueError("Font mode must be auto, times, or stix.")

    def resolve_faces(family: str) -> dict:
        faces = {}
        for face, style, weight in (("regular", "normal", "normal"), ("italic", "italic", "normal"), ("bold", "normal", "bold"), ("bold_italic", "italic", "bold")):
            font_path = Path(font_manager.findfont(font_manager.FontProperties(family=family, style=style, weight=weight), fallback_to_default=False))
            actual_name = font_manager.FontProperties(fname=font_path).get_name()
            if actual_name != family:
                raise RuntimeError(f"The {face} face is {actual_name!r}, not {family!r}.")
            faces[face] = {"name": actual_name, "file": font_path.name, "sha256": sha256(font_path)}
        return faces

    selected = "stix" if mode == "stix" else "times"
    if selected == "times":
        try:
            # Use licensed system fonts without distributing their binaries.
            font_dir = Path(os.environ.get("CAC_TIMES_FONT_DIR", str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")))
            for basename in ("times.ttf", "timesi.ttf", "timesbd.ttf", "timesbi.ttf"):
                candidate = font_dir / basename
                if candidate.is_file():
                    font_manager.fontManager.addfont(candidate)
            faces = resolve_faces("Times New Roman")
        except (ValueError, RuntimeError, OSError) as exc:
            if mode == "times":
                raise RuntimeError("--font times requires Times New Roman. Install its four faces or set CAC_TIMES_FONT_DIR; use --font auto or --font stix for portable rendering.") from exc
            selected = "stix"
            warnings.warn("Times New Roman is unavailable; rendering with STIXGeneral and STIX mathematics. The supplied figure PDFs use Times New Roman. Use --font times to require it, or --font stix to select STIX explicitly.", RuntimeWarning, stacklevel=2)
    if selected == "stix":
        faces = resolve_faces("STIXGeneral")
    family = "Times New Roman" if selected == "times" else "STIXGeneral"
    return {"requested_mode": mode, "selected_mode": selected, "fallback_used": mode == "auto" and selected == "stix",
            "family": family, "mathtext_fontset": "custom" if selected == "times" else "stix", "faces": faces,
            "pdf": "Embedded TrueType subsets (fonttype 42)",
            "svg": f"Glyph outlines from {family}{' and STIX mathematics' if selected == 'stix' else ''}; no font binaries distributed"}


def plot_figures(root: Path, values: dict, font: str = "auto") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter, ScalarFormatter

    font_audit = select_figure_font(font)
    family = font_audit["family"]
    (root / "reproduction/journal_figure_font_audit.json").write_text(json.dumps(font_audit, indent=2) + "\n", encoding="utf-8")
    print("CAC_JOURNAL_FONT_AUDIT:" + json.dumps(font_audit, separators=(",", ":")))

    plt.rcParams.update({
        "font.family": family, "font.size": 9,
        "mathtext.fontset": font_audit["mathtext_fontset"], "mathtext.rm": family,
        "mathtext.it": f"{family}:italic", "mathtext.bf": f"{family}:bold",
        "mathtext.cal": family, "mathtext.sf": family,
        "mathtext.tt": family, "mathtext.fallback": None,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.65, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "lines.linewidth": 1.25, "lines.markersize": 4,
        "legend.fontsize": 8.3, "legend.frameon": False,
        "legend.handlelength": 2.2, "legend.labelspacing": 0.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "path", "svg.hashsalt": "cac-times-figures",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False,
    })
    if "mathtext.bfit" in plt.rcParams:
        plt.rcParams["mathtext.bfit"] = f"{family}:bold:italic"
    pdf_dir, svg_dir = root / "manuscript_source", root / "docs/assets"
    preview_dir = root / "reproduction/figure_previews"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, stem: str, svg_stem: str) -> None:
        font_description = "All text uses Times New Roman." if font_audit["selected_mode"] == "times" else "Text uses STIXGeneral and mathematics uses STIX."
        description = font_description + " Pale-red stroke halos are decorative only, with no statistical or mathematical interval meaning. In the funding figure only, whiskers show the computed calendar-month bootstrap 95% intervals."
        fig.savefig(pdf_dir / f"fig_{stem}.pdf", bbox_inches="tight", pad_inches=0.045,
                    metadata={"CreationDate": None, "ModDate": None, "Creator": "CAC figure reproduction", "Subject": description})
        fig.savefig(svg_dir / f"{svg_stem}.svg", bbox_inches="tight", pad_inches=0.045,
                    metadata={"Date": None, "Creator": "CAC figure reproduction", "Description": description})
        fig.savefig(preview_dir / f"{svg_stem}.png", bbox_inches="tight", pad_inches=0.045, dpi=200,
                    metadata={"Description": description})
        plt.close(fig)

    def finish_axes(ax) -> None:
        ax.grid(False, which="both")
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)

    def red_halo(line):
        line.set_path_effects([path_effects.Stroke(linewidth=3.1, foreground=PALE_RED), path_effects.Normal()])
        return line

    near = values["nearcritical"]
    eps = np.array(near["epsilon"])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.05), layout="constrained")
    red_halo(axes[0].loglog(eps, near["sharp_asymptotic_reference"], color=RED, label="Sharp asymptotic reference")[0])
    axes[0].loglog([v["epsilon"] for v in near["best_two_look_values"]],
                   [v["policy_capacity"] for v in near["best_two_look_values"]],
                   color=INK, marker="o", markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.8,
                   label="Feasible two-look values")
    axes[0].loglog(eps, near["fixed_time_benchmark"], color=GRAY, linestyle="--", label="Fixed-time benchmark")
    axes[0].set(xlabel=r"Distance from frontier, $\varepsilon=A-A_*$", ylabel="Expected information time saved")
    axes[0].legend(loc="center left", bbox_to_anchor=(0.02, 0.70), borderaxespad=0, fontsize=8.3)
    red_halo(axes[1].loglog(eps, near["information_scale"], color=RED, label="Information: exponent 1")[0])
    axes[1].loglog(eps, near["unit_cash_two_thirds_scale"], color=INK, linestyle="--", label="Unit cash: exponent 2/3")
    axes[1].loglog(eps, near["unit_cash_three_fifths_scale"], color=GRAY, linestyle=":", label="Unit cash: exponent 3/5")
    axes[1].set(xlabel=r"Distance from frontier, $\varepsilon=A-A_*$", ylabel="Asymptotic scale (coefficient 1)")
    axes[1].legend(loc="upper left", fontsize=8.3)
    for letter, ax in zip("ab", axes):
        ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold")
        ax.set_xlim(1e-6, 1e-1)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        finish_axes(ax)
    save(fig, "nearcritical", "near_critical")

    frontier = values["survival_frontier"]
    s = np.array(frontier["sharpe"])
    h = frontier["half_life_years"]
    fig, ax = plt.subplots(figsize=(6.7, 3.5), layout="constrained")
    ax.set_yscale("log")
    frontier_lines = {}
    for m, color, style in [(1, INK, ":"), (10, DARK_GRAY, "--"), (100, RED, "-"), (1000, GRAY, "-.")]:
        line = ax.plot(s, h[str(m)], color=color, linestyle=style, label=f"$M={m:,}$", lw=1.4 if m == 100 else 1.15)[0]
        frontier_lines[m] = line
        if m == 100:
            line.set_marker("o")
            line.set_markevery(np.linspace(0, len(s) - 1, 10, dtype=int).tolist())
            line.set_markersize(3.8)
            line.set_markerfacecolor("white")
            line.set_markeredgecolor(RED)
            line.set_markeredgewidth(0.8)
            red_halo(line)
    anchor = frontier["highlight"]
    ax.plot(2, anchor["half_life_years"], "o", color=RED, ms=5, markerfacecolor="white", markeredgecolor=RED, markeredgewidth=1.0, zorder=5)
    ax.annotate(f"$S=2$, $M=100$: $h_{{\\min}}={anchor['half_life_years']:.2f}$ years",
                xy=(2, anchor["half_life_years"]), xytext=(2.28, 12),
                fontsize=8.5, color=INK, ha="left",
                arrowprops={"arrowstyle": "-", "color": DARK_GRAY, "lw": 0.55, "shrinkA": 3, "shrinkB": 4})
    ax.set(xlim=(0.75, 4), ylim=(1, 60), xlabel="Instantaneous Sharpe ratio, $S$", ylabel="Minimum half-life, $h_{\\min}$ (years)")
    ax.set_xticks(np.arange(1, 4.1, 0.5))
    ax.set_yticks([1, 10])
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(handles=[frontier_lines[m] for m in (1, 10, 1000, 100)], loc="upper right", ncol=2)
    finish_axes(ax)
    save(fig, "survival_frontier", "survival_frontier")

    competition = values["equilibrium_competition"]
    ns = np.array(competition["competitor_counts"])
    fig, ax = plt.subplots(figsize=(6.7, 3.0), layout="constrained")
    ax.set_xscale("log")
    for q, color, style in [(0.0, INK, "--"), (0.1, RED, "-"), (0.5, GRAY, "-.")]:
        label = "$q=0$" if q == 0 else f"$q={q:g}$"
        line = ax.plot(ns, competition["residual_fractions"][str(q)], color=color, linestyle=style, label=label)[0]
        if q == 0.1:
            red_halo(line)
        if q:
            ax.axhline(competition["limits"][str(q)], color=color, ls=":", lw=0.8, alpha=0.75)
    ax.set(xlim=(2, 100), ylim=(0, 0.7), xlabel="Number of arbitrageurs, $N$", ylabel="Residual information fraction")
    ax.set_xticks([2, 5, 10, 20, 50, 100])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.legend(loc="upper right", title=r"$q=\psi/(a\kappa_{\rm crowd})$", title_fontsize=8)
    finish_axes(ax)
    save(fig, "equilibrium_competition", "equilibrium_competition")

    associations = values["funding_associations"]
    labels = ["Lifetime score", "Current edge magnitude", "Historical $t$ statistic", "Lifetime component | edge"]
    fig, ax = plt.subplots(figsize=(7.0, 2.8), layout="constrained")
    for index, (value, (lo, hi)) in enumerate(zip(associations["estimates"], associations["calendar_month_cluster_ci95"])):
        y = 3 - index
        color = RED if index in (0, 3) else DARK_GRAY
        ax.errorbar(value, y, xerr=[[value - lo], [hi - value]], fmt="D" if index == 3 else "o",
                    color=color, ecolor=color, capsize=2.5, capthick=0.8, elinewidth=1.0,
                    markersize=4.8, markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.85, zorder=3)
        ax.text(hi + 0.018, y, f"{value:.3f}", va="center", color=color, fontsize=8)
    ax.axvline(0, color=GRAY, ls="--", lw=0.65)
    ax.set_yticks([3, 2, 1, 0], labels)
    ax.tick_params(axis="y", pad=8)
    ax.set(xlim=(-0.04, 0.72), ylim=(-0.5, 3.5),
           xlabel="Spearman association with next-seven-day signed funding")
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    finish_axes(ax)
    save(fig, "lifetime_benchmark", "lifetime_benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Package root containing the preserved public inputs.")
    parser.add_argument("--check-only", action="store_true", help="Validate and write numerical content without generating figures.")
    parser.add_argument("--font", choices=("auto", "times", "stix"), default="auto", help="auto: Times when available, otherwise STIX with a warning; times: require Times; stix: use bundled STIX. Ignored by --check-only.")
    args = parser.parse_args()
    root = args.root.resolve()
    values = numerical_content(root)
    destination = root / "reproduction/journal_figure_values.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.check_only:
        plot_figures(root, values, font=args.font)
    near, frontier, equilibrium = (values[k] for k in ("nearcritical", "survival_frontier", "equilibrium_competition"))
    compact = {
        "schema": values["schema"], "inputs": values["inputs"],
        "nearcritical": {k: near[k] for k in ("alpha", "power", "A_star", "K", "best_two_look_values", "preserved_policy_rows", "scope")},
        "survival_frontier": {k: frontier[k] for k in ("alpha_familywise", "power", "formula", "multiplicities", "highlight", "shading")},
        "equilibrium_competition": {k: equilibrium[k] for k in ("formula", "cost_ratios", "limits", "shading", "scope")},
        "funding_associations": values["funding_associations"],
        "full_values_sha256": hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
    }
    compact["nearcritical"]["epsilon_grid"] = {"min": min(near["epsilon"]), "max": max(near["epsilon"]), "points": len(near["epsilon"])}
    compact["survival_frontier"]["sharpe_grid"] = {"min": min(frontier["sharpe"]), "max": max(frontier["sharpe"]), "points": len(frontier["sharpe"])}
    compact["equilibrium_competition"]["competitor_grid"] = {"min": min(equilibrium["competitor_counts"]), "max": max(equilibrium["competitor_counts"]), "points": len(equilibrium["competitor_counts"])}
    print("CAC_JOURNAL_FIGURE_VALUES:" + json.dumps(compact, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
