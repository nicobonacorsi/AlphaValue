# AlphaValue — Certified Alpha Capacity

**Version 1.5.0** · Nicolò Bonacorsi

*Statistical Arbitrage When Learning Takes Time*

This package accompanies the theory of Certified Alpha Capacity: how much value can remain when learning and a trading opportunity's decay occur on comparable time scales. It includes the paper and proofs, the Python reference implementation, numerical constructions, synthetic examples and an interactive calculator.

## Scientific scope

The mathematical results describe a specified statistical experiment and reward model. The canonical Gaussian frontier separates feasible reliability contracts from information budgets that cannot support them. Near that frontier, residual information follows a sharp inverse-logarithmic law; cash payoff maps require their own uniform regularity assumptions.

Funding persistence is included as a descriptive empirical illustration. Association with future signed funding does not establish executable hedged-trade returns. **This release makes no claim of confirmatory trading-profit validation.** See SCIENTIFIC_STATUS.md for the distinction between mathematical guarantees, numerical checks and empirical description.

## Install and reproduce

```bash
python -m pip install -e ".[test,plot]"
python reproduce/run.py
```

The command runs the software tests, verifies the funding summaries from the supplied event table and regenerates all four manuscript figures. It requires no network access after installation. Times New Roman is used when available; otherwise the renderer reports its use of STIX, included with Matplotlib. Raw-data reconstruction and numerical theory checks are documented in reproduction/README.md.

## Contents

- paper/Certified_Alpha_Capacity.pdf — paper and proofs.
- manuscript_source/ — editable LaTeX and the four required figures.
- alphavalue/, tests/, examples/ — Python package, tests and synthetic examples.
- docs/ — website and reference calculator.
- reproduce/ and reproduction/ — verification entry point, figures and descriptive funding reconstruction.
- research/near_critical/, research/theory/, research/experiments/ — numerical constructions and Gaussian-model checks.

Compile manuscript_source/main.tex with pdflatex three times; its bibliography is embedded. The package contains the theoretical results and the reproducible descriptive funding illustration.

## Citation and licenses

Paper: arXiv:2610.01115 — https://arxiv.org/abs/2610.01115
Software concept DOI: https://doi.org/10.5281/zenodo.23070742
Repository: https://github.com/nicobonacorsi/AlphaValue

Use CITATION.cff for the software. Software is licensed under MIT; third-party data retain their upstream terms. The paper uses the arXiv non-exclusive distribution license.
