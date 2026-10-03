# Reproduction

## Complete offline check

```bash
python -m pip install -e ".[test,plot]"
python reproduce/run.py
```

After installation, this command needs no network access. It runs the software tests, recomputes the descriptive funding estimates and their calendar-month bootstrap intervals from the included event table, checks them against the supplied numerical references, and regenerates all four manuscript figures.

## Figure rendering

```bash
python reproduction/make_journal_figures.py
python reproduction/make_journal_figures.py --font stix
python reproduction/make_journal_figures.py --font times
python reproduction/make_journal_figures.py --check-only
```

The default font mode uses Times New Roman when its four faces are available and otherwise emits a warning and selects Matplotlib's included STIXGeneral font with STIX mathematics. Explicit `--font times` requires a locally installed, licensed copy; `CAC_TIMES_FONT_DIR` can identify its folder. Font files are not redistributed. The supplied vector PDFs embed Times New Roman glyphs and the SVGs preserve their outlines. Figure metadata records the font actually used. Numerical checks do not require rendering.

The near-frontier panel reads `research/near_critical/two_look_results.csv`. It separates feasible two-look values, the asymptotic reference and the fixed-time benchmark. The numerical curves and visual encoding are documented in `journal_figure_values.json`. A pale-red stroke around deterministic curves is a visual accent; the funding panel's whiskers are statistical intervals. The two alternative figure command names forward to the unified generator.

## Descriptive funding persistence

```bash
python reproduction/verify_funding.py
python reproduction/ar1_predictor_audit.py
```

The included `data/funding_events.csv` contains 545 selected events, of which 543 have finite lifetime scores, across 12 series and 38 calendar-month clusters. Actual observation times and training/follow-up boundaries identify the sample. The verification command recomputes all reported estimates and 95% percentile intervals using 5,000 calendar-month cluster bootstrap resamples and seed 1729. It checks every scientific field with absolute and relative tolerances of 1e-10. Months with selected finite events form the resampling units; repeated sampled months repeat their events. Events receive equal weight in the pooled rank statistics.

The extraction uses BTC-USDT and ETH-USDT on Binance, Bybit, Gate, HTX, KuCoin and MEXC. It requires 540 prior eight-hour settlements, a current absolute-funding value at or above the trailing 95th percentile, the declared cooldown and 21 complete subsequent settlements. Settlement gaps must be within five minutes of eight hours. The no-intercept AR(1) fit and lifetime score are specified in `funding_persistence.py`; its finite-score sample is used consistently for the comparator analyses.

`data/funding_results.json` contains the complete statistical output. `reference_results.json` supplies the funding figure values. `ar1_predictor_reference.json` distinguishes the cumulative AR(1) forecast from its variance-standardized counterpart and gives their formulas. `data/funding_provenance.json` connects the event table, summaries, figure input and scripts by SHA256. These are descriptive associations with sign-oriented subsequent funding; execution, hedge, financing and collateral costs require separate trading-return analysis.

### Reconstruct the event table from pinned inputs

```bash
python reproduction/export_funding_events.py --data-root /path/to/funding-inputs --download
python reproduction/verify_funding.py --refresh
python reproduction/verify_funding.py
```

Inputs come from `supervik/historical-funding-rates-fetcher` at commit `66a085bc68147df2dd3360a25ac6e9f38e7077b5`. `data/funding_input_manifest.json` identifies all 12 source files and their Git blob hashes. The download command verifies these identities before extraction. Omit `--download` to use an existing copy in the specified directory. Raw third-party histories are not redistributed and remain subject to their source terms. `--refresh` writes statistics and provenance only after completing the fixed computation.

## Numerical theory

```bash
python research/near_critical/verify_critical_capacity.py
python research/theory/critical_robustness_check.py
python research/theory/finite_family_crosscheck.py
```

The numerical checks state their model and precision assumptions in their own READMEs. The finite-family crosscheck uses `research/experiments/gaussian_two_profile_results.json`. Floating-point checks complement the analytical proofs. The manuscript sources include all four figures needed for compilation.
