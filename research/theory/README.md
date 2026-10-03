# Numerical theory checks

Run `python critical_robustness_check.py` for 80-digit frontier/margin identities,
and `python finite_family_crosscheck.py` for a Gauss-Hermite calculation independent
of the adaptive quadrature in `../experiments/`. Both scripts use the declared
Gaussian model. They do not estimate a lifetime or validate real trading profit.
The proofs are in `../../manuscript_source/main.tex` and the revised paper in `../../paper/Certified_Alpha_Capacity.pdf`.
Dependencies: NumPy, SciPy and mpmath.
