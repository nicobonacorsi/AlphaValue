"""Planning quantities that are consequences of the declared CAC model.

These helpers deliberately separate theorem-level necessary conditions from
forecasting claims.  In particular, the canonical one-percent lifetime barrier
is a *necessary* history floor in the canonical asymptotic experiment; it is not
an estimate of the sample size sufficient to certify an arbitrary real signal.
"""
from __future__ import annotations

import math

CANONICAL_ONE_PERCENT_RHO = 7.34088615005057626221967048575
CANONICAL_ONE_PERCENT_HISTORY_PER_HORIZON = CANONICAL_ONE_PERCENT_RHO / 2.0


def canonical_one_percent_history_floor(
    horizon: int,
    observed_periods: int | None = None,
    *,
    period_unit: str = "periods",
) -> dict:
    """Return the frozen G3 necessary-history floor for the canonical 1% problem.

    If deployment lasts ``horizon=T`` periods, the canonical scaling has ``T=2m``
    and the affinity barrier implies asymptotically ``n/T >= 3.670443...`` for
    any procedure that uniformly guarantees at most 1% oracle-relative regret.

    This function never labels the floor as sufficient sample size.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if observed_periods is not None and (not isinstance(observed_periods, int) or observed_periods < 0):
        raise ValueError("observed_periods must be a nonnegative integer")
    if not isinstance(period_unit, str) or not period_unit.strip():
        raise ValueError("period_unit must be a nonempty string")

    ratio = CANONICAL_ONE_PERCENT_HISTORY_PER_HORIZON
    continuous_floor = ratio * horizon
    integer_floor = int(math.ceil(continuous_floor - 1e-14))
    out = {
        "target_relative_regret": 0.01,
        "canonical_rho_root": CANONICAL_ONE_PERCENT_RHO,
        "necessary_history_per_deployment_horizon": ratio,
        "deployment_horizon": horizon,
        "period_unit": period_unit,
        "necessary_history_floor_continuous": continuous_floor,
        "necessary_history_floor_integer": integer_floor,
        "status": "necessary_not_sufficient",
        "scope": (
            "Asymptotic necessary-history barrier for the frozen canonical near-unit-root/high-friction "
            "experiment. It is not a sufficient sample-size calculation for this dataset and does not "
            "replace the finite-sample confidence set."
        ),
    }
    if observed_periods is not None:
        out["observed_periods"] = observed_periods
        out["additional_periods_to_floor"] = max(0, integer_floor - observed_periods)
        out["floor_comparison"] = (
            "below_canonical_necessary_floor"
            if observed_periods < integer_floor
            else "at_or_above_floor_not_sufficient"
        )
    return out
