"""Predeclared predictor transformations for secondary zoo analyses."""
from __future__ import annotations
import math
import numpy as np


def fixed_ewma(values, *, half_life: float = 6.0) -> np.ndarray:
    """EWMA with a fixed half-life and no data-driven tuning.

    This helper is intended for a prespecified *secondary* persistent-predictor
    branch. Inputs must be finite; missingness must be resolved by the upstream
    predeclared data protocol rather than silently imputed here.
    """
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or x.size<1 or not np.isfinite(x).all():
        raise ValueError('values must be a finite nonempty one-dimensional array')
    if not math.isfinite(half_life) or half_life<=0:
        raise ValueError('half_life must be positive and finite')
    decay=2.0**(-1.0/half_life)
    out=np.empty_like(x)
    out[0]=x[0]
    for t in range(1,len(x)):
        out[t]=decay*out[t-1]+(1.0-decay)*x[t]
    return out
