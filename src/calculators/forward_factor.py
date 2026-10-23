"""Forward volatility and forward factor between two expiries.

Forward variance identity, with T = DTE / 365 and sigma the annualised IV:

    sigma_fwd = sqrt( (sigma2^2 * T2 - sigma1^2 * T1) / (T2 - T1) )

Forward factor compares the front-month IV with the forward vol from front to back:

    FF = (sigma1 - sigma_fwd) / sigma_fwd

A high FF means the front month is rich versus the forward period, which favours a
calendar spread (sell front, buy back). FF >= ``min_ff`` passes.

Ported from a GUI calculator that took DTE1, IV1, DTE2 and IV2 by hand. ``evaluate``
is pure and takes one pair; choosing expiries and strikes from a live chain is the
job of ``scanners.forward_factor``.
"""

import math
from dataclasses import dataclass

from analytics.volatility import forward_variance, total_variance
from calculators.results import Check, Serializable, at_least
from calculators.spec import CalculatorSpec
from errors import CalculatorError

SPEC = CalculatorSpec(
    id="forward_factor",
    name="Forward Factor",
    version=1,
    scope="spread",
    requires=("chain",),
)


@dataclass(frozen=True)
class Params:
    min_ff: float = 0.2
    days_per_year: float = 365.0


@dataclass(frozen=True)
class Inputs:
    """Front (1) and back (2) legs. IVs are annualised decimals (0.245 = 24.5%)."""

    dte1: float
    iv1: float
    dte2: float
    iv2: float


@dataclass(frozen=True)
class Result(Serializable):
    t1: float  # years
    t2: float
    iv1: float
    iv2: float
    tv1: float  # total variance iv1^2 * t1
    tv2: float
    fwd_var: float
    fwd_vol: float | None  # None when forward variance is negative
    ff: Check | None  # None when fwd_vol is None or zero
    calculator_version: int = SPEC.version


def evaluate(inputs: Inputs, params: Params = Params()) -> Result:
    """Forward vol and forward factor for one front/back pair. Pure."""
    if inputs.dte1 < 0 or inputs.dte2 < 0:
        raise CalculatorError("DTEs must be non-negative.")
    if inputs.dte2 <= inputs.dte1:
        raise CalculatorError("Require DTE2 > DTE1 so that T2 > T1.")
    if inputs.iv1 < 0 or inputs.iv2 < 0:
        raise CalculatorError("IVs must be non-negative.")

    t1 = inputs.dte1 / params.days_per_year
    t2 = inputs.dte2 / params.days_per_year
    fwd_var = forward_variance(inputs.iv1, t1, inputs.iv2, t2)

    fwd_vol = math.sqrt(fwd_var) if fwd_var >= 0 else None
    ff = at_least((inputs.iv1 - fwd_vol) / fwd_vol, params.min_ff) if fwd_vol else None

    return Result(
        t1=t1,
        t2=t2,
        iv1=inputs.iv1,
        iv2=inputs.iv2,
        tv1=total_variance(inputs.iv1, t1),
        tv2=total_variance(inputs.iv2, t2),
        fwd_var=fwd_var,
        fwd_vol=fwd_vol,
        ff=ff,
    )
