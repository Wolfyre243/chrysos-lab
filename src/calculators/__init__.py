from types import ModuleType

from calculators import calendar_iv_crush, forward_factor
from calculators.spec import CalculatorSpec

# Calculator id -> module. Each module exposes SPEC, Params, Inputs, Result and evaluate(),
# plus run() when the calculator fetches its own data.
CALCULATORS: dict[str, ModuleType] = {
    calendar_iv_crush.SPEC.id: calendar_iv_crush,
    forward_factor.SPEC.id: forward_factor,
}

__all__ = ["CALCULATORS", "CalculatorSpec", "calendar_iv_crush", "forward_factor"]
