from types import ModuleType

from calculators import calendar_iv_crush
from calculators.spec import CalculatorSpec

# Calculator id -> module. Each module exposes SPEC, Params, Inputs, Result, evaluate() and run().
CALCULATORS: dict[str, ModuleType] = {
    calendar_iv_crush.SPEC.id: calendar_iv_crush,
}

__all__ = ["CALCULATORS", "CalculatorSpec", "calendar_iv_crush"]
