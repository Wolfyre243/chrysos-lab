from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CalculatorSpec:
    """Registry metadata for a calculator, mirroring the Chrysos analytics registry fields."""

    id: str
    name: str
    version: int  # bump whenever the formula changes
    scope: Literal["stock", "option", "spread"]
    requires: tuple[Literal["chain", "price_history", "fundamentals"], ...]
