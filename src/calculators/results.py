"""Result building blocks shared by calculators and scanners."""

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal


@dataclass(frozen=True)
class Check:
    value: float
    threshold: float
    op: Literal[">=", "<="]
    passed: bool


def at_least(value: float, threshold: float) -> Check:
    return Check(value, threshold, ">=", value >= threshold)


def at_most(value: float, threshold: float) -> Check:
    return Check(value, threshold, "<=", value <= threshold)


class Serializable:
    """Mixin for result dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (dates as ISO strings)."""
        return jsonable(self)


def jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {name: jsonable(getattr(obj, name)) for name in obj.__dataclass_fields__}
    if isinstance(obj, (list, tuple)):
        return [jsonable(item) for item in obj]
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj
