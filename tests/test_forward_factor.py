import json
import math

import pytest

from calculators import CALCULATORS
from calculators.forward_factor import SPEC, Inputs, Params, evaluate
from errors import CalculatorError


def test_known_values():
    # 30d at 40%, 60d at 30%: forward variance (0.09*60 - 0.16*30) / 30 = 0.02
    result = evaluate(Inputs(dte1=30, iv1=0.40, dte2=60, iv2=0.30))

    assert result.t1 == pytest.approx(30 / 365)
    assert result.t2 == pytest.approx(60 / 365)
    assert result.tv1 == pytest.approx(0.16 * 30 / 365)
    assert result.tv2 == pytest.approx(0.09 * 60 / 365)
    assert result.fwd_var == pytest.approx(0.02)
    assert result.fwd_vol == pytest.approx(math.sqrt(0.02))
    assert result.ff.value == pytest.approx((0.40 - math.sqrt(0.02)) / math.sqrt(0.02))
    assert result.ff.passed


def test_flat_term_structure_has_zero_ff():
    result = evaluate(Inputs(dte1=30, iv1=0.30, dte2=60, iv2=0.30))
    assert result.fwd_vol == pytest.approx(0.30)
    assert result.ff.value == pytest.approx(0.0)
    assert not result.ff.passed


def test_negative_forward_variance_leaves_vol_and_ff_undefined():
    result = evaluate(Inputs(dte1=30, iv1=0.60, dte2=60, iv2=0.30))
    assert result.fwd_var < 0
    assert result.fwd_vol is None
    assert result.ff is None


def test_zero_forward_vol_leaves_ff_undefined():
    result = evaluate(Inputs(dte1=30, iv1=0.0, dte2=60, iv2=0.0))
    assert result.fwd_vol == 0.0
    assert result.ff is None


def test_min_ff_param_drives_check():
    inputs = Inputs(dte1=30, iv1=0.40, dte2=60, iv2=0.30)
    assert not evaluate(inputs, Params(min_ff=2.0)).ff.passed
    assert evaluate(inputs, Params(min_ff=2.0)).ff.threshold == 2.0


@pytest.mark.parametrize(
    "inputs, message",
    [
        (Inputs(dte1=-1, iv1=0.3, dte2=60, iv2=0.3), "non-negative"),
        (Inputs(dte1=60, iv1=0.3, dte2=60, iv2=0.3), "DTE2 > DTE1"),
        (Inputs(dte1=60, iv1=0.3, dte2=30, iv2=0.3), "DTE2 > DTE1"),
        (Inputs(dte1=30, iv1=-0.1, dte2=60, iv2=0.3), "IVs must be non-negative"),
    ],
)
def test_invalid_inputs_raise(inputs, message):
    with pytest.raises(CalculatorError, match=message):
        evaluate(inputs)


def test_to_dict_is_json_serialisable():
    data = evaluate(Inputs(dte1=30, iv1=0.40, dte2=60, iv2=0.30)).to_dict()
    json.dumps(data)
    assert data["ff"]["op"] == ">="
    assert data["calculator_version"] == SPEC.version

    negative = evaluate(Inputs(dte1=30, iv1=0.60, dte2=60, iv2=0.30)).to_dict()
    assert negative["fwd_vol"] is None and negative["ff"] is None


def test_registry_lists_calculator():
    assert CALCULATORS["forward_factor"].SPEC is SPEC
