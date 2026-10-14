"""Known-answer tests for CombinedFeatureLeakageCheck.

Same style as bdp_model_gate's own tests/test_validation_checks.py: build a
case where the correct verdict is derivable on paper, then assert it. The
first case below is a synthetic stand-in for the real one that motivated
this check — see paper/leaky-benchmarks-governance-gates.md — with the same
shape: two columns, each too weak alone to trip LeakageCheck, whose
difference reconstructs the label exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bdp_model_gate import StructuredGateContext
from bdp_model_gate.structured.validation_checks import LeakageCheck

from bdp_gate_dataco.checks import CombinedFeatureLeakageCheck, CombinedLeakageConfig


def _context(X, y, y_pred, **kwargs):
    return StructuredGateContext(
        model=None,
        X=X,
        y_true=y,
        y_pred=y_pred,
        predict_fn=lambda frame: np.zeros(len(frame)),
        **kwargs,
    )


def _shipping_style_frame(n=2000, seed=7):
    """Two columns shaped like DataCo's shipping-day pair: each weak alone,
    their difference exact. `scheduled` is independent noise; `real` is
    `scheduled` plus a small signed offset whose *sign* is the label — same
    construction as `Late_delivery_risk = (real > scheduled)` upstream."""
    rng = np.random.default_rng(seed)
    scheduled = rng.integers(1, 30, n).astype(float)
    offset = rng.choice([-1.0, 1.0], n) * rng.integers(1, 3, n).astype(float)
    real = scheduled + offset
    late = (offset > 0).astype(int)
    X = pd.DataFrame({"scheduled": scheduled, "real": real})
    return X, late


def test_combo_is_flagged_when_no_single_feature_is():
    X, y = _shipping_style_frame()
    y_pred = y.astype(float)  # a model that reproduces the label exactly

    single = LeakageCheck().run(_context(X, y, y_pred))
    assert [r.flag for r in single] == ["OK"], (
        "sanity check on the fixture: neither column alone should trip the "
        "existing single-feature check"
    )

    combined = CombinedFeatureLeakageCheck().run(_context(X, y, y_pred))
    assert len(combined) == 1
    result = combined[0]
    assert result.flag == "COMBINED_LEAKAGE_RISK"
    assert set(result.metadata["features"]) == {"scheduled", "real"}
    assert result.metadata["combo_power"] > 0.9
    assert result.blocking is True


def test_independent_noise_columns_are_not_flagged():
    rng = np.random.default_rng(3)
    n = 1000
    X = pd.DataFrame(
        {
            "noise_a": rng.normal(size=n),
            "noise_b": rng.normal(size=n),
            "noise_c": rng.normal(size=n),
        }
    )
    y = rng.integers(0, 2, n)
    y_pred = rng.random(n)

    results = CombinedFeatureLeakageCheck().run(_context(X, y, y_pred))
    assert [r.flag for r in results] == ["OK"]


def test_not_applicable_without_y_true_or_y_pred():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    context = StructuredGateContext(model=None, X=X, y_true=None, y_pred=None)
    results = CombinedFeatureLeakageCheck().run(context)
    assert results[0].flag == "NOT_APPLICABLE"


def test_not_applicable_with_fewer_than_two_numeric_columns():
    X = pd.DataFrame({"only_col": np.arange(100, dtype=float)})
    y = np.random.default_rng(0).integers(0, 2, 100)
    results = CombinedFeatureLeakageCheck().run(_context(X, y, y.astype(float)))
    assert results[0].flag == "NOT_APPLICABLE"
    assert "nothing to combine" in results[0].detail


def test_not_applicable_below_min_rows():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    y = np.array([0, 1, 0])
    results = CombinedFeatureLeakageCheck().run(_context(X, y, y.astype(float)))
    assert results[0].flag == "NOT_APPLICABLE"


def test_skip_covered_features_excludes_single_column_leaks():
    """A column that is already a near-exact copy of the target should be
    dropped from the candidate pool, not re-reported paired with noise."""
    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 2, n)
    X = pd.DataFrame(
        {
            "leaked_copy": y.astype(float) * 1000.0 + rng.normal(scale=0.01, size=n),
            "noise": rng.normal(size=n),
        }
    )
    y_pred = y.astype(float)

    default = CombinedFeatureLeakageCheck(CombinedLeakageConfig(skip_covered_features=True))
    results = default.run(_context(X, y, y_pred))
    assert [r.flag for r in results] == ["NOT_APPLICABLE"]
    assert "already flag alone" in results[0].detail or "already" in results[0].detail

    unfiltered = CombinedFeatureLeakageCheck(CombinedLeakageConfig(skip_covered_features=False))
    results_unfiltered = unfiltered.run(_context(X, y, y_pred))
    assert any(r.flag == "COMBINED_LEAKAGE_RISK" for r in results_unfiltered)


def test_regression_target_is_supported():
    rng = np.random.default_rng(42)
    n = 1000
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    target = a - b  # perfectly reconstructed by the pair, not by either alone
    X = pd.DataFrame({"a": a, "b": b})
    y_pred = target.copy()

    single = LeakageCheck().run(_context(X, target, y_pred, task="regression"))
    assert [r.flag for r in single] == ["OK"]

    combined = CombinedFeatureLeakageCheck().run(_context(X, target, y_pred, task="regression"))
    assert any(r.flag == "COMBINED_LEAKAGE_RISK" for r in combined)


def test_multiclass_without_class_order_is_not_applicable():
    rng = np.random.default_rng(5)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = rng.integers(0, 3, n)
    y_pred = rng.integers(0, 3, n)
    results = CombinedFeatureLeakageCheck().run(
        _context(X, y, y_pred, task="multiclass")
    )
    assert results[0].flag == "NOT_APPLICABLE"


def test_entry_point_is_registered():
    """The whole point of shipping this as a plugin: installing the package
    should be enough for default_structured_checks() to find it."""
    from bdp_model_gate.structured import default_structured_checks

    checks = default_structured_checks()
    names = {c.name for c in checks}
    assert "combined_feature_leakage" in names


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
