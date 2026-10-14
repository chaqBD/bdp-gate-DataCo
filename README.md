# bdp-gate-DataCo

**Leaky benchmarks: governance gates catch inflated supply-chain results.**

**[Live demo →](https://chaqBD.github.io/bdp-gate-DataCo/)**

A case study and a governance check, both built around one finding: the
public **DataCo Smart Supply Chain** dataset's `Late_delivery_risk` label is
defined by comparing two of its own feature columns. Neither column alone
looks suspicious to a standard single-feature leakage check — each reaches
only ~0.40 "power" on a 0–1 scale, well under any ordinary threshold — but
their simple difference reconstructs the label with 97.5% accuracy. A model
trained with both columns as features benchmarks at **0.974 ROC AUC**; the
same model without the leaking one benchmarks at **0.731**. The 0.243 gap is
not a better model — it's arithmetic.

This repo contains:

| | |
|---|---|
| [`paper/leaky-benchmarks-governance-gates.md`](paper/leaky-benchmarks-governance-gates.md) | The full write-up — paper or talk — with the numbers, the reasoning, and the design of the fix. **Start here.** |
| [`src/bdp_gate_dataco/`](src/bdp_gate_dataco/) | `CombinedFeatureLeakageCheck` — an installable [bdp-model-gate](https://github.com/vanjy-eng/model-gate) plugin that catches this class of leak: a small combination of columns that does what the whole model does, even though none of them does alone. |
| [`patches/`](patches/) | The same check as a patch against `bdp-model-gate` itself, prepared for upstream review. |
| [`examples/dataco_leakage_demo.py`](examples/dataco_leakage_demo.py) | Reproduces the case study end to end: trains both models, runs the gate, shows the stock check pass it and this one catch it. |
| [`tests/`](tests/) | Known-answer tests for the check (synthetic + regression cases), including one asserting the plugin entry point actually auto-registers. |
| [`results/dataco_leakage_findings.json`](results/dataco_leakage_findings.json) | Raw output from the last run of the demo above. |

## Why this exists

[bdp-model-gate](https://github.com/vanjy-eng/model-gate) is a pre-deployment
governance gate for ML models. Its `LeakageCheck` already catches the classic
single-column leak — a field whose solo predictive power rivals the whole
model's. It does not catch a leak **split across two columns**, each
individually unremarkable. That gap is not hypothetical: it's exactly how a
widely reused public benchmark dataset defines its own label, which means
every published result on this dataset using both shipping-day columns as
features is, to that extent, an inflated one.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[demo]"
pip install bdp-model-gate   # or use the wheel you already have

python examples/dataco_leakage_demo.py /path/to/DataCoSupplyChainDataset.csv
```

See [`data/README.md`](data/README.md) for where to get the dataset (not
committed here — ~95 MB).

### Using the check in your own gate

```bash
pip install bdp-model-gate bdp-gate-dataco
```

```python
from bdp_model_gate.structured import default_structured_checks

checks = default_structured_checks()  # combined_feature_leakage is in here automatically
```

No other wiring — `bdp-model-gate` discovers it via the
`bdp_model_gate.checks` entry-point group it already documents. To use it
standalone, or tune its thresholds:

```python
from bdp_gate_dataco import CombinedFeatureLeakageCheck, CombinedLeakageConfig

check = CombinedFeatureLeakageCheck(
    CombinedLeakageConfig(combo_sizes=(2, 3), max_features=40)
)
```

### Running the tests

```bash
pip install -e ".[dev,demo]"
pytest -q
```

## Repository layout

```
paper/      the write-up
src/        the plugin package (bdp_gate_dataco)
tests/      known-answer tests for the check
examples/   the reproducible demo against the real dataset
patches/    the same check as a diff against bdp-model-gate's own source
results/    small, derived output from the demo (no raw data)
data/       pointer to the dataset + its column glossary (DescriptionDataCoSupplyChain.csv)
```

## License

MIT — see [`LICENSE`](LICENSE). Built against
[bdp-model-gate](https://github.com/vanjy-eng/model-gate) 0.6.0 (also MIT).
