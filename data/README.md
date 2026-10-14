# Data

This repo does not commit the DataCo Supply Chain dataset (~95 MB as a
single CSV) — only the small, derived artifacts it produced are checked in,
under [`results/`](../results/).

## Getting the dataset

The dataset is the public **DataCo Smart Supply Chain for Big Data Analysis**
set, most commonly found via Kaggle/Mendeley under that name (search for
"DataCo Smart Supply Chain"). Download `DataCoSupplyChainDataset.csv` and
place it at the repo root (or pass its path explicitly), then:

```bash
pip install -e ".[demo]"
python examples/dataco_leakage_demo.py /path/to/DataCoSupplyChainDataset.csv
```

[`DescriptionDataCoSupplyChain.csv`](DescriptionDataCoSupplyChain.csv) — the
dataset's own column glossary — is small enough to commit and is included
for reference.

## The columns this repo's case study turns on

| Column | What it is |
|---|---|
| `Late_delivery_risk` | The label: 1 if the order arrived later than scheduled. |
| `Days for shipment (scheduled)` | The promised shipping window — known **at order time**. |
| `Days for shipping (real)` | How long shipping actually took — known only **after delivery**, since that is what "late" is measured against. |

`Late_delivery_risk` is derived from comparing those last two columns. Used
together as model features, they reconstruct the label almost exactly; used
alone, neither reaches a single-feature leakage threshold. See
[`paper/leaky-benchmarks-governance-gates.md`](../paper/leaky-benchmarks-governance-gates.md)
for the full numbers and argument, and
[`examples/dataco_leakage_demo.py`](../examples/dataco_leakage_demo.py) to
reproduce them.
