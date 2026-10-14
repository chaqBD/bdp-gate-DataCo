# Leaky Benchmarks: Governance Gates Catch Inflated Supply-Chain Results

*A case study on the DataCo Smart Supply Chain dataset, and a check for the
class of leak it exposed.*

## Abstract

A published benchmark number is only as trustworthy as the evaluation that
produced it. We show that a widely used public supply-chain dataset — DataCo
Smart Supply Chain — contains a textbook example of **target leakage built
from a combination of features**, where no single column crosses a
conventional leakage threshold but a simple pairwise combination of two
columns reconstructs the label almost exactly. A naive classifier that
includes both columns reports 0.974 ROC AUC; the same model without one of
them reports 0.731. The gap is not model skill — it is arithmetic. We show
that an existing single-feature leakage gate (`bdp-model-gate`'s
`LeakageCheck`) passes the naive model clean, and we contribute a check,
`CombinedFeatureLeakageCheck`, that catches it: an out-of-fold test for
whether small combinations of columns do what the whole model does. We
release it as both an installable plugin and a patch ready for upstream
review.

## 1. The problem

A "benchmark" result on a public dataset is often read as evidence about a
*method* — this model architecture, this feature-engineering approach, beats
that one. It is only that if the evaluation is sound. The specific failure
mode this paper is about is not new in kind — it is the same one every
data-leakage warning describes — but it has a shape that single-column
leakage checks do not cover: **a combination of features reconstructs the
target even though neither feature does so alone.**

This matters for governance tooling specifically. `bdp-model-gate` is a
pre-deployment gate built to catch exactly this class of problem before a
model reaches production — its `LeakageCheck` flags "a feature whose solo
predictive power rivals the whole model's," described in its own
documentation as "the classic signature of a leaked target." That check is
correct and useful. It is also, by construction, blind to a leak that no
single column carries alone.

## 2. Case study: DataCo's `Late_delivery_risk`

The DataCo Smart Supply Chain dataset is a public, widely reused tabular
dataset covering roughly 180,000 orders, commonly used as a benchmark for
supply-chain classification tasks — most often predicting
`Late_delivery_risk`, whether an order arrives later than scheduled.

The dataset carries two columns that, by construction, define that label:

| Column | Meaning | Known at scoring time? |
|---|---|---|
| `Days for shipment (scheduled)` | The promised shipping window | Yes — set when the order is placed |
| `Days for shipping (real)` | How long shipping actually took | **No** — only known once the order has arrived |

`Late_delivery_risk` is 1 exactly when `Days for shipping (real)` exceeds
`Days for shipment (scheduled)`. Using the *realised* shipping time to
predict whether a shipment will be late is a category error: it is only
available after the thing being predicted has already happened. That alone
would be a conventional, single-column leak — except that neither column
individually gives it away.

### 2.1 Neither column alone is suspicious

Scored individually against `Late_delivery_risk` on the same 0–1 power scale
`bdp-model-gate` uses internally (`|2·AUC − 1|`, a Gini-style measure where 0
is a coin flip and 1 is perfect separation):

| Column | Power, alone |
|---|---|
| `Days for shipping (real)` | **0.404** |
| `Days for shipment (scheduled)` | **0.401** |

`LeakageCheck`'s default threshold flags a column once its solo power
reaches 0.80 **and** 95% of the model's own power. Both columns sit at
roughly half that floor. Scored one at a time, there is nothing here to flag.

### 2.2 The pair reconstructs the label almost exactly

The simple difference of the two columns — `real − scheduled` — is a
different matter:

| Statistic | Value |
|---|---|
| Power of `real − scheduled` alone | **0.941** |
| Accuracy of the rule "late ⟺ `real − scheduled` > 0" | **97.5%** |

A rule with no learned parameters, built from two columns that each look
unremarkable on their own, recovers the label on 97.5% of the full dataset.
That is not a model finding anything about shipping logistics; it is
recovering the arithmetic the label was built from.

### 2.3 What this does to a benchmark number

We trained a RandomForest classifier on a realistic feature set — order
economics, shipping mode, customer segment, market, region, and the
scheduled shipping window — twice: once with `Days for shipping (real)`
included (the mistake several public notebooks on this dataset make), and
once without it.

| Model | Features | ROC AUC | Accuracy |
|---|---|---|---|
| **Naive** | includes `Days for shipping (real)` | **0.974** | **97.6%** |
| **Honest** | excludes it | **0.731** | **69.6%** |

Both numbers come from the same dataset, the same split, the same model
class. The 0.243 AUC gap is the leak, not an improvement in method. Any
paper, leaderboard, or vendor benchmark reporting late-delivery-prediction
results on this dataset that used both shipping-day columns as features is
reporting a number of this kind — and because dozens of public analyses of
this exact dataset do exactly that, this is not a hypothetical risk but an
already-circulating one.

*Reproduce this section:* `examples/dataco_leakage_demo.py` (see
[README](../README.md) for setup); raw output in
[`results/dataco_leakage_findings.json`](../results/dataco_leakage_findings.json).

## 3. Why the existing gate misses it, by design

Running `bdp-model-gate`'s default check suite against the naive model's
validation set, the stock `target_leakage` check reports:

> `OK` — *"no single feature approaches the model's own predictive power
> (0.948)"*

This is not a bug. `LeakageCheck` is explicitly, deliberately a
**single-column** test: for each feature, compare its solo power against the
model's. Two columns that are each individually unremarkable, but whose
*combination* is the whole story, satisfy that check by construction — there
is no column for it to flag, because the leak does not live in a column, it
lives in a relationship between two of them.

This gap generalises past DataCo. Pairs like "an actual value and the plan
or schedule it is compared against," "a running total and a prior-period
running total," or "a count and a rate derived from that same count," all
produce the same shape of leak in other structured-data domains — claims
(settlement date vs. report date), churn (contract end date vs. last-active
date), fraud (transaction amount vs. refunded amount). A governance gate
whose leakage check only ever looks at one column at a time will pass every
one of them.

## 4. `CombinedFeatureLeakageCheck`

We contribute a check that generalises `LeakageCheck` from single columns to
small combinations of them, designed to slot directly beside it:

**What it does.** For every candidate pair (optionally triple) of numeric
features, fit a plain least-squares combination of the two and score it on
the same 0–1 power scale `LeakageCheck` already uses — so the two checks'
numbers are directly comparable, and a report can say "no single feature
reaches it, but this pair does."

**Why least squares, not a black-box model.** The question being asked is
whether a *simple* arithmetic combination reconstructs the target — exactly
what a feature-engineering step or a careless join produces. A
gradient-boosted combo-finder would catch more exotic relationships and
explain none of them; a linear fit finds the boring, structural leaks, which
are also the common ones, and the finding is legible: "these two columns,
linearly combined."

**Why out-of-fold, not in-sample.** Fitting and scoring a 2–3 parameter model
on the same rows it is scored against inflates its apparent power, and that
inflation compounds with the number of combinations tried — test enough
pairs of pure noise and some will look like something in-sample. Each
combination is instead scored with k-fold cross-validation (5-fold by
default): every row's score comes from a fit that never saw that row. A flag
from this check is not itself a smaller version of the mistake it is
checking for.

**Why deterministic folds.** Fold assignment is derived from
`bdp_model_gate.uncertainty.canonical_order` — the same row-content ordering
the library's own bootstrap intervals use — rather than a random split.
Re-sorting or reshuffling the validation frame cannot move a row into a
different fold, and therefore cannot change the verdict. A gate that a
sorted CSV can quietly pass is not a gate.

**Why covered features are dropped first.** A pair made of two columns that
each already trip `LeakageCheck` on their own is not a new finding — it is
noise beside a result that check already reported. The candidate pool for
this check is restricted, by default, to columns `LeakageCheck` would pass;
every flag this check produces is something the existing one could not have
told you.

**Why a cap on candidate columns.** Testing every pair among *p* features is
`p·(p−1)/2` fits; triples are cubic. Past a configurable ceiling (30 columns
by default), only the features with the highest *individual* power are
combined — the same trade-off `bdp-model-gate`'s own `ProxyCorrelationCheck`
already makes for its comparison grid, logged the same way when it
truncates.

### 4.1 Validating it against the case study

Run against the naive model's validation context (`examples
/dataco_leakage_demo.py`), the two checks disagree exactly as predicted:

```
[                  OK] target_leakage: no single feature approaches the
                        model's own predictive power (0.948)
[COMBINED_LEAKAGE_RISK] combined_feature_leakage: Days for shipment
                        (scheduled) + Days for shipping (real) together
                        reach 0.948 of the target out-of-fold (100% of the
                        whole model's 0.948), while no single one of them
                        alone reaches 0.80
```

The reported combination is exactly the pair the case study identifies, with
no other feature in the dataset implicated — the check does not merely
detect "something is wrong," it names the two columns responsible, which is
what makes the finding actionable rather than merely alarming.

### 4.2 Synthetic validation and limitations

`tests/test_combined_feature_leakage.py` adds known-answer tests beyond the
real dataset: a synthetic pair shaped like the shipping-day case (each column
individually weak, their difference exact) is flagged; three independent
noise columns are not; a regression target built as the difference of two
independent features is flagged under the regression task; a column that is
already a near-exact copy of the target is excluded from the candidate pool
rather than re-reported.

Known limitations, stated rather than hidden: the check is linear and
numeric-only (a leak expressed as a ratio, an XOR of two categoricals, or a
three-way interaction not well-approximated by a linear combination of three
raw columns will not be caught by the default configuration); it costs
`O(p²)` least-squares fits for pairs, bounded by `max_features`; and like any
threshold-based check, its two-condition gate (absolute floor **and** ratio
to the model's own power) is a judgment call, not a law of nature — the same
honest caveat `bdp-model-gate`'s documentation already makes about its own
`LeakageCheck` thresholds.

## 5. Delivery: plugin today, PR for review

Two ways to get this check running, matching the two audiences for it:

**As a plugin, today.** `bdp-gate-dataco` (this repository) registers
`CombinedFeatureLeakageCheck` under the `bdp_model_gate.checks` entry-point
group — the extension mechanism `bdp-model-gate` already documents.
Installing it alongside `bdp-model-gate` is enough for
`default_structured_checks()` to discover it with no other wiring:

```bash
pip install bdp-model-gate bdp-gate-dataco
```

```python
from bdp_model_gate.structured import default_structured_checks
checks = default_structured_checks()  # combined_feature_leakage is in here
```

**As a pull request, for upstream review.** [`patches/0001-add-combined-feature-leakage-check.patch`](../patches/0001-add-combined-feature-leakage-check.patch)
adds the same check directly into
`bdp_model_gate/structured/validation_checks.py`, beside `LeakageCheck`,
reusing that module's own private reduction helpers instead of
reimplementing them — the natural shape of the change if the maintainer
prefers it live in core rather than as a separate dependency. It is offered
as a draft for discussion, not applied unilaterally: review, naming, and
threshold defaults are the maintainer's call, and this paper's job is to
make the case for the check, not to pre-empt that review.

## 6. Conclusion

A benchmark is a claim about the data as much as it is a claim about the
model. DataCo's `Late_delivery_risk` shows that a leak can hide in plain
sight from a single-feature check for the same reason it is dangerous: the
information is split across two unremarkable-looking columns, and recombined
by any model expressive enough to subtract. The fix is not a smarter model —
the naive RandomForest above is not wrong to find the signal, the signal
itself is the problem — it is a governance check that looks at the same
question `LeakageCheck` already asks, over combinations instead of columns.
We show that check catches the DataCo case precisely, with no other feature
implicated, and ship it in a form that costs nothing to try: install the
plugin, or read the patch.

---

*Repository: [github.com/chaqBD/bdp-gate-DataCo](https://github.com/chaqBD/bdp-gate-DataCo).
Built against [bdp-model-gate](https://github.com/vanjy-eng/model-gate) 0.6.0.*
