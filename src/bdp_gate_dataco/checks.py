"""Does a *combination* of columns do what the whole model does?

`bdp_model_gate.structured.validation_checks.LeakageCheck` asks whether any
single feature's solo predictive power rivals the model's own. That catches
a settlement amount joined back onto a claims model, or a `closed_reason`
column on churn — the classic single-field leak.

It does not catch a leak built from *several* features together, each one
individually unremarkable. That gap is not hypothetical: it is how the
DataCo Supply Chain dataset's own `Late_delivery_risk` label is defined.

    Days for shipping (real)      — power 0.40 alone (well under any
    Days for shipment (scheduled) — power 0.40 alone   single-feature bar)

Neither column alone approaches `LeakageCheck`'s default threshold
(`leakage_min_power=0.80`). Their difference — `real - scheduled` — recovers
the label with power 0.94 and a simple sign rule gets 97.5% accuracy,
because that comparison is literally how `Late_delivery_risk` was derived
upstream of the CSV. A model "learning to predict late delivery" from these
two columns is learning to subtract them; every benchmark published against
this target using both shipping-day columns as features is, to that
extent, measuring arithmetic rather than forecasting. See
`paper/leaky-benchmarks-governance-gates.md` in this repository for the full
case study and the numbers behind this docstring.

`CombinedFeatureLeakageCheck` generalises `LeakageCheck` from single columns
to small combinations of them. For every candidate pair (optionally triple)
of numeric features, it fits a plain least-squares combination of the two
and scores it **out-of-fold** — never in-sample — on the same 0-1 power
scale `LeakageCheck` already uses, so the two checks' numbers are directly
comparable and a report can show "no single feature reaches it, but this
pair does."

Design choices, and why:

* **Least squares, not a black-box model.** The question is whether a
  *simple* arithmetic combination reconstructs the target — exactly the
  thing a feature-engineering step or a careless join produces. A
  gradient-boosted combo-finder would catch more and explain less; this
  finds the boring, structural leaks, which are also the common ones.
* **Out-of-fold, not in-sample.** Fitting and scoring a 2-3 parameter model
  on the same rows inflates its apparent power, and that inflation grows
  with the number of combinations tried. K-fold cross-validation (default
  5-fold) reports the combination's power on rows it did not see, so a
  flag here is not itself a smaller version of the mistake it is checking
  for.
* **Deterministic folds.** Assigned from `bdp_model_gate.uncertainty
  .canonical_order` — the same row-content ordering the library's own
  bootstrap uses — rather than a random split, so re-sorting the input
  cannot change which rows share a fold and therefore cannot move a
  verdict. Resampling or reshuffling a validation set must not be a way to
  dodge this check.
* **Candidates that already fail alone are dropped first.** A pair made of
  two columns that each already trip `LeakageCheck` is not a new finding;
  reporting it again would just be noise beside the single-feature result.
  This check's candidate pool is restricted to columns `LeakageCheck`
  passes, so every flag here is something that check could not have told
  you (`CombinedLeakageConfig.skip_covered_features`, on by default).
* **A cap on how many features are combined.** Testing every pair among p
  features is `p * (p - 1) / 2` least-squares fits; triples are cubic.
  `CombinedLeakageConfig.max_features` (default 30) keeps that bounded by
  pre-selecting the features with the highest *individual* power when
  there are more candidates than that — the same trade-off
  `ProxyCorrelationCheck` makes for its own comparison grid, and logged the
  same way when it bites.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from bdp_model_gate.classes import favourable_mask, resolve_favourable
from bdp_model_gate.core.base import BaseCheck, CheckResult
from bdp_model_gate.metrics import to_class_labels
from bdp_model_gate.stats import pearson_r, rank_auc
from bdp_model_gate.task import ALL_TASKS, MULTICLASS, REGRESSION, resolve_task
from bdp_model_gate.uncertainty import canonical_order

# Standard logging, not bdp_model_gate's own get_logger() — that helper lives
# in a private (`_logging`) module, and this package only depends on
# bdp_model_gate's public surface. Composes the same way: the README notes
# the library "uses the standard logging module throughout".
logger = logging.getLogger("bdp_gate_dataco.checks")

#: A combination needs enough rows per fold for the fit itself to mean
#: anything, not just enough rows overall — stricter than the single-column
#: LeakageCheck's floor of 20 for exactly that reason.
_MIN_ROWS_PER_FOLD = 10


@dataclass
class CombinedLeakageConfig:
    """Thresholds for `CombinedFeatureLeakageCheck`.

    Mirrors `bdp_model_gate.config.ValidationConfig`'s leakage fields
    (`leakage_ratio`, `leakage_min_power`) so the two checks agree on what
    "suspiciously close to the model" means — only the thing being measured
    differs. Kept as its own dataclass, not folded into `ValidationConfig`,
    because a plugin check is instantiated with no arguments
    (`bdp_model_gate.registry.discover_plugin_checks`) and has to carry its
    own defaults; see that module's docstring.
    """

    #: Column subset sizes to test. (2,) by default; add 3 deliberately —
    #: it costs roughly `max_features` times as many fits as pairs do.
    combo_sizes: tuple[int, ...] = (2,)
    #: A combination is a suspected leak when its out-of-fold power reaches
    #: this fraction of the whole model's own power.
    leakage_ratio: float = 0.95
    #: ...and only when that power is high in absolute terms. Same scale and
    #: same default as `ValidationConfig.leakage_min_power`.
    leakage_min_power: float = 0.80
    #: Folds for the out-of-fold power estimate. More folds trade a smaller
    #: per-fold training set for a less noisy pooled estimate; 5 is the usual
    #: default for this trade-off.
    cv_folds: int = 5
    #: Above this many numeric candidate columns, only the `max_features`
    #: with the highest *individual* power are combined — otherwise the
    #: number of pairs (or triples) grows too fast to score inside a gate
    #: run. Logged when it truncates.
    max_features: int = 30
    #: Drop a column from the candidate pool if it already alone reaches
    #: `leakage_min_power` and `leakage_ratio * model_power` — i.e. if
    #: `LeakageCheck` would already flag it solo. Keeps this check's findings
    #: disjoint from that one's: everything reported here is a combination
    #: no single feature in it already explained.
    skip_covered_features: bool = True


def _not_applicable(check: BaseCheck, reason: str) -> list[CheckResult]:
    return [CheckResult(check.name, check.category, "NOT_APPLICABLE", reason, check.blocking)]


def _binary_target(context, task: str) -> np.ndarray | None:
    """`y_true` as 0/1 against the favourable outcome, or None for regression.

    Same reduction `bdp_model_gate.structured.validation_checks` uses for
    `LeakageCheck`, rebuilt here from the library's public primitives
    (`classes.resolve_favourable`/`favourable_mask`, `metrics.to_class_labels`)
    rather than imported from that module's private helpers, so this check
    has no dependency on bdp_model_gate internals that could change without
    notice — only on the same public contract every other plugin has.
    """
    if task == REGRESSION:
        return None
    if task == MULTICLASS:
        class_order = getattr(context, "class_order", None)
        favourable = resolve_favourable(
            getattr(context, "favourable_classes", None), class_order, MULTICLASS
        )
        if not favourable:
            return None
        labels = to_class_labels(context.y_true, class_order)
        return favourable_mask(labels, favourable).astype(float)
    return np.asarray(context.y_true, dtype=float)


def _model_score(context, task: str) -> np.ndarray | None:
    """`y_pred` reduced to a 1-D score on the same footing as `_binary_target`."""
    raw = np.asarray(context.y_pred)
    if task != MULTICLASS:
        return raw.astype(float) if raw.ndim == 1 else None

    class_order = getattr(context, "class_order", None)
    favourable = resolve_favourable(
        getattr(context, "favourable_classes", None), class_order, MULTICLASS
    )
    if not favourable:
        return None
    if raw.ndim == 2:
        if class_order is None:
            return None
        by_model_order = sorted(class_order, key=str)
        columns = [by_model_order.index(c) for c in favourable]
        return raw[:, columns].sum(axis=1).astype(float)
    return favourable_mask(to_class_labels(raw, class_order), favourable).astype(float)


def _power(scores: np.ndarray, target: np.ndarray, binary: bool) -> float | None:
    """How much of the target a 1-D score explains, 0-1, same scale as
    `LeakageCheck`: |2*AUC-1| for classification, |r| for regression."""
    if binary:
        auc = rank_auc(target, scores)
        return None if np.isnan(auc) else abs(2 * auc - 1)
    return abs(pearson_r(scores, target))


def _already_covered(power: float | None, model_power: float, config: CombinedLeakageConfig) -> bool:
    if power is None:
        return False
    return power >= config.leakage_min_power and power >= config.leakage_ratio * model_power


def _fold_ids(frame: pd.DataFrame, k: int) -> np.ndarray:
    """Deterministic, order-invariant fold id (0..k-1) per row.

    Built from `canonical_order` — rows sorted by content, not position — so
    permuting the validation frame cannot move a row into a different fold
    and therefore cannot change which rows the combination is scored on.
    """
    order = canonical_order(frame)
    fold_of_position = np.empty(len(frame), dtype=int)
    fold_of_position[order] = np.arange(len(frame)) % k
    return fold_of_position


def _oof_combo_power(
    columns: pd.DataFrame, target: np.ndarray, binary: bool, k: int
) -> float | None:
    """Out-of-fold power of a least-squares combination of `columns`.

    Each fold's rows are scored from a fit that never saw them: `beta` comes
    from `np.linalg.lstsq` on the other folds, and the held-out fold's score
    is `[1, columns] @ beta`. Returns None if any fold is too small to fit,
    or the pooled out-of-fold scores carry no computable power (e.g. a
    fold's target is constant).
    """
    n = len(columns)
    folds = _fold_ids(columns, k)
    design = np.column_stack([np.ones(n), columns.to_numpy(dtype=float)])
    oof = np.full(n, np.nan)

    for fold in range(k):
        test = folds == fold
        train = ~test
        if train.sum() < design.shape[1] + _MIN_ROWS_PER_FOLD or test.sum() == 0:
            return None
        beta, *_ = np.linalg.lstsq(design[train], target[train], rcond=None)
        oof[test] = design[test] @ beta

    if np.isnan(oof).any():
        return None
    return _power(oof, target, binary)


class CombinedFeatureLeakageCheck(BaseCheck):
    """Does a small combination of columns do what the whole model does?

    See the module docstring for the DataCo case study that motivated this
    check and the reasoning behind each design choice. In short:
    `LeakageCheck` asks the question for one column at a time; this asks it
    for pairs (and optionally triples) of columns, scored out-of-fold so a
    flag reflects real, reusable signal rather than the fit's own freedom to
    find *some* line through the data.
    """

    name = "combined_feature_leakage"
    category = "validation"
    blocking = True
    supported_tasks = ALL_TASKS

    def __init__(self, config: CombinedLeakageConfig | None = None):
        self.config = config or CombinedLeakageConfig()

    def run(self, context) -> list[CheckResult]:
        cfg = self.config
        if context.y_true is None or context.y_pred is None:
            return _not_applicable(
                self,
                "no y_true/y_pred — combined-feature leakage is measured by comparing "
                "a fitted combination of features against the model's own power",
            )

        task = resolve_task(context)
        binary = task != REGRESSION
        if binary:
            target = _binary_target(context, task)
            if target is None:
                return _not_applicable(
                    self,
                    "multiclass combined-feature leakage needs context.class_order or "
                    "context.favourable_classes, so the target can be reduced to the one "
                    "outcome a combination might be leaking",
                )
        else:
            target = np.asarray(context.y_true, dtype=float)

        scores = _model_score(context, task)
        if scores is None:
            return _not_applicable(
                self,
                "could not reduce y_pred to a single score to compare combinations "
                "against — for multiclass, set context.class_order",
            )
        model_power = _power(scores, target, binary)
        if model_power is None or model_power <= 0:
            return _not_applicable(
                self,
                "the model's predictions carry no signal against y_true, so there is "
                "nothing for a combination to be suspiciously close to",
            )

        numeric_cols = [c for c in context.X.columns if context.X[c].dtype.kind in "if"]
        if len(numeric_cols) < 2:
            return _not_applicable(
                self, "fewer than two numeric feature columns — nothing to combine"
            )

        usable = context.X[numeric_cols].notna().all(axis=1) & pd.notna(pd.Series(target))
        min_rows = max(20, cfg.cv_folds * (2 + _MIN_ROWS_PER_FOLD))
        if int(usable.sum()) < min_rows:
            return _not_applicable(
                self,
                f"fewer than {min_rows} fully-populated numeric rows (need enough per "
                f"fold for a {cfg.cv_folds}-fold out-of-fold fit to mean anything)",
            )

        Xu = context.X.loc[usable, numeric_cols].reset_index(drop=True)
        yu = np.asarray(target)[usable.to_numpy()]

        individual_power = {c: _power(Xu[c].to_numpy(dtype=float), yu, binary) for c in numeric_cols}
        candidates = numeric_cols
        if cfg.skip_covered_features:
            candidates = [
                c for c in candidates if not _already_covered(individual_power[c], model_power, cfg)
            ]
        if len(candidates) < 2:
            return _not_applicable(
                self,
                "fewer than two candidate columns remain once columns LeakageCheck "
                "would already flag alone are set aside — nothing novel left to combine",
            )

        if len(candidates) > cfg.max_features:
            ranked = sorted(candidates, key=lambda c: -(individual_power[c] or 0.0))
            logger.warning(
                "%s: %d candidate column(s) exceeds max_features=%d — scoring only the "
                "%d with the highest individual power. Raise max_features, or narrow "
                "the feature set, to widen the search.",
                self.name,
                len(candidates),
                cfg.max_features,
                cfg.max_features,
            )
            candidates = ranked[: cfg.max_features]

        suspects: list[tuple[tuple[str, ...], float]] = []
        n_tested = 0
        for size in cfg.combo_sizes:
            if size < 2 or size > len(candidates):
                continue
            for combo in combinations(candidates, size):
                n_tested += 1
                power = _oof_combo_power(Xu[list(combo)], yu, binary, cfg.cv_folds)
                if power is None:
                    continue
                if power >= cfg.leakage_min_power and power >= cfg.leakage_ratio * model_power:
                    suspects.append((combo, power))

        if not suspects:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "OK",
                    f"no combination of up to {max(cfg.combo_sizes)} feature(s) approaches "
                    f"the model's own predictive power ({model_power:.3f}) across "
                    f"{n_tested} combination(s) tested over {len(candidates)} candidate "
                    "column(s)",
                    self.blocking,
                    metadata={
                        "model_power": round(model_power, 4),
                        "threshold_ratio": cfg.leakage_ratio,
                        "min_power": cfg.leakage_min_power,
                        "n_candidates_scored": len(candidates),
                        "n_combinations_tested": n_tested,
                    },
                )
            ]

        # Smallest, strongest combination first; drop a larger combo once a
        # subset of it has already been reported, since the smaller finding
        # is the more actionable one and the larger is not independent of it.
        suspects.sort(key=lambda pair: (len(pair[0]), -pair[1]))
        reported: list[tuple[tuple[str, ...], float]] = []
        for combo, power in suspects:
            if any(set(seen) <= set(combo) for seen, _ in reported):
                continue
            reported.append((combo, power))

        return [
            CheckResult(
                self.name,
                self.category,
                "COMBINED_LEAKAGE_RISK",
                detail=(
                    f"{' + '.join(combo)} together reach {power:.3f} of the target "
                    f"out-of-fold ({power / model_power:.0%} of the whole model's "
                    f"{model_power:.3f}), while no single one of them alone reaches "
                    f"{cfg.leakage_min_power:.2f} — check whether they are different "
                    "views of the same post-outcome fact (e.g. an actual value and the "
                    "plan or schedule it is compared against to define the label)"
                ),
                blocking=self.blocking,
                metadata={
                    "features": list(combo),
                    "combo_power": round(power, 4),
                    "model_power": round(model_power, 4),
                    "power_ratio": round(power / model_power, 4),
                    "individual_power": {
                        c: (round(individual_power[c], 4) if individual_power[c] is not None else None)
                        for c in combo
                    },
                    "threshold_ratio": cfg.leakage_ratio,
                    "min_power": cfg.leakage_min_power,
                    "cv_folds": cfg.cv_folds,
                    "n_candidates_scored": len(candidates),
                    "n_combinations_tested": n_tested,
                },
            )
            for combo, power in reported
        ]


__all__ = ["CombinedFeatureLeakageCheck", "CombinedLeakageConfig"]
