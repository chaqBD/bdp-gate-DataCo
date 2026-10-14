# Upstream patch

[`0001-add-combined-feature-leakage-check.patch`](0001-add-combined-feature-leakage-check.patch)
adds `CombinedFeatureLeakageCheck` directly into
`bdp_model_gate/structured/validation_checks.py`, as a draft for upstream
review at [vanjy-eng/model-gate](https://github.com/vanjy-eng/model-gate) —
the alternative to installing this repo's plugin package, for a maintainer
who would rather the check live in core. See
[`paper/leaky-benchmarks-governance-gates.md`](../paper/leaky-benchmarks-governance-gates.md)
for the motivation and design rationale.

## What it touches

- `bdp_model_gate/config.py` — four new `ValidationConfig` fields
  (`combined_leakage_combo_sizes`, `_cv_folds`, `_max_features`,
  `_skip_covered_features`); reuses `leakage_ratio`/`leakage_min_power` as
  this check's threshold too.
- `bdp_model_gate/structured/validation_checks.py` — the check itself,
  beside `LeakageCheck`, reusing that module's own private reduction
  helpers (`_binary_target`, `_model_score`, `_power`, `_not_applicable`)
  rather than reimplementing them, since this patch assumes it becomes part
  of the same module.
- `bdp_model_gate/structured/__init__.py` — wired into
  `default_structured_checks()`.
- `tests/test_validation_checks.py` — known-answer tests for the new check,
  plus one pre-existing test (`test_validation_findings_block_and_lead_the_report`)
  updated to expect six validation check names instead of five.
- `README.md` — one new bullet under **Validation**.

## Validated against the upstream test suite

Applied against the `bdp-model-gate` 0.6.0 sdist and run with
`pytest --ignore=tests/test_documentation.py` (that one file needs
`CONTRIBUTING.md`/`CHANGELOG.md`, not shipped in the sdist — unrelated to
this change): the full suite passes except for tests that need fixtures or
files the **sdist itself doesn't ship** (`tests/conftest.py`, `CHANGELOG.md`,
the `web/` docs site) — none of that is specific to this patch, and would
not reproduce against the actual git repository, which is the one a real PR
would be opened against.

## Known gap: two files this patch could not touch

`tests/test_package.py::test_every_default_check_is_documented` requires
every default check's name to appear in `web/docs/reference/checks.md`, and
the project's own contribution guide additionally asks for the check grid in
`web/landing/index.html` to be updated. Neither file ships in the PyPI
sdist this patch was prepared against, so this patch does not include that
edit. **Whoever opens the real PR against the git repository should add
`combined_feature_leakage` to both before submitting** — the one-line
addition this patch already made to this sdist's own `README.md` shows the
wording to reuse.

## Applying it

```bash
git clone https://github.com/vanjy-eng/model-gate
cd model-gate
git apply /path/to/0001-add-combined-feature-leakage-check.patch
```

If context lines have drifted since this was prepared (0.6.0), `git apply
--3way` or a manual rebase of the hunk in `validation_checks.py` should be
straightforward — the new class is additive and self-contained.
