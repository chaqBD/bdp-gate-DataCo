"""bdp-gate-dataco — a combined-feature leakage check for bdp-model-gate.

Registers `CombinedFeatureLeakageCheck` under the `bdp_model_gate.checks`
entry-point group, so installing this package alongside `bdp-model-gate`
is enough for `default_structured_checks()` to pick it up with no other
wiring. See `bdp_gate_dataco.checks` for the check itself, and
`paper/leaky-benchmarks-governance-gates.md` in this repository for the
DataCo case study that motivated it.
"""

from .checks import CombinedFeatureLeakageCheck, CombinedLeakageConfig

__version__ = "0.1.0"

__all__ = ["CombinedFeatureLeakageCheck", "CombinedLeakageConfig", "__version__"]
