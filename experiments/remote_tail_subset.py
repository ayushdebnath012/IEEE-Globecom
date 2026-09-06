"""Run a disjoint experiment chunk through the canonical audited runner.

Environment variables select the output and experiment set. For E5 only,
OM_FUSION_TYPES restricts the canonical registry without changing training code.
"""

from __future__ import annotations

import os
import sys

ROOT = "/home/trishita/omnimed"
sys.path.insert(0, f"{ROOT}/repo/experiments")

import omnimed_experiments as runner  # noqa: E402


fusion_types = os.environ.get("OM_FUSION_TYPES", "").strip()
if fusion_types:
    runner.FUSION_TYPES = [v.strip() for v in fusion_types.split(",") if v.strip()]

runner.main(
    base_py=f"{ROOT}/repo/source/MedFederate_Colab_Complete.py",
    tier="standard",
    out=os.environ["OM_RESULTS_OUT"],
    cache=f"{ROOT}/data_cache_standard_controlled.pkl",
    only=[v.strip() for v in os.environ["OM_ONLY"].split(",") if v.strip()],
)
