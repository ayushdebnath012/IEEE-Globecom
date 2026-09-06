"""Thin Kaggle launcher for the audited OmniMed-FL experiment suite.

Place the canonical experiment files in ``/kaggle/working``. This launcher does
not duplicate training logic; it delegates to ``omnimed_experiments.py`` so the
Kaggle path cannot retain the rejected pooled-data initialization or FedBN code.
"""

from __future__ import annotations

import importlib
from pathlib import Path


WORK = Path("/kaggle/working")


def _modules():
    import omnimed_experiments as experiments
    import omnimed_make_tables as tables
    return importlib.reload(experiments), importlib.reload(tables)


def run(tier: str = "smoke", only=None, out: str | None = None,
        alphas=None, cache: str | None = None) -> Path:
    """Run or resume a chunk using the canonical audited implementation."""
    experiments, _ = _modules()
    out_path = Path(out) if out else WORK / (
        "results_smoke.json" if tier == "smoke" else "results_v2.json")
    experiments.main(
        base_py=str(WORK / "MedFederate_Colab_Complete.py"),
        tier=tier,
        out=str(out_path),
        only=only,
        alphas=alphas,
        cache=cache,
    )
    return out_path


def build_tables(results: str | None = None,
                 outdir: str = "/kaggle/working/paper_assets") -> Path:
    """Generate tables/figures from a completed result store."""
    _, tables = _modules()
    result_path = Path(results) if results else WORK / "results_v2.json"
    tables.main(result_path, Path(outdir))
    return Path(outdir)


if __name__ == "__main__":
    raise SystemExit(
        "Import this launcher in a Kaggle notebook, then call run('smoke') first."
    )
