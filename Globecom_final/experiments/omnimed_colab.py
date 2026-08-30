"""Thin Google Colab launcher for the audited OmniMed-FL experiment suite.

Upload this file together with ``omnimed_experiments.py``,
``omnimed_make_tables.py``, and ``MedFederate_Colab_Complete.py``. Keeping the
experiment logic in one canonical module prevents notebook-specific copies from
silently reverting to the invalid pooled-data warm-start protocol.
"""

from __future__ import annotations

import importlib
from pathlib import Path


CONTENT = Path("/content")


def _modules():
    import omnimed_experiments as experiments
    import omnimed_make_tables as tables
    return importlib.reload(experiments), importlib.reload(tables)


def run(tier: str = "smoke", only=None, out: str | None = None,
        alphas=None, cache: str | None = None) -> Path:
    """Run or resume a chunk using the canonical audited implementation."""
    experiments, _ = _modules()
    out_path = Path(out) if out else CONTENT / (
        "results_smoke.json" if tier == "smoke" else "results_v2.json")
    experiments.main(
        base_py=str(CONTENT / "MedFederate_Colab_Complete.py"),
        tier=tier,
        out=str(out_path),
        only=only,
        alphas=alphas,
        cache=cache,
    )
    return out_path


def build_tables(results: str | None = None,
                 outdir: str = "/content/paper_assets") -> Path:
    """Generate tables/figures from a completed result store."""
    _, tables = _modules()
    result_path = Path(results) if results else CONTENT / "results_v2.json"
    tables.main(result_path, Path(outdir))
    return Path(outdir)


if __name__ == "__main__":
    raise SystemExit(
        "Import this launcher in Colab, then call run('smoke') before a standard chunk."
    )
