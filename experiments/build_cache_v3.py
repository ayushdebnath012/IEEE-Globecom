"""Build the real-only (v3) corpus cache and report its provenance.

Runs the same data path the suite uses, but stops after the cache is written so
the real-image sourcing can be verified before committing GPU hours.

    OM_COVID_ROOT=~/omnimed/covid19_radiography \
    ./.venv/bin/python repo/experiments/build_cache_v3.py \
        --base repo/source/MedFederate_Colab_Complete.py \
        --tier standard --cache ~/omnimed/data_cache_v3_real.pkl
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import omnimed_experiments as ox  # noqa: E402
from real_covid_loader import PROTOCOL, install_real_only_sources  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tier", default="standard", choices=list(ox.TIERS))
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()

    covid_root = os.environ.get("OM_COVID_ROOT")
    if not covid_root:
        raise SystemExit("OM_COVID_ROOT must point at the Kaggle download")

    t = dict(ox.TIERS[args.tier])
    ox._patch_transformers()
    ox._patch_datasets()
    mf = ox.load_base(args.base)
    ox._patch_image_loader(mf)
    install_real_only_sources(mf, covid_root, seed=mf.Config().seed)

    # Same construction as omnimed_experiments.main, so the cache this writes is
    # the cache the suite will load.
    cfg = mf.Config(
        batch_size=t["batch_size"],
        max_samples_per_class=t["samples_per_class"],
        fed_rounds=t["fed_rounds"],
        local_epochs=t["local_epochs"],
        epochs=t["central_epochs"],
    )

    cache = Path(args.cache).expanduser()
    if cache.exists():
        raise SystemExit(f"{cache} already exists; remove it to rebuild")

    data = ox.build_data(mf, cfg, cache)

    train = Counter(l[0] for l in data["train_ilbls"])
    val = Counter(l[0] for l in data["val_ilbls"])
    total = Counter()
    total.update(train)
    total.update(val)

    print("\nclass totals (train + validation)")
    for idx in range(len(mf.CONDITION_LABELS)):
        print(f"  {mf.IDX_TO_LABEL[idx]:<18} {total.get(idx, 0)}")
    print(f"  train={sum(train.values())}  validation={sum(val.values())}")

    h = hashlib.sha256()
    with open(cache, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    print(f"\nprotocol            {PROTOCOL}")
    print(f"data_cache_sha256   {h.hexdigest()}")
    print(f"cache               {cache} ({cache.stat().st_size} bytes)")
    print("image provenance    3000 public radiographs, 0 synthetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
