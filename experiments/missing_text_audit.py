"""Measure what each branch does when the clinical note is missing.

A text-only model has no input when the note is absent, while a multimodal model
still has the radiograph. This trains both branches under the same federated
loop and evaluates each twice, once with notes present and once with every note
replaced by an empty document.

An empty note is ``[CLS] [SEP]`` with attention on those two positions only,
which is what the tokenizer produces for an empty string. It is not a zeroed
tensor, so the encoder runs normally and the score is a real measurement rather
than a masked-out NaN.

    OM_COVID_ROOT=... ./.venv/bin/python repo/experiments/missing_text_audit.py \
        --base repo/source/MedFederate_Colab_Complete.py \
        --cache data_cache_v3_real.pkl --out missing_text_v3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import omnimed_experiments as ox  # noqa: E402


class BlankNotes(Dataset):
    """The validation set with every note replaced by an empty document."""

    def __init__(self, base: Dataset, cls_id: int, sep_id: int, pad_id: int):
        self.base = base
        self.cls_id, self.sep_id, self.pad_id = cls_id, sep_id, pad_id

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        item = dict(self.base[idx])
        ids = item["input_ids"].clone()
        mask = item["attention_mask"].clone()
        ids[:] = self.pad_id
        ids[0] = self.cls_id
        ids[1] = self.sep_id
        mask[:] = 0
        mask[0] = 1
        mask[1] = 1
        item["input_ids"] = ids
        item["attention_mask"] = mask
        return item


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tier = dict(ox.TIERS["standard"])
    ox._patch_transformers()
    ox._patch_datasets()
    mf = ox.load_base(args.base)
    ox._patch_image_loader(mf)

    cfg = mf.Config(
        batch_size=tier["batch_size"],
        max_samples_per_class=tier["samples_per_class"],
        fed_rounds=args.rounds,
        local_epochs=tier["local_epochs"],
        epochs=tier["central_epochs"],
    )
    data = ox.build_data(mf, cfg, Path(args.cache))
    tokenizer = mf.get_text_tokenizer(tier["text_model"], cfg.max_seq_length)
    train_ds, val_ds, _, val_loader = ox.make_mm_loaders(mf, data, cfg, tokenizer)

    blank = BlankNotes(val_ds, tokenizer.cls_token_id, tokenizer.sep_token_id,
                       tokenizer.pad_token_id)
    blank_loader = DataLoader(blank, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mm_kwargs = dict(text_model_name=tier["text_model"],
                     vision_model_name=tier["vision_model"],
                     num_labels=5, fusion_type="concat")

    branches = {
        "text_only": (mf.LightweightTextClassifier,
                      dict(model_name=tier["text_model"], num_labels=5), "text"),
        "multimodal": (mf.MultiModalClassifier, mm_kwargs, "multimodal"),
    }

    results = {}
    for name, (cls, kw, mtype) in branches.items():
        print(f"\n=== {name} ===")
        ox.set_seed(args.seed)
        res = ox.federated_train_ex(
            mf, cls, kw, train_ds, val_loader, cfg, device, mtype,
            alpha=1.0, num_clients=5, rounds=args.rounds,
            local_epochs=tier["local_epochs"], seed=args.seed,
            log_prefix=f"[{name}] ", extra_evals={"notes_missing": blank_loader})
        results[name] = {
            "notes_present_f1": res["f1"],
            "notes_missing_f1": res["extra"]["notes_missing"]["f1"],
            "notes_present_accuracy": res["accuracy"],
            "notes_missing_accuracy": res["extra"]["notes_missing"]["accuracy"],
            "notes_missing_diversity": res["extra"]["notes_missing"]["diversity"],
        }
        r = results[name]
        print(f"  {name}: notes present {r['notes_present_f1']:.3f} -> "
              f"notes missing {r['notes_missing_f1']:.3f}")

    payload = {
        "task": "missing_text_audit",
        "protocol": {"alpha": 1.0, "clients": 5, "rounds": args.rounds,
                     "seed": args.seed,
                     "missing_note": "[CLS] [SEP] with attention on both"},
        "result": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
