"""Branch comparison with per-class scores, at any skew, with and without notes.

Answers three questions the aggregate macro-F1 at alpha=1 cannot:

* how each branch degrades when the clinical note is absent,
* whether the text branch's advantage survives severe label skew,
* which classes each branch actually wins, since a macro average over five
  classes hides the ones where the radiograph carries the finding.

    ./.venv/bin/python repo/experiments/branch_audit_ex.py \
        --base repo/source/MedFederate_Colab_Complete.py \
        --cache data_cache_controlled_v2.pkl --alpha 0.1 --out branch_a01.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import omnimed_experiments as ox  # noqa: E402

CLASSES = ["Normal", "Pneumonia", "COVID-19", "Pleural Effusion", "Cardiomegaly"]


class BlankNotes(Dataset):
    """Validation set with every note replaced by an empty document."""

    def __init__(self, base, cls_id, sep_id, pad_id):
        self.base, self.cls_id, self.sep_id, self.pad_id = base, cls_id, sep_id, pad_id

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = dict(self.base[idx])
        ids, mask = item["input_ids"].clone(), item["attention_mask"].clone()
        ids[:] = self.pad_id
        ids[0], ids[1] = self.cls_id, self.sep_id
        mask[:] = 0
        mask[0], mask[1] = 1, 1
        item["input_ids"], item["attention_mask"] = ids, mask
        return item


def per_class_f1(model, loader, device, model_type):
    """Macro and per-class F1 for one model on one loader."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            kw = {}
            if model_type in ("text", "multimodal"):
                kw["input_ids"] = batch["input_ids"].to(device)
                kw["attention_mask"] = batch["attention_mask"].to(device)
            if model_type in ("image", "multimodal"):
                kw["pixel_values"] = batch["pixel_values"].to(device)
            out = model(**kw)
            preds.append(out["logits"].argmax(dim=-1).cpu().numpy())
            labels.append(batch["labels"].argmax(dim=-1).cpu().numpy())
    p, y = np.concatenate(preds), np.concatenate(labels)
    scores = []
    for c in range(len(CLASSES)):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"macro_f1": float(np.mean(scores)),
            "per_class_f1": {CLASSES[i]: round(float(v), 4) for i, v in enumerate(scores)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fusion", default="concat",
                    help="fusion operator for the multimodal branch; the "
                         "pooled screen selects it, not the branch audit")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of text_only,image_only,multimodal")
    args = ap.parse_args()

    tier = dict(ox.TIERS["standard"])
    ox._patch_transformers()
    ox._patch_datasets()
    mf = ox.load_base(args.base)
    ox._patch_image_loader(mf)

    cfg = mf.Config(batch_size=tier["batch_size"],
                    max_samples_per_class=tier["samples_per_class"],
                    fed_rounds=tier["fed_rounds"],
                    local_epochs=tier["local_epochs"],
                    epochs=tier["central_epochs"])
    data = ox.build_data(mf, cfg, Path(args.cache))
    tok = mf.get_text_tokenizer(tier["text_model"], cfg.max_seq_length)
    train_ds, val_ds, _, val_loader = ox.make_mm_loaders(mf, data, cfg, tok)
    blank_loader = DataLoader(
        BlankNotes(val_ds, tok.cls_token_id, tok.sep_token_id, tok.pad_token_id),
        batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mm_kwargs = dict(text_model_name=tier["text_model"],
                     vision_model_name=tier["vision_model"],
                     num_labels=5, fusion_type=args.fusion)
    branches = {
        "text_only": (mf.LightweightTextClassifier,
                      dict(model_name=tier["text_model"], num_labels=5), "text"),
        "image_only": (mf.LightweightVisionClassifier,
                       dict(model_name=tier["vision_model"], num_labels=5), "image"),
        "multimodal": (mf.MultiModalClassifier, mm_kwargs, "multimodal"),
    }

    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        branches = {k: v for k, v in branches.items() if k in keep}

    results = {}
    for name, (cls, kw, mtype) in branches.items():
        print(f"\n=== {name}  alpha={args.alpha} ===")

        def hook(model, dev, _mtype=mtype):
            out = {"notes_present": per_class_f1(model, val_loader, dev, _mtype)}
            if _mtype != "image":
                out["notes_missing"] = per_class_f1(model, blank_loader, dev, _mtype)
            return out

        ox.set_seed(args.seed)
        res = ox.federated_train_ex(
            mf, cls, kw, train_ds, val_loader, cfg, device, mtype,
            alpha=args.alpha, num_clients=5, rounds=tier["fed_rounds"],
            local_epochs=tier["local_epochs"], seed=args.seed,
            log_prefix=f"[{name}] ", extra_eval_fn=hook)
        results[name] = res["extra"]
        present = res["extra"]["notes_present"]["macro_f1"]
        missing = res["extra"].get("notes_missing", {}).get("macro_f1")
        print(f"  {name}: notes present {present:.3f}"
              + (f" -> notes missing {missing:.3f}" if missing is not None else ""))

    Path(args.out).write_text(json.dumps(
        {"task": "branch_audit_ex",
         "protocol": {"alpha": args.alpha, "clients": 5,
                      "rounds": tier["fed_rounds"], "seed": args.seed,
                      "fusion": args.fusion},
         "result": results}, indent=1), encoding="utf8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
