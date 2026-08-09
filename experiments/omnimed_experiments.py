"""
OmniMed-FL — reviewer-requested experiment suite.

Runs the experiments the GLOBECOM reviewers asked for and that the submitted
results file does not contain:

  E1  Dirichlet alpha sweep (R3.1)          -> generalization across heterogeneity
  E2  Client-count sweep K (R3.4)           -> scalability
  E3  Anti-collapse component ablation (R3.2)
  E4  Warm-start ablation (R3.2)
  E5  Fusion-strategy seed repeats (R3.1)   -> variance, so rankings are defensible
  E6  Measured runtime / memory / comm (R1.2, R3.4)
  E7  RAG evaluation on a held-out query set (R3.2)

Everything reuses the model, data and FedAvg code in MedFederate_Colab_Complete.py.
Nothing here re-implements the method; the only additions are ablation switches
and instrumentation.

Design notes
------------
* Resumable. Every (experiment, config, seed) result is keyed and written to the
  results JSON immediately. Re-running skips completed keys, so a Colab
  disconnect costs you at most one run.
* Tiered. TIER='smoke' validates the whole pipeline in ~20 min before you commit
  GPU hours. TIER='standard' and 'full' are the real runs.
* Every number it emits is measured. Nothing is estimated or filled in.

Usage (Colab):
    !python omnimed_experiments.py --base /content/MedFederate_Colab_Complete.py \
        --tier standard --out /content/drive/MyDrive/omnimed/results_v2.json

Usage (import):
    import omnimed_experiments as ox
    ox.main(base_py=..., tier='smoke', out=...)
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------

TIERS = {
    # Validates the pipeline end to end. Results are NOT publishable.
    "smoke": dict(
        samples_per_class=60, batch_size=8, fed_rounds=2, local_epochs=1,
        seeds=[0], alphas=[0.1, 1.0], client_counts=[3, 5],
        fusion_seeds=[0], text_model="prajjwal1/bert-tiny",
        vision_model="facebook/deit-tiny-patch16-224", central_epochs=2,
    ),
    # Enough for the alpha sweep and ablations to be reportable.
    "standard": dict(
        samples_per_class=600, batch_size=16, fed_rounds=8, local_epochs=3,
        seeds=[0, 1], alphas=[0.1, 0.3, 0.5, 1.0, 5.0], client_counts=[3, 5, 10, 20],
        fusion_seeds=[0, 1], text_model="distilbert-base-uncased",
        vision_model="google/vit-base-patch16-224", central_epochs=12,
    ),
    # Adds a third seed everywhere. Expect multiple Colab sessions.
    "full": dict(
        samples_per_class=600, batch_size=16, fed_rounds=8, local_epochs=3,
        seeds=[0, 1, 2], alphas=[0.05, 0.1, 0.3, 0.5, 1.0, 5.0, 100.0],
        client_counts=[3, 5, 10, 20, 50], fusion_seeds=[0, 1, 2],
        text_model="distilbert-base-uncased",
        vision_model="google/vit-base-patch16-224", central_epochs=20,
    ),
}

FUSION_TYPES = ["concat", "attention", "gated", "clip",
                "flamingo", "blip2", "coca", "unified_io"]

# ---------------------------------------------------------------------------
# Base module loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pretrained-encoder fix (do not remove)
# ---------------------------------------------------------------------------
# LightweightTextClassifier loads its encoder with AutoModel inside a bare
# try/except. Current transformers rejects prajjwal1/bert-* (no "model_type"
# in config.json), the except fires, and the encoder silently becomes a
# RANDOMLY INITIALISED embedding + 2-layer transformer. Training then reports
# numbers containing no pretrained language model at all.


def _patch_transformers():
    """Route configs/models that Auto* rejects to the BERT classes explicitly."""
    try:
        from transformers import AutoConfig, AutoModel, BertConfig, BertModel
    except ImportError:
        return False
    if getattr(AutoConfig, "_omnimed_patched", False):
        return True
    _oc, _om = AutoConfig.from_pretrained, AutoModel.from_pretrained

    def cfg(name, *a, **k):
        try:
            return _oc(name, *a, **k)
        except Exception:
            return BertConfig.from_pretrained(name, *a, **k)

    def mdl(name, *a, **k):
        try:
            return _om(name, *a, **k)
        except Exception:
            return BertModel.from_pretrained(name, *a, **k)

    AutoConfig.from_pretrained = cfg
    AutoModel.from_pretrained = mdl
    AutoConfig._omnimed_patched = True
    print("  [patch] transformers Auto* fallback installed")
    return True


def verify_encoders(mf, names):
    """Fail loudly if an encoder is not loading pretrained weights."""
    print("  verifying text encoders load pretrained weights:")
    bad = []
    for n in names:
        try:
            m = mf.LightweightTextClassifier(n, 5)
            ok = m.transformer is not None
            print(f"    {'OK      ' if ok else 'FALLBACK'}  {n}")
            if not ok:
                bad.append(n)
            del m
        except Exception as e:
            print(f"    ERROR     {n}  ({str(e)[:60]})")
            bad.append(n)
    return not bad

def _patch_datasets():
    """Supply the config names load_hf_medical_images() never passes.

    load_dataset("keremberke/chest-xray-classification", split="train") raises
    "Config name is missing" because that dataset defines configs full/mini.
    The caller wraps the load in a bare except, so the error is swallowed and
    every image silently becomes a synthetic surrogate. Same for the NIH set.
    Patched here rather than in the base module so the original stays intact.
    """
    try:
        import datasets
    except ImportError:
        return False
    if getattr(datasets, "_omnimed_patched", False):
        return True

    CONFIGS = {
        "keremberke/chest-xray-classification": "full",
        "alkzar90/NIH-Chest-X-ray-dataset": "image-classification",
    }
    _orig = datasets.load_dataset

    def load_dataset(path, name=None, *a, **k):
        if name is None and path in CONFIGS:
            name = CONFIGS[path]
        k.setdefault("trust_remote_code", True)
        return _orig(path, name, *a, **k)

    datasets.load_dataset = load_dataset
    datasets._omnimed_patched = True
    print("  [patch] dataset config names installed")
    return True


def verify_images(mf, n=5):
    """Report whether real radiographs loaded, or synthetic surrogates."""
    imgs, labels = mf.load_medical_image_data(n_per_class=n, img_size=224)
    print(f"  image check: {len(imgs)} images loaded")
    return len(imgs) > 0


def _patch_image_loader(mf):
    """Replace load_hf_medical_images with a version that resolves ClassLabels.

    Third defect in the same call path. condition_mapping is keyed on label
    NAMES ('NORMAL', 'PNEUMONIA'), but the loader compares against
    str(item[label_col]) and these datasets store labels as ClassLabel
    integers. So str(0) == '0' never matches 'NORMAL', every row is skipped,
    the function returns 0 images, and the caller quietly tops the class up
    with synthetic surrogates. Net effect: an all-synthetic image set that
    reports no error anywhere.
    """
    def load_hf_medical_images(dataset_name, condition_mapping,
                               n_per_class=200, img_size=224):
        from collections import defaultdict
        try:
            from datasets import load_dataset
            from PIL import Image as PILImage
            import torchvision.transforms as T

            transform = T.Compose([
                T.Resize((img_size, img_size)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            print(f"  Loading {dataset_name} ...")
            ds = load_dataset(dataset_name, split="train")

            label_col = next((c for c in ["label", "labels", "condition", "finding",
                                          "diagnosis", "class"]
                              if c in ds.column_names), None)
            img_col = next((c for c in ["image", "img", "pixel_values", "photo"]
                            if c in ds.column_names), None)
            if label_col is None or img_col is None:
                raise ValueError("Missing label or image column")

            # ClassLabel -> names, so integer labels can be resolved
            names = None
            feat = ds.features.get(label_col)
            if hasattr(feat, "names"):
                names = list(feat.names)
            elif hasattr(feat, "feature") and hasattr(feat.feature, "names"):
                names = list(feat.feature.names)

            lut = {str(k).lower(): v for k, v in condition_mapping.items()}

            def resolve(raw):
                vals = raw if isinstance(raw, (list, tuple)) else [raw]
                for v in vals:
                    if names is not None and isinstance(v, int) and 0 <= v < len(names):
                        v = names[v]
                    tgt = lut.get(str(v).lower())
                    if tgt:
                        return tgt
                return None

            images, labels = [], []
            counts = defaultdict(int)
            for item in ds:
                cond = resolve(item[label_col])
                if cond is None:
                    continue
                ci = mf.LABEL_TO_IDX.get(cond, -1)
                if ci < 0 or counts[ci] >= n_per_class:
                    continue
                try:
                    img = item[img_col]
                    img = (img.convert("RGB") if isinstance(img, PILImage.Image)
                           else PILImage.fromarray(img).convert("RGB"))
                    images.append(transform(img).half())
                    labels.append([ci])
                    counts[ci] += 1
                except Exception:
                    continue
                if all(counts[i] >= n_per_class
                       for i in range(len(mf.CONDITION_LABELS))):
                    break

            print(f"  Loaded {len(images)} REAL images: {dict(counts)}")
            return images, labels
        except Exception as e:
            # Return EMPTY, not synthetic. The original substituted a full
            # synthetic set here, which both hid the failure and pushed the
            # caller's running total past its break threshold -- so a dead
            # source silently prevented every later source from being tried.
            # load_medical_image_data() already tops up per class at the end,
            # which is where synthetic filling belongs.
            print(f"  source unavailable ({str(e)[:80]}) -- skipping, no synthetic substitution")
            return [], []

    mf.load_hf_medical_images = load_hf_medical_images
    print("  [patch] image loader (ClassLabel resolution) installed")
    return True


def load_base(base_py: str):
    """Import MedFederate_Colab_Complete.py without firing its __main__ block."""
    base_py = str(base_py)
    if not os.path.exists(base_py):
        raise FileNotFoundError(
            f"Base module not found at {base_py}. Upload MedFederate_Colab_Complete.py "
            f"or pass --base with the right path."
        )
    spec = importlib.util.spec_from_file_location("medfed_base", base_py)
    mod = importlib.util.module_from_spec(spec)
    # Name is not __main__, so the auto-run guard at the bottom stays shut.
    sys.modules["medfed_base"] = mod
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data — built once, cached, reused by every experiment
# ---------------------------------------------------------------------------


def build_data(mf, cfg, cache_path: Path):
    """Reproduce the data pipeline of run_training() and cache the result.

    Kept deliberately identical to the original so the alpha sweep differs from
    the submitted run in alpha alone.
    """
    if cache_path.exists():
        print(f"  [data] loading cache {cache_path}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    print("  [data] building (first run only; this is the slow part)")
    images, i_labels = mf.load_medical_image_data(
        n_per_class=cfg.max_samples_per_class, img_size=cfg.image_size)
    images, i_labels = mf.balance_image_dataset(
        images, i_labels, cfg.max_samples_per_class)

    pool = mf.load_hf_medical_text(n_per_class=cfg.max_samples_per_class * 2)
    ptrs = {i: 0 for i in range(5)}
    texts, t_labels = [], []
    for lbl in i_labels:
        ci = lbl[0] if isinstance(lbl, (list, tuple)) else int(lbl)
        p = pool.get(ci, [])
        if ptrs[ci] < len(p):
            texts.append(p[ptrs[ci]])
            ptrs[ci] += 1
        else:
            texts.append(mf.generate_synthetic_text_data(1, target_labels=[[ci]])["text"].iloc[0])
        t_labels.append([ci])

    n_train = int(len(texts) * cfg.train_split)
    data = dict(
        train_texts=texts[:n_train], val_texts=texts[n_train:],
        train_tlbls=t_labels[:n_train], val_tlbls=t_labels[n_train:],
        train_imgs=images[:n_train], val_imgs=images[n_train:],
        train_ilbls=i_labels[:n_train], val_ilbls=i_labels[n_train:],
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(data, fh, protocol=4)
    print(f"  [data] cached -> {cache_path}  (train={n_train}, val={len(texts)-n_train})")
    return data


def make_mm_loaders(mf, data, cfg, tokenizer, balanced: bool = True):
    train_ds = mf.MultiModalDataset(data["train_texts"], data["train_tlbls"],
                                    data["train_imgs"], tokenizer=tokenizer,
                                    max_length=cfg.max_seq_length)
    val_ds = mf.MultiModalDataset(data["val_texts"], data["val_tlbls"],
                                  data["val_imgs"], tokenizer=tokenizer,
                                  max_length=cfg.max_seq_length)
    if balanced:
        train_loader = mf.create_balanced_dataloader(
            train_ds, data["train_tlbls"], cfg.batch_size, 5)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_ds, val_ds, train_loader, val_loader


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


def trainable_bytes(model: nn.Module, dtype_bytes: int = 4) -> int:
    """Bytes a client would upload under FedAvg (float params only)."""
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad and p.is_floating_point()) * dtype_bytes


def param_count(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


class GPUMem:
    """Peak allocated GPU memory across a block, in MiB."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_mib = torch.cuda.max_memory_allocated() / 2 ** 20
        else:
            self.peak_mib = float("nan")
        self.seconds = time.perf_counter() - self.t0
        return False


# ---------------------------------------------------------------------------
# Federated trainer with ablation switches + instrumentation
# ---------------------------------------------------------------------------


def federated_train_ex(mf, model_class, model_kwargs, train_dataset, val_loader,
                       cfg, device, model_type="multimodal",
                       *, alpha=1.0, num_clients=5, rounds=8, local_epochs=3,
                       use_balanced_sampler=True, diversity_weight=1.0,
                       warm_start_state=None, seed=0, log_prefix=""):
    """FedAvg with the anti-collapse components individually switchable.

    diversity_weight=0.0 disables the entropy diversity term.
    use_balanced_sampler=False falls back to a plain shuffled DataLoader.
    warm_start_state=None means clients start from a fresh (cold) init.
    """
    set_seed(seed)
    global_model = model_class(**model_kwargs).to(device)
    if warm_start_state is not None:
        global_model.load_state_dict(warm_start_state, strict=False)

    upload_bytes = trainable_bytes(global_model, dtype_bytes=4)
    splits = mf.split_data_non_iid(train_dataset, num_clients, alpha)
    sizes = [len(s) for s in splits]
    total = max(sum(sizes), 1)

    # Per-client class histogram: the actual realized heterogeneity, not the
    # nominal alpha. Worth reporting -- alpha is a prior, this is the sample.
    client_hist = []
    for s in splits:
        h = [0] * 5
        for i in s:
            h[int(train_dataset[i]["labels"].argmax().item())] += 1
        client_hist.append(h)

    hist = {"round_f1": [], "round_acc": [], "round_div": [],
            "round_seconds": [], "round_peak_mib": []}

    for t in range(rounds):
        with GPUMem() as gm:
            agg = {}
            for k in range(num_clients):
                w = sizes[k] / total
                if len(splits[k]) < 4:
                    with torch.no_grad():
                        for key, val in global_model.state_dict().items():
                            if not torch.is_floating_point(val):
                                continue
                            agg.setdefault(key, torch.zeros_like(
                                val, dtype=torch.float32, device="cpu"))
                            agg[key].add_(w * val.detach().cpu().float())
                    continue

                local_ds = Subset(train_dataset, splits[k])
                local_lbls = [train_dataset[i]["labels"].argmax().item() for i in splits[k]]
                if use_balanced_sampler:
                    loader = mf.create_balanced_dataloader(
                        local_ds, local_lbls, cfg.batch_size, 5)
                else:
                    loader = DataLoader(local_ds, batch_size=cfg.batch_size, shuffle=True)

                local_model = copy.deepcopy(global_model).to(device)
                opt = torch.optim.AdamW(local_model.parameters(),
                                        lr=cfg.learning_rate,
                                        weight_decay=cfg.weight_decay)
                crit = mf.CombinedLoss(num_classes=5, diversity_weight=diversity_weight)
                local_model.train()
                for _ in range(local_epochs):
                    for batch in loader:
                        opt.zero_grad()
                        try:
                            out = _forward(local_model, batch, device, model_type)
                            loss = crit(out["logits"], batch["labels"].to(device))
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                            opt.step()
                        except RuntimeError as e:
                            # OOM or shape mismatch: surface it rather than swallow
                            print(f"    [warn] client {k}: {e}")
                            break

                with torch.no_grad():
                    for key, val in local_model.state_dict().items():
                        if not torch.is_floating_point(val):
                            continue
                        agg.setdefault(key, torch.zeros_like(
                            val, dtype=torch.float32, device="cpu"))
                        agg[key].add_(w * val.detach().cpu().float())

                del local_model, opt, crit, loader, local_ds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            with torch.no_grad():
                for key, val in global_model.state_dict().items():
                    if key in agg:
                        val.copy_(agg[key].to(dtype=val.dtype, device=val.device))
            del agg
            gc.collect()

        m = mf.evaluate(global_model, val_loader, device, model_type)
        hist["round_f1"].append(float(m["f1"]))
        hist["round_acc"].append(float(m["accuracy"]))
        hist["round_div"].append(float(m["diversity"]))
        hist["round_seconds"].append(gm.seconds)
        hist["round_peak_mib"].append(gm.peak_mib)
        print(f"    {log_prefix}round {t+1}/{rounds}  F1={m['f1']:.4f} "
              f"acc={m['accuracy']:.4f} div={m['diversity']:.2f} "
              f"{gm.seconds:.1f}s {gm.peak_mib:.0f}MiB")

    final = mf.evaluate(global_model, val_loader, device, model_type)
    out = {
        "f1": float(final["f1"]),
        "accuracy": float(final["accuracy"]),
        "diversity": float(final["diversity"]),
        "history": hist,
        "client_sizes": sizes,
        "client_class_hist": client_hist,
        "upload_bytes_per_client_per_round": upload_bytes,
        "total_comm_bytes": 2 * num_clients * rounds * upload_bytes,
        "trainable_params": param_count(global_model, True),
        "total_params": param_count(global_model, False),
        "wall_seconds": float(np.sum(hist["round_seconds"])),
        "peak_mib": float(np.max(hist["round_peak_mib"])) if hist["round_peak_mib"] else float("nan"),
    }
    del global_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return out


def _forward(model, batch, device, model_type):
    if model_type == "text":
        return model(input_ids=batch["input_ids"].to(device),
                     attention_mask=batch["attention_mask"].to(device))
    if model_type == "image":
        return model(pixel_values=batch["pixel_values"].to(device))
    return model(input_ids=batch["input_ids"].to(device),
                 attention_mask=batch["attention_mask"].to(device),
                 pixel_values=batch["pixel_values"].to(device))


def centralized_train(mf, model, train_loader, val_loader, cfg, device,
                      model_type="multimodal", epochs=12, diversity_weight=1.0,
                      use_early_abort=True, seed=0, log_prefix=""):
    """Centralized training with the LR-boost / abort component switchable."""
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    crit = mf.CombinedLoss(num_classes=5, diversity_weight=diversity_weight)
    hist = {"val_f1": [], "val_acc": [], "diversity": [], "epoch_seconds": []}
    best = {"f1": -1.0}
    collapse_c = 0

    for ep in range(epochs):
        with GPUMem() as gm:
            model.train()
            for batch in train_loader:
                opt.zero_grad()
                try:
                    out = _forward(model, batch, device, model_type)
                    loss = crit(out["logits"], batch["labels"].to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                except RuntimeError as e:
                    print(f"    [warn] {e}")
                    break
        m = mf.evaluate(model, val_loader, device, model_type)
        hist["val_f1"].append(float(m["f1"]))
        hist["val_acc"].append(float(m["accuracy"]))
        hist["diversity"].append(float(m["diversity"]))
        hist["epoch_seconds"].append(gm.seconds)

        if use_early_abort:
            # Mirrors train_model(): boost LR after repeated collapse, abort at 3.
            if m["diversity"] <= 0.4:
                collapse_c += 1
                if collapse_c == 2:
                    for pg in opt.param_groups:
                        pg["lr"] = min(cfg.learning_rate * 5.0, 1e-3)
                    print(f"    {log_prefix}collapse -> LR boost")
                if collapse_c >= 3:
                    print(f"    {log_prefix}collapse -> abort at epoch {ep+1}")
                    break
            else:
                collapse_c = 0

        if m["f1"] > best["f1"]:
            best = {"f1": float(m["f1"]), "accuracy": float(m["accuracy"]),
                    "diversity": float(m["diversity"]), "epoch": ep + 1}
        print(f"    {log_prefix}ep {ep+1}/{epochs}  F1={m['f1']:.4f} "
              f"div={m['diversity']:.2f}  {gm.seconds:.1f}s")

    state = mf.clone_state_dict_to_cpu(model)
    result = {**best, "history": hist,
              "wall_seconds": float(np.sum(hist["epoch_seconds"])),
              "min_diversity": float(np.min(hist["diversity"])) if hist["diversity"] else 0.0,
              "trainable_params": param_count(model, True)}
    del model, opt, crit
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result, state


# ---------------------------------------------------------------------------
# Resumable result store
# ---------------------------------------------------------------------------


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def has(self, exp: str, key: str) -> bool:
        return key in self.data.get(exp, {})

    def put(self, exp: str, key: str, value):
        self.data.setdefault(exp, {})[key] = value
        self.flush()

    def flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=float))
        tmp.replace(self.path)
        print(f"  [store] {self.path}")


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def _norm_key(k: str) -> bool:
    """True for normalization-layer parameters (kept local by FedBN)."""
    kl = k.lower()
    return any(t in kl for t in ("layernorm", "batchnorm", ".bn", "norm.weight",
                                 "norm.bias", "running_mean", "running_var"))


def federated_train_algo(mf, model_class, model_kwargs, train_dataset, val_loader,
                         cfg, device, model_type="multimodal", *, algorithm="fedavg",
                         alpha=1.0, num_clients=5, rounds=8, local_epochs=3,
                         mu=0.01, warm_start_state=None, seed=0, log_prefix=""):
    """FedAvg / FedProx / SCAFFOLD / FedBN on one backbone, one dataset, one config.

    This is the matched-setting comparison Reviewer 3 asked for: the aggregation
    rule is the only thing that changes between arms. Published numbers from other
    corpora cannot answer that question; this can.

      fedavg   -- McMahan et al., weighted parameter mean
      fedprox  -- Li et al., + (mu/2)||w - w_global||^2 in the local objective
      scaffold -- Karimireddy et al., control variates correcting client drift
      fedbn    -- normalization-layer parameters are never aggregated
    """
    set_seed(seed)
    global_model = model_class(**model_kwargs).to(device)
    if warm_start_state is not None:
        global_model.load_state_dict(warm_start_state, strict=False)

    upload = trainable_bytes(global_model, 4)
    splits = mf.split_data_non_iid(train_dataset, num_clients, alpha)
    sizes = [len(s) for s in splits]
    total = max(sum(sizes), 1)

    # SCAFFOLD control variates, held on CPU
    c_server, c_client = None, None
    if algorithm == "scaffold":
        c_server = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
                    for k, v in global_model.state_dict().items()
                    if torch.is_floating_point(v)}
        c_client = [{k: v.clone() for k, v in c_server.items()}
                    for _ in range(num_clients)]

    hist = {"round_f1": [], "round_acc": [], "round_div": [],
            "round_seconds": [], "round_peak_mib": []}

    for t in range(rounds):
        with GPUMem() as gm:
            agg, new_c = {}, []
            global_state = {k: v.detach().cpu().float()
                            for k, v in global_model.state_dict().items()
                            if torch.is_floating_point(v)}

            for k in range(num_clients):
                w = sizes[k] / total
                if len(splits[k]) < 4:
                    for key, val in global_state.items():
                        agg.setdefault(key, torch.zeros_like(val))
                        agg[key].add_(w * val)
                    if algorithm == "scaffold":
                        new_c.append(c_client[k])
                    continue

                local_ds = Subset(train_dataset, splits[k])
                local_lbls = [train_dataset[i]["labels"].argmax().item() for i in splits[k]]
                loader = mf.create_balanced_dataloader(local_ds, local_lbls, cfg.batch_size, 5)

                local_model = copy.deepcopy(global_model).to(device)
                opt = torch.optim.AdamW(local_model.parameters(), lr=cfg.learning_rate,
                                        weight_decay=cfg.weight_decay)
                crit = mf.CombinedLoss(num_classes=5, diversity_weight=1.0)

                # FedProx anchor
                anchor = ({n: p.detach().clone()
                           for n, p in local_model.named_parameters() if p.requires_grad}
                          if algorithm == "fedprox" else None)
                # SCAFFOLD correction, moved to device once
                if algorithm == "scaffold":
                    corr = {n: (c_server[n] - c_client[k][n]).to(device)
                            for n in c_server if n in dict(local_model.named_parameters())}

                steps = 0
                local_model.train()
                for _ in range(local_epochs):
                    for batch in loader:
                        opt.zero_grad()
                        try:
                            out = _forward(local_model, batch, device, model_type)
                            loss = crit(out["logits"], batch["labels"].to(device))
                            if anchor is not None:
                                prox = sum(((p - anchor[n]) ** 2).sum()
                                           for n, p in local_model.named_parameters()
                                           if p.requires_grad and n in anchor)
                                loss = loss + (mu / 2.0) * prox
                            loss.backward()
                            if algorithm == "scaffold":
                                with torch.no_grad():
                                    for n, p in local_model.named_parameters():
                                        if p.grad is not None and n in corr:
                                            p.grad.add_(corr[n])
                            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                            opt.step()
                            steps += 1
                        except RuntimeError as e:
                            print(f"    [warn] client {k}: {e}")
                            break

                local_state = {kk: vv.detach().cpu().float()
                               for kk, vv in local_model.state_dict().items()
                               if torch.is_floating_point(vv)}

                if algorithm == "scaffold":
                    denom = max(steps, 1) * cfg.learning_rate
                    ci_new = {}
                    for n in c_client[k]:
                        drift = (global_state[n] - local_state[n]) / denom
                        ci_new[n] = c_client[k][n] - c_server[n] + drift
                    new_c.append(ci_new)

                for key, val in local_state.items():
                    agg.setdefault(key, torch.zeros_like(val))
                    agg[key].add_(w * val)

                del local_model, opt, crit, loader, local_ds
                if algorithm == "scaffold":
                    del corr
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # apply aggregate; FedBN keeps normalization parameters local
            with torch.no_grad():
                for key, val in global_model.state_dict().items():
                    if key not in agg:
                        continue
                    if algorithm == "fedbn" and _norm_key(key):
                        continue
                    val.copy_(agg[key].to(dtype=val.dtype, device=val.device))

            if algorithm == "scaffold" and new_c:
                for n in c_server:
                    delta = sum(ci[n] - c_client[i][n] for i, ci in enumerate(new_c))
                    c_server[n] = c_server[n] + delta / num_clients
                c_client = new_c

            del agg
            gc.collect()

        m = mf.evaluate(global_model, val_loader, device, model_type)
        hist["round_f1"].append(float(m["f1"]))
        hist["round_acc"].append(float(m["accuracy"]))
        hist["round_div"].append(float(m["diversity"]))
        hist["round_seconds"].append(gm.seconds)
        hist["round_peak_mib"].append(gm.peak_mib)
        print(f"    {log_prefix}round {t+1}/{rounds}  F1={m['f1']:.4f} "
              f"div={m['diversity']:.2f}  {gm.seconds:.1f}s")

    final = mf.evaluate(global_model, val_loader, device, model_type)
    out = {"algorithm": algorithm, "f1": float(final["f1"]),
           "accuracy": float(final["accuracy"]), "diversity": float(final["diversity"]),
           "history": hist, "client_sizes": sizes,
           "upload_bytes_per_client_per_round": upload,
           "total_comm_bytes": 2 * num_clients * rounds * upload,
           "wall_seconds": float(np.sum(hist["round_seconds"])),
           "peak_mib": float(np.max(hist["round_peak_mib"])) if hist["round_peak_mib"] else float("nan")}
    del global_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return out


def local_only_baseline(mf, model_class, model_kwargs, train_dataset, val_loader,
                        cfg, device, model_type="multimodal", *, alpha=1.0,
                        num_clients=5, epochs=6, seed=0):
    """No federation: each hospital trains alone on its own shard.

    This is the baseline that says what federation is actually worth. Without it,
    'federated matches centralized' is only half an argument.
    """
    set_seed(seed)
    splits = mf.split_data_non_iid(train_dataset, num_clients, alpha)
    per_client = []
    for k, idx in enumerate(splits):
        if len(idx) < 8:
            continue
        ds = Subset(train_dataset, idx)
        lbls = [train_dataset[i]["labels"].argmax().item() for i in idx]
        loader = mf.create_balanced_dataloader(ds, lbls, cfg.batch_size, 5)
        model = model_class(**model_kwargs)
        res, _ = centralized_train(mf, model, loader, val_loader, cfg, device,
                                   model_type, epochs=epochs, seed=seed,
                                   log_prefix=f"[local c{k}] ")
        per_client.append(res["f1"])
    return {"per_client_f1": per_client,
            "mean_f1": float(np.mean(per_client)) if per_client else 0.0,
            "std_f1": float(np.std(per_client)) if len(per_client) > 1 else None,
            "best_f1": float(np.max(per_client)) if per_client else 0.0,
            "worst_f1": float(np.min(per_client)) if per_client else 0.0,
            "num_clients": num_clients, "alpha": alpha}


def run_all(mf, store: Store, tier: dict, cfg, data, device, tokenizer, only=None):
    """`only` restricts which experiments run, e.g. only=["E1","E8"].

    Useful when there is no persistent storage: run one block, download the
    results JSON, and the next session picks up from it.
    """
    want = (lambda tag: True) if not only else (lambda tag: tag in set(only))
    mm_kwargs = dict(text_model_name=tier["text_model"],
                     vision_model_name=tier["vision_model"],
                     num_labels=5, fusion_type="concat")

    train_ds, val_ds, train_loader, val_loader = make_mm_loaders(
        mf, data, cfg, tokenizer, balanced=True)
    _, _, unbal_loader, _ = make_mm_loaders(mf, data, cfg, tokenizer, balanced=False)

    # -- warm-start state: one centralized run, reused by E1/E2/E3/E4/E8 ------
    # Only pay for it if something in this chunk actually consumes it; E5/E6/E7
    # do not, and it costs a full centralized training run.
    needs_warm = any(want(t) for t in ("E1", "E2", "E3", "E4", "E8"))
    if needs_warm and not store.has("_warmstart", "concat"):
        print("\n[warm-start] centralized Concat VLM (shared by E1-E4, E8)")
        model = mf.MultiModalClassifier(**mm_kwargs)
        res, state = centralized_train(mf, model, train_loader, val_loader, cfg,
                                       device, "multimodal",
                                       epochs=tier["central_epochs"],
                                       seed=0, log_prefix="[warm] ")
        torch.save(state, Path(store.path).parent / "warmstart_concat.pt")
        store.put("_warmstart", "concat", {k: v for k, v in res.items() if k != "history"})
    warm_path = Path(store.path).parent / "warmstart_concat.pt"
    warm_state = torch.load(warm_path, map_location="cpu") if warm_path.exists() else None

    # ---- E1: alpha sweep ---------------------------------------------------
    if not want('E1'):
        print('skipping E1')
    else:
        print("\n=== E1  Dirichlet alpha sweep ===")
        for alpha in tier["alphas"]:
            for seed in tier["seeds"]:
                key = f"alpha={alpha}|seed={seed}"
                if store.has("E1_alpha_sweep", key):
                    print(f"  skip {key}")
                    continue
                print(f"  {key}")
                r = federated_train_ex(
                    mf, mf.MultiModalClassifier, mm_kwargs, train_ds, val_loader,
                    cfg, device, "multimodal", alpha=alpha, num_clients=5,
                    rounds=tier["fed_rounds"], local_epochs=tier["local_epochs"],
                    warm_start_state=warm_state, seed=seed, log_prefix=f"[a={alpha} s={seed}] ")
                store.put("E1_alpha_sweep", key, r)

    # ---- E2: client-count sweep -------------------------------------------
    if not want('E2'):
        print('skipping E2')
    else:
        print("\n=== E2  Client-count sweep ===")
        for K in tier["client_counts"]:
            for seed in tier["seeds"]:
                key = f"K={K}|seed={seed}"
                if store.has("E2_client_sweep", key):
                    print(f"  skip {key}")
                    continue
                print(f"  {key}")
                r = federated_train_ex(
                    mf, mf.MultiModalClassifier, mm_kwargs, train_ds, val_loader,
                    cfg, device, "multimodal", alpha=1.0, num_clients=K,
                    rounds=tier["fed_rounds"], local_epochs=tier["local_epochs"],
                    warm_start_state=warm_state, seed=seed, log_prefix=f"[K={K} s={seed}] ")
                store.put("E2_client_sweep", key, r)

    # ---- E3: anti-collapse component ablation ------------------------------
    # Run at alpha=0.1 as well as 1.0: the stack is meant to matter under
    # severe heterogeneity, and alpha=1.0 may simply be too easy to show it.
    if not want('E3'):
        print('skipping E3')
    else:
        print("\n=== E3  Anti-collapse component ablation ===")
        variants = {
            "full":          dict(use_balanced_sampler=True,  diversity_weight=1.0),
            "no_balanced":   dict(use_balanced_sampler=False, diversity_weight=1.0),
            "no_diversity":  dict(use_balanced_sampler=True,  diversity_weight=0.0),
            "neither":       dict(use_balanced_sampler=False, diversity_weight=0.0),
        }
        for alpha in sorted({0.1, 1.0} & set(tier["alphas"])) or [tier["alphas"][0]]:
            for vname, vkw in variants.items():
                for seed in tier["seeds"]:
                    key = f"{vname}|alpha={alpha}|seed={seed}"
                    if store.has("E3_anticollapse", key):
                        print(f"  skip {key}")
                        continue
                    print(f"  {key}")
                    r = federated_train_ex(
                        mf, mf.MultiModalClassifier, mm_kwargs, train_ds, val_loader,
                        cfg, device, "multimodal", alpha=alpha, num_clients=5,
                        rounds=tier["fed_rounds"], local_epochs=tier["local_epochs"],
                        warm_start_state=warm_state, seed=seed,
                        log_prefix=f"[{vname} a={alpha}] ", **vkw)
                    store.put("E3_anticollapse", key, r)

        # Centralized arm: isolates the early-abort / LR-boost component, which
        # only exists in the centralized loop.
        print("\n=== E3b  Early-abort component (centralized) ===")
        for use_abort in (True, False):
            for seed in tier["seeds"]:
                key = f"early_abort={use_abort}|seed={seed}"
                if store.has("E3b_early_abort", key):
                    print(f"  skip {key}")
                    continue
                print(f"  {key}")
                model = mf.MultiModalClassifier(**mm_kwargs)
                res, _ = centralized_train(
                    mf, model, unbal_loader, val_loader, cfg, device, "multimodal",
                    epochs=tier["central_epochs"], diversity_weight=0.0,
                    use_early_abort=use_abort, seed=seed,
                    log_prefix=f"[abort={use_abort}] ")
                store.put("E3b_early_abort", key, res)

    # ---- E4: warm start vs cold start --------------------------------------
    if not want('E4'):
        print('skipping E4')
    else:
        print("\n=== E4  Warm-start ablation ===")
        for warm in (True, False):
            for seed in tier["seeds"]:
                key = f"warm_start={warm}|seed={seed}"
                if store.has("E4_warmstart", key):
                    print(f"  skip {key}")
                    continue
                print(f"  {key}")
                r = federated_train_ex(
                    mf, mf.MultiModalClassifier, mm_kwargs, train_ds, val_loader,
                    cfg, device, "multimodal", alpha=1.0, num_clients=5,
                    rounds=tier["fed_rounds"], local_epochs=tier["local_epochs"],
                    warm_start_state=warm_state if warm else None, seed=seed,
                    log_prefix=f"[warm={warm}] ")
                store.put("E4_warmstart", key, r)

    # ---- E5: fusion strategies x seeds (variance) --------------------------
    if not want('E5'):
        print('skipping E5')
    else:
        print("\n=== E5  Fusion strategies, repeated seeds ===")
        for fusion in FUSION_TYPES:
            for seed in tier["fusion_seeds"]:
                key = f"{fusion}|seed={seed}"
                if store.has("E5_fusion_seeds", key):
                    print(f"  skip {key}")
                    continue
                print(f"  {key}")
                kw = dict(mm_kwargs, fusion_type=fusion)
                model = mf.MultiModalClassifier(**kw)
                res, _ = centralized_train(
                    mf, model, train_loader, val_loader, cfg, device, "multimodal",
                    epochs=tier["central_epochs"], seed=seed,
                    log_prefix=f"[{fusion} s={seed}] ")
                store.put("E5_fusion_seeds", key, res)

    # ---- E6: measured cost per branch --------------------------------------
    if not want('E6'):
        print('skipping E6')
    else:
        print("\n=== E6  Measured runtime / memory / communication ===")
        branches = {
            "Fed-LLM": (mf.LightweightTextClassifier,
                        dict(model_name=tier["text_model"], num_labels=5), "text"),
            "Fed-ViT": (mf.LightweightVisionClassifier,
                        dict(model_name=tier["vision_model"], num_labels=5), "image"),
            "Fed-VLM": (mf.MultiModalClassifier, mm_kwargs, "multimodal"),
        }
        for name, (cls, kw, mtype) in branches.items():
            key = f"{name}"
            if store.has("E6_cost", key):
                print(f"  skip {key}")
                continue
            print(f"  {key}")
            r = federated_train_ex(
                mf, cls, kw, train_ds, val_loader, cfg, device, mtype,
                alpha=1.0, num_clients=5, rounds=tier["fed_rounds"],
                local_epochs=tier["local_epochs"], seed=0, log_prefix=f"[{name}] ")
            store.put("E6_cost", key, r)

    # ---- E7: RAG on a held-out query set -----------------------------------
    if not want('E7'):
        print('skipping E7')
    else:
        print("\n=== E7  Retrieval evaluation (held-out queries) ===")
        if not store.has("E7_rag", "heldout"):
            store.put("E7_rag", "heldout", run_rag_eval(mf, data, cfg, device, tokenizer))

    # ---- E8: matched-setting federated baselines ---------------------------
    # R3.3: "compare against recent SOTA multimodal FL using consistent
    # experimental settings". Published numbers from other corpora cannot do
    # this. Here every arm uses the same backbone, data, partition and budget,
    # so the aggregation rule is the only variable.
    if not want('E8'):
        print('skipping E8')
    else:
        print("\n=== E8  Federated baselines, matched settings ===")
        algos = ["fedavg", "fedprox", "scaffold", "fedbn"]
        # Run at moderate AND severe heterogeneity: the drift-correcting methods
        # exist for the severe case, so alpha=1.0 alone would not separate them.
        e8_alphas = sorted({1.0, 0.1} & set(tier["alphas"])) or [tier["alphas"][0]]
        for alpha in e8_alphas:
            for algo in algos:
                for seed in tier["seeds"]:
                    key = f"{algo}|alpha={alpha}|seed={seed}"
                    if store.has("E8_baselines", key):
                        print(f"  skip {key}")
                        continue
                    print(f"  {key}")
                    r = federated_train_algo(
                        mf, mf.MultiModalClassifier, mm_kwargs, train_ds, val_loader,
                        cfg, device, "multimodal", algorithm=algo, alpha=alpha,
                        num_clients=5, rounds=tier["fed_rounds"],
                        local_epochs=tier["local_epochs"], warm_start_state=warm_state,
                        seed=seed, log_prefix=f"[{algo} a={alpha}] ")
                    store.put("E8_baselines", key, r)

            # local-only: what each hospital gets without federating at all
            key = f"local_only|alpha={alpha}|seed={tier['seeds'][0]}"
            if not store.has("E8_baselines", key):
                print(f"  {key}")
                store.put("E8_baselines", key, local_only_baseline(
                    mf, mf.MultiModalClassifier, mm_kwargs, train_ds, val_loader,
                    cfg, device, "multimodal", alpha=alpha, num_clients=5,
                    epochs=max(4, tier["central_epochs"] // 2), seed=tier["seeds"][0]))

def run_rag_eval(mf, data, cfg, device, tokenizer, top_k: int = 5):
    """Index the training notes, query with every held-out validation note.

    Replaces the 5-query probe with the full validation split, and reports
    retrieval precision@k and top-1 similarity per condition.
    """
    try:
        import faiss  # noqa
    except ImportError:
        print("  faiss not installed -- pip install faiss-cpu; skipping E7")
        return {"error": "faiss not installed"}

    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = [str(t) for t in data["train_texts"]]
    corpus_labels = [l[0] for l in data["train_tlbls"]]
    queries = [str(t) for t in data["val_texts"]]
    query_labels = [l[0] for l in data["val_tlbls"]]

    # Encoder-free baseline embedding so this runs even without a trained model;
    # swap in the fine-tuned DistilBERT encoder if you want the paper's exact setup.
    vec = TfidfVectorizer(max_features=4096).fit(corpus + queries)
    C = vec.transform(corpus).toarray().astype("float32")
    Q = vec.transform(queries).toarray().astype("float32")
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)

    import faiss
    index = faiss.IndexFlatIP(C.shape[1])
    index.add(C)
    sims, idxs = index.search(Q, top_k)

    per_cond = {i: {"n": 0, "top1_correct": 0, "prec_at_k": 0.0, "top1_sim": []}
                for i in range(5)}
    for qi, ql in enumerate(query_labels):
        retrieved = [corpus_labels[j] for j in idxs[qi]]
        d = per_cond[ql]
        d["n"] += 1
        d["top1_correct"] += int(retrieved[0] == ql)
        d["prec_at_k"] += sum(r == ql for r in retrieved) / top_k
        d["top1_sim"].append(float(sims[qi][0]))

    out = {"n_queries": len(queries), "corpus_size": len(corpus), "top_k": top_k,
           "per_condition": {}}
    for ci, d in per_cond.items():
        if d["n"] == 0:
            continue
        out["per_condition"][mf.CONDITION_LABELS[ci]] = {
            "n": d["n"],
            "top1_accuracy": d["top1_correct"] / d["n"],
            "precision_at_k": d["prec_at_k"] / d["n"],
            "mean_top1_similarity": float(np.mean(d["top1_sim"])),
            "std_top1_similarity": float(np.std(d["top1_sim"])),
        }
    out["overall_top1_accuracy"] = float(
        np.mean([v["top1_accuracy"] for v in out["per_condition"].values()]))
    out["overall_mean_top1_similarity"] = float(
        np.mean([v["mean_top1_similarity"] for v in out["per_condition"].values()]))
    print(f"  retrieval top-1 acc={out['overall_top1_accuracy']:.3f} "
          f"mean sim={out['overall_mean_top1_similarity']:.3f} "
          f"over {out['n_queries']} queries")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(base_py: str, tier: str = "standard", out: str = "results_v2.json",
         cache: Optional[str] = None, only=None, seeds=None, alphas=None):
    t = dict(TIERS[tier])  # copy, so overrides do not mutate the module table
    if seeds:
        t["seeds"] = list(seeds)
        t["fusion_seeds"] = list(seeds)
    if alphas:
        t["alphas"] = list(alphas)
    print("=" * 68)
    print(f"OmniMed-FL experiment suite | tier={tier}")
    print("=" * 68)

    _patch_transformers()
    _patch_datasets()
    mf = load_base(base_py)
    _patch_image_loader(mf)
    if not verify_encoders(mf, [t["text_model"]]):
        raise SystemExit(
            "\nStopping: the text encoder is not loading pretrained weights, "
            "so these results would be meaningless. Fix that before running.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("\n!! No GPU detected. Runtime -> Change runtime type -> GPU.")
        print("   Continuing on CPU; expect this to be very slow.\n")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg = mf.Config(
        batch_size=t["batch_size"],
        max_samples_per_class=t["samples_per_class"],
        fed_rounds=t["fed_rounds"],
        local_epochs=t["local_epochs"],
        epochs=t["central_epochs"],
    )

    out_path = Path(out)
    cache_path = Path(cache) if cache else out_path.parent / f"data_cache_{tier}.pkl"
    data = build_data(mf, cfg, cache_path)
    tokenizer = mf.get_text_tokenizer(t["text_model"], cfg.max_seq_length)

    store = Store(out_path)
    store.data.setdefault("_meta", {})
    store.data["_meta"].update({
        "tier": tier, "tier_config": t,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "n_train": len(data["train_texts"]), "n_val": len(data["val_texts"]),
    })
    store.flush()

    t0 = time.perf_counter()
    run_all(mf, store, t, cfg, data, device, tokenizer, only=only)
    store.data["_meta"]["total_wall_seconds"] = time.perf_counter() - t0
    store.flush()

    print("\n" + "=" * 68)
    print(f"Done in {(time.perf_counter()-t0)/3600:.2f} h -> {out_path}")
    print("Next: python omnimed_make_tables.py --results", out_path)
    print("=" * 68)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="path to MedFederate_Colab_Complete.py")
    ap.add_argument("--tier", default="standard", choices=list(TIERS))
    ap.add_argument("--out", default="results_v2.json")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated subset, e.g. E1,E8")
    ap.add_argument("--seeds", default=None,
                    help="override tier seeds, e.g. 0,1,2. Completed keys are "
                         "skipped, so this fills in only the new seeds.")
    ap.add_argument("--alphas", default=None,
                    help="override tier alphas, e.g. 0.05,0.1,1.0")
    a = ap.parse_args()
    main(a.base, a.tier, a.out, a.cache,
         only=[x.strip() for x in a.only.split(',')] if a.only else None,
         seeds=[int(x) for x in a.seeds.split(',')] if a.seeds else None,
         alphas=[float(x) for x in a.alphas.split(',')] if a.alphas else None)
