# =============================================================================
# OmniMed-FL - reviewer experiment suite, single-file KAGGLE build
# =============================================================================
# Paste this whole file into ONE Kaggle notebook cell and run it.
#
# Before running, in the right-hand panel:
#   Settings -> Accelerator -> GPU T4 x2  (or P100)
#   Settings -> Internet    -> On         (needed for HuggingFace + GitHub)
#
# No uploads. The training module is fetched from
#   github.com/ayushdebnath012/IEEE-Globecom
#
# Kaggle beats Colab here in two ways:
#   - /kaggle/working persists for the whole session and is saved with the
#     notebook version, so results survive without downloading after every step
#   - sessions run up to 9 h, long enough for the full sweep in one go
#
# What runs, and which reviewer point it answers:
#   E1 Dirichlet alpha sweep .................. R3.1
#   E2 client-count sweep K ................... R3.4
#   E3 anti-collapse component ablation ....... R3.2
#   E4 warm-start ablation .................... R3.2
#   E5 fusion strategies x seeds .............. R3.1
#   E6 measured runtime / memory / comm ....... R1.2, R3.4
#   E7 retrieval over the held-out split ...... R3.2
#   E8 FedAvg/FedProx/SCAFFOLD/FedBN/local .... R3.3
# =============================================================================

from __future__ import annotations

import argparse, copy, gc, importlib.util, json, os, pickle, random, re, sys, time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
except ImportError:
    raise SystemExit("PyTorch missing - check the Kaggle accelerator setting.")

# Kaggle always sets these; a bare path check false-positives on machines
# that happen to have a C:\kaggle or /kaggle directory lying around.
ON_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
                 or os.environ.get("KAGGLE_URL_BASE")) or (
    os.name != "nt" and Path("/kaggle/working").exists()
    and Path("/kaggle/input").exists())
WORK = Path("/kaggle/working") if ON_KAGGLE else Path(".")
IN_NOTEBOOK = ON_KAGGLE or "ipykernel" in sys.modules

BASE_NAME = "MedFederate_Colab_Complete.py"
BASE_URL = ("https://raw.githubusercontent.com/ayushdebnath012/"
            "IEEE-Globecom/main/source/" + BASE_NAME)

# =============================================================================
# Pretrained-encoder fix  (do not remove)
# =============================================================================
# MedFederate_Colab_Complete.py loads text encoders with AutoModel inside a
# try/except. Newer transformers refuses prajjwal1/bert-* because their
# config.json has no "model_type" key, the except fires, and the model silently
# becomes a RANDOMLY INITIALISED nn.Embedding + 2-layer encoder. Training then
# "works" and reports numbers that contain no pretrained language model at all.
# This restores the intended behaviour and then verifies it loudly.

def _patch_transformers():
    """Route configs/models that Auto* rejects to the BERT classes explicitly."""
    try:
        from transformers import AutoConfig, AutoModel, BertConfig, BertModel
    except ImportError:
        print("  [patch] transformers not importable yet; skipping")
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


def verify_encoders(mf, names=None):
    """Instantiate each text encoder and assert real weights loaded.

    The original failure mode is silent, so this is deliberately noisy. If any
    encoder reports FALLBACK, stop: results from that run would be meaningless.
    """
    names = names or ["distilbert-base-uncased", "prajjwal1/bert-tiny",
                      "prajjwal1/bert-mini", "prajjwal1/bert-small",
                      "prajjwal1/bert-medium"]
    print("\n  verifying text encoders load pretrained weights:")
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
    if bad:
        print("\n  !! These fell back to a RANDOM encoder: " + ", ".join(bad))
        print("  !! Any text or fusion result from them is not a real result.")
    else:
        print("  all encoders loaded pretrained weights")
    return not bad


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
         cache: Optional[str] = None, only=None):
    t = TIERS[tier]
    print("=" * 68)
    print(f"OmniMed-FL experiment suite | tier={tier}")
    print("=" * 68)

    mf = load_base(base_py)
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


def parse_key(key: str) -> dict:
    out = {}
    for part in key.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
        else:
            out["variant"] = part
    return out


FUSION_DISPLAY = {
    "concat": "Concat", "attention": "Cross-Attn", "gated": "Gated",
    "clip": "CLIP", "flamingo": "Flamingo", "blip2": "BLIP-2",
    "coca": "CoCa", "unified_io": "Unified-IO",
}


def tex_escape(s: str) -> str:
    """Escape the characters that break a LaTeX table cell."""
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def agg(values):
    a = np.asarray(values, dtype=float)
    if a.size == 1:
        return a[0], None
    return a.mean(), a.std(ddof=1)


def fmt(mean, std, prec=3):
    if std is None:
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f}\\,$\\pm$\\,{std:.{prec}f}"


def group(exp: dict, by: str, metric: str = "f1"):
    """Collect metric values grouped by one key field."""
    buckets = defaultdict(list)
    for key, rec in exp.items():
        p = parse_key(key)
        if by in p:
            buckets[p[by]].append(rec[metric])
    return buckets


# ---------------------------------------------------------------------------


def tab_alpha(d, outdir):
    exp = d.get("E1_alpha_sweep")
    if not exp:
        return None
    rows = []
    f1 = group(exp, "alpha", "f1")
    div = group(exp, "alpha", "diversity")
    for a in sorted(f1, key=float):
        m, s = agg(f1[a])
        dm, _ = agg(div[a])
        # realized heterogeneity: mean over clients of max class share
        skews = []
        for key, rec in exp.items():
            if parse_key(key)["alpha"] != a:
                continue
            for h in rec.get("client_class_hist", []):
                tot = sum(h)
                if tot:
                    skews.append(max(h) / tot)
        skew = np.mean(skews) if skews else float("nan")
        rows.append((a, skew, fmt(m, s), f"{dm:.2f}"))

    tex = [
        r"\begin{table}[t]",
        r"\caption{Effect of partition heterogeneity on Fed-VLM. $\alpha$ is the",
        r"Dirichlet concentration; \emph{skew} is the realized mean per-client share of",
        r"the dominant class. Mean\,$\pm$\,std over seeds.}",
        r"\label{tab:alpha}",
        r"\centering\footnotesize",
        r"\setlength\tabcolsep{5pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$\alpha$ & Skew & Macro F1 & Diversity\\",
        r"\midrule",
    ]
    for a, skew, f, dv in rows:
        tex.append(f"{a} & {skew:.2f} & {f} & {dv} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(outdir / "tab_alpha_sweep.tex", tex)


def tab_clients(d, outdir):
    exp = d.get("E2_client_sweep")
    if not exp:
        return None
    f1 = group(exp, "K", "f1")
    tex = [
        r"\begin{table}[t]",
        r"\caption{Scalability in the number of clients $K$ at $\alpha{=}1.0$.",
        r"Communication is the measured total exchanged over the run,",
        r"$V=2KT|\theta|b$. Mean\,$\pm$\,std over seeds.}",
        r"\label{tab:clients}",
        r"\centering\footnotesize",
        r"\setlength\tabcolsep{5pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$K$ & Macro F1 & Wall-clock (s) & Total comm. (GiB)\\",
        r"\midrule",
    ]
    for K in sorted(f1, key=int):
        m, s = agg(f1[K])
        secs = [r["wall_seconds"] for k, r in exp.items() if parse_key(k)["K"] == K]
        gib = [r["total_comm_bytes"] / 2 ** 30 for k, r in exp.items()
               if parse_key(k)["K"] == K]
        tex.append(f"{K} & {fmt(m, s)} & {np.mean(secs):.0f} & {np.mean(gib):.1f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(outdir / "tab_client_sweep.tex", tex)


def tab_anticollapse(d, outdir):
    exp = d.get("E3_anticollapse")
    if not exp:
        return None
    label = {"full": "Full stack",
             "no_balanced": r"$-$ balanced sampler",
             "no_diversity": r"$-$ diversity term",
             "neither": r"$-$ both"}
    alphas = sorted({parse_key(k)["alpha"] for k in exp}, key=float)
    tex = [
        r"\begin{table}[t]",
        r"\caption{Anti-collapse component ablation on Fed-VLM. Each row removes one",
        r"component from the full stack. Diversity is the fraction of the five classes",
        r"predicted at the final round. Mean\,$\pm$\,std over seeds.}",
        r"\label{tab:anticollapse}",
        r"\centering\footnotesize",
        r"\setlength\tabcolsep{4pt}",
        r"\begin{tabular}{l" + "cc" * len(alphas) + "}",
        r"\toprule",
        r" & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{$\alpha={a}$}}" for a in alphas) + r"\\",
    ]
    tex.append("".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(alphas))))
    tex.append(r"Configuration & " + " & ".join(["F1 & Div."] * len(alphas)) + r"\\")
    tex.append(r"\midrule")
    for v in ["full", "no_balanced", "no_diversity", "neither"]:
        cells = []
        for a in alphas:
            f1s = [r["f1"] for k, r in exp.items()
                   if parse_key(k).get("variant") == v and parse_key(k)["alpha"] == a]
            dvs = [r["diversity"] for k, r in exp.items()
                   if parse_key(k).get("variant") == v and parse_key(k)["alpha"] == a]
            if not f1s:
                cells += ["--", "--"]
                continue
            m, s = agg(f1s)
            cells += [fmt(m, s), f"{np.mean(dvs):.2f}"]
        tex.append(f"{label[v]} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(outdir / "tab_anticollapse.tex", tex)


def tab_warmstart(d, outdir):
    exp = d.get("E4_warmstart")
    if not exp:
        return None
    f1 = group(exp, "warm_start", "f1")
    tex = [
        r"\begin{table}[t]",
        r"\caption{Warm-start ablation: federated training from centralized",
        r"pretrained weights versus from a cold initialization.}",
        r"\label{tab:warmstart}",
        r"\centering\footnotesize",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Initialization & Macro F1 & Round-1 F1\\",
        r"\midrule",
    ]
    for w, name in [("True", "Warm start (pretrained)"), ("False", "Cold start (random)")]:
        if w not in f1:
            continue
        m, s = agg(f1[w])
        r1 = [r["history"]["round_f1"][0] for k, r in exp.items()
              if parse_key(k)["warm_start"] == w]
        tex.append(f"{name} & {fmt(m, s)} & {np.mean(r1):.3f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(outdir / "tab_warmstart.tex", tex)


def tab_fusion(d, outdir):
    exp = d.get("E5_fusion_seeds")
    if not exp:
        return None
    buckets = defaultdict(list)
    for k, r in exp.items():
        buckets[parse_key(k)["variant"]].append(r["f1"])
    ordered = sorted(buckets.items(), key=lambda kv: -np.mean(kv[1]))
    tex = [
        r"\begin{table}[t]",
        r"\caption{Fusion strategies with repeated seeds. The std column is what",
        r"decides whether the ordering is real; strategies whose intervals overlap",
        r"should be treated as tied.}",
        r"\label{tab:fusionvar}",
        r"\centering\footnotesize",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Fusion & Macro F1 & Seeds\\",
        r"\midrule",
    ]
    for name, vals in ordered:
        m, s = agg(vals)
        disp = FUSION_DISPLAY.get(name, tex_escape(name))
        tex.append(f"{disp} & {fmt(m, s)} & {len(vals)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(outdir / "tab_fusion_variance.tex", tex)


def tab_cost(d, outdir):
    exp = d.get("E6_cost")
    if not exp:
        return None
    tex = [
        r"\begin{table}[t]",
        r"\caption{Measured cost per branch at $K{=}5$, $\alpha{=}1.0$. Payload is",
        r"per client per round; wall-clock and peak memory are measured on the GPU",
        r"named in the results metadata.}",
        r"\label{tab:comm}",
        r"\centering\footnotesize",
        r"\setlength\tabcolsep{4pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Branch & $|\theta|$ & Payload & Total $V$ & s/round & Peak mem.\\",
        r"\midrule",
    ]
    for name in ["Fed-LLM", "Fed-ViT", "Fed-VLM"]:
        r = exp.get(name)
        if not r:
            continue
        p = r["trainable_params"] / 1e6
        pay = r["upload_bytes_per_client_per_round"] / 2 ** 20
        tot = r["total_comm_bytes"] / 2 ** 30
        spr = np.mean(r["history"]["round_seconds"])
        mem = r["peak_mib"]
        tex.append(f"{name} & {p:.1f}\\,M & {pay:.0f}\\,MiB & {tot:.1f}\\,GiB "
                   f"& {spr:.0f} & {mem:.0f}\\,MiB \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(outdir / "tab_measured_cost.tex", tex)


def tab_rag(d, outdir):
    exp = (d.get("E7_rag") or {}).get("heldout")
    if not exp or "error" in exp:
        return None
    tex = [
        r"\begin{table}[t]",
        rf"\caption{{Retrieval evaluation over {exp['n_queries']} held-out queries",
        rf"against a {exp['corpus_size']}-note index (top-$k$={exp['top_k']}).}}",
        r"\label{tab:rag}",
        r"\centering\footnotesize",
        r"\setlength\tabcolsep{4pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Condition & $n$ & Top-1 acc. & P@$k$ & Top-1 sim.\\",
        r"\midrule",
    ]
    for cond, v in exp["per_condition"].items():
        tex.append(f"{tex_escape(cond.replace('_',' ').title())} & {v['n']} & "
                   f"{v['top1_accuracy']:.3f} & {v['precision_at_k']:.3f} & "
                   f"{v['mean_top1_similarity']:.3f}\\,$\\pm$\\,{v['std_top1_similarity']:.3f} \\\\")
    tex += [
        r"\midrule",
        rf"\textbf{{Overall}} & {exp['n_queries']} & "
        rf"\textbf{{{exp['overall_top1_accuracy']:.3f}}} & -- & "
        rf"{exp['overall_mean_top1_similarity']:.3f} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    return _write(outdir / "tab_rag.tex", tex)


def tab_baselines(d, outdir):
    """R3.3 -- matched-setting comparison. This is the table that replaces the
    literature-positioning Table IV, because every arm here ran on our data."""
    exp = d.get("E8_baselines")
    if not exp:
        return None
    label = {"local_only": "Local only (no federation)", "fedavg": "FedAvg",
             "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fedbn": "FedBN"}
    alphas = sorted({parse_key(k)["alpha"] for k in exp}, key=float)

    tex = [
        r"\begin{table}[t]",
        r"\caption{Federated aggregation rules under \emph{matched} settings: same",
        r"Concat-VLM backbone, data, partition, local budget and hyperparameters, so",
        r"the aggregation rule is the only variable. Mean\,$\pm$\,std over seeds.}",
        r"\label{tab:baselines}",
        r"\centering\footnotesize",
        r"\setlength\tabcolsep{5pt}",
        r"\begin{tabular}{l" + "c" * len(alphas) + "}",
        r"\toprule",
        r"\textbf{Method} & " + " & ".join(rf"$\alpha={a}$" for a in alphas) + r"\\",
        r"\midrule",
    ]
    for m in ["local_only", "fedavg", "fedprox", "scaffold", "fedbn"]:
        cells = []
        for a in alphas:
            if m == "local_only":
                recs = [r for k, r in exp.items()
                        if parse_key(k).get("variant") == m and parse_key(k)["alpha"] == a]
                if not recs:
                    cells.append("--")
                    continue
                r = recs[0]
                cells.append(f"{r['mean_f1']:.3f}")
            else:
                vals = [r["f1"] for k, r in exp.items()
                        if parse_key(k).get("variant") == m and parse_key(k)["alpha"] == a]
                if not vals:
                    cells.append("--")
                    continue
                mn, sd = agg(vals)
                cells.append(fmt(mn, sd))
        tex.append(f"{label[m]} & " + " & ".join(cells) + r" \\")
    tex += [
        r"\bottomrule", r"\end{tabular}",
        r"\vspace{1pt}", "",
        r"{\scriptsize Local-only is the mean over the five clients, each trained on its",
        r"own shard alone. Every other row is the global model after $T$ rounds.}",
        r"\end{table}",
    ]
    return _write(outdir / "tab_baselines.tex", tex)


def figs(d, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    e1 = d.get("E1_alpha_sweep")
    if e1:
        f1 = group(e1, "alpha", "f1")
        xs = sorted(f1, key=float)
        m = [np.mean(f1[a]) for a in xs]
        s = [np.std(f1[a], ddof=1) if len(f1[a]) > 1 else 0 for a in xs]
        fig, ax = plt.subplots(figsize=(3.4, 2.0), dpi=300)
        ax.errorbar([float(a) for a in xs], m, yerr=s, marker="o", ms=4,
                    lw=1.6, color="#C44E52", capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel(r"Dirichlet $\alpha$ (log scale)", fontsize=8)
        ax.set_ylabel("Macro F1", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(pad=0.2)
        p = outdir / "fig_alpha_sweep.png"
        fig.savefig(p, bbox_inches="tight")
        made.append(p)

    e2 = d.get("E2_client_sweep")
    if e2:
        f1 = group(e2, "K", "f1")
        xs = sorted(f1, key=int)
        m = [np.mean(f1[k]) for k in xs]
        fig, ax = plt.subplots(figsize=(3.4, 2.0), dpi=300)
        ax.plot([int(k) for k in xs], m, marker="s", ms=4, lw=1.6, color="#4C72B0")
        ax.set_xlabel("Number of clients $K$", fontsize=8)
        ax.set_ylabel("Macro F1", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(pad=0.2)
        p = outdir / "fig_client_scaling.png"
        fig.savefig(p, bbox_inches="tight")
        made.append(p)
    return made


def summary(d, outdir):
    """Plain-language readout, so the numbers get interpreted before they get pasted."""
    L = ["# Experiment readout", ""]
    meta = d.get("_meta", {})
    L += [f"- tier: `{meta.get('tier')}`  GPU: `{meta.get('gpu')}`",
          f"- train/val: {meta.get('n_train')}/{meta.get('n_val')}", ""]

    e1 = d.get("E1_alpha_sweep")
    if e1:
        f1 = group(e1, "alpha", "f1")
        xs = sorted(f1, key=float)
        lo, hi = np.mean(f1[xs[0]]), np.mean(f1[xs[-1]])
        ref = np.mean(f1["1.0"]) if "1.0" in f1 else hi
        L += ["## E1 alpha sweep (R3.1)",
              f"- most heterogeneous (alpha={xs[0]}): F1 {lo:.3f}",
              f"- least heterogeneous (alpha={xs[-1]}): F1 {hi:.3f}",
              f"- cost of severe heterogeneity vs the paper's alpha=1.0: {lo-ref:+.3f}",
              "- If that cost is small, the 99.1% retention claim generalizes beyond "
              "alpha=1.0 and you can say so. If it is large, the claim must be scoped "
              "to moderate heterogeneity.", ""]

    e3 = d.get("E3_anticollapse")
    if e3:
        L += ["## E3 anti-collapse ablation (R3.2)"]
        for v in ["full", "no_balanced", "no_diversity", "neither"]:
            vals = [r["f1"] for k, r in e3.items() if parse_key(k).get("variant") == v]
            dvs = [r["diversity"] for k, r in e3.items() if parse_key(k).get("variant") == v]
            if vals:
                L.append(f"- {v}: F1 {np.mean(vals):.3f}, final diversity {np.mean(dvs):.2f}")
        L += ["- This is the table the reviewer asked for. If `neither` does not "
              "collapse, the anti-collapse stack is not doing the work the paper "
              "attributes to it -- say that rather than keep the claim.", ""]

    e5 = d.get("E5_fusion_seeds")
    if e5:
        b = defaultdict(list)
        for k, r in e5.items():
            b[parse_key(k)["variant"]].append(r["f1"])
        stds = [np.std(v, ddof=1) for v in b.values() if len(v) > 1]
        if stds:
            L += ["## E5 fusion variance (R3.1)",
                  f"- median seed std: {np.median(stds):.4f}",
                  f"- spread across strategies: "
                  f"{max(np.mean(v) for v in b.values()) - min(np.mean(v) for v in b.values()):.4f}",
                  "- If the spread is within ~2x the seed std, the fusion ranking is "
                  "noise and the paper should report them as tied.", ""]

    e6 = d.get("E6_cost")
    if e6 and "Fed-VLM" in e6 and "Fed-LLM" in e6:
        a = e6["Fed-VLM"]["upload_bytes_per_client_per_round"]
        b_ = e6["Fed-LLM"]["upload_bytes_per_client_per_round"]
        L += ["## E6 measured cost (R1.2/R3.4)",
              f"- Fed-VLM payload {a/2**20:.0f} MiB vs Fed-LLM {b_/2**20:.0f} MiB "
              f"({a/b_:.2f}x)",
              f"- Fed-VLM {np.mean(e6['Fed-VLM']['history']['round_seconds']):.0f} s/round, "
              f"peak {e6['Fed-VLM']['peak_mib']:.0f} MiB",
              "- These replace the analytic Table V. Swap them in and delete the "
              "'analytic, not measured' caveat.", ""]

    e8 = d.get("E8_baselines")
    if e8:
        L += ["## E8 matched-setting baselines (R3.3)"]
        for a in sorted({parse_key(k)["alpha"] for k in e8}, key=float):
            row = []
            for m in ["local_only", "fedavg", "fedprox", "scaffold", "fedbn"]:
                recs = [r for k, r in e8.items()
                        if parse_key(k).get("variant") == m and parse_key(k)["alpha"] == a]
                if not recs:
                    continue
                v = (recs[0]["mean_f1"] if m == "local_only"
                     else np.mean([r["f1"] for r in recs]))
                row.append(f"{m} {v:.3f}")
            L.append(f"- alpha={a}: " + ", ".join(row))
        L += ["- This is the comparison the reviewer asked for, and the only one in "
              "the paper run under matched conditions. Two readings matter: how far "
              "every federated arm sits above local-only (that is what federation "
              "buys), and whether the drift-correcting methods separate from FedAvg "
              "at low alpha. If FedProx/SCAFFOLD beat FedAvg at alpha=0.1, say so and "
              "switch the aggregator -- do not keep FedAvg because it is what the "
              "submitted version used.", ""]

    e7 = (d.get("E7_rag") or {}).get("heldout")
    if e7 and "error" not in e7:
        L += ["## E7 retrieval (R3.2)",
              f"- {e7['n_queries']} queries, top-1 acc {e7['overall_top1_accuracy']:.3f}, "
              f"mean sim {e7['overall_mean_top1_similarity']:.3f}",
              "- Replaces the 5-query probe. Report the similarity range honestly; "
              "do not restore the 0.89 claim unless this run produces it.", ""]

    p = outdir / "SUMMARY.md"
    p.write_text("\n".join(L), encoding="utf8")
    return p


def _write(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")
    return path


def _tables_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    d = json.loads(Path(a.results).read_text())
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    made = [f for f in [tab_alpha(d, outdir), tab_clients(d, outdir),
                        tab_anticollapse(d, outdir), tab_warmstart(d, outdir),
                        tab_fusion(d, outdir), tab_cost(d, outdir),
                        tab_rag(d, outdir), tab_baselines(d, outdir)] if f]
    made += figs(d, outdir)
    made.append(summary(d, outdir))

    print("Wrote:")
    for p in made:
        print("  ", p)
    print("\nRead SUMMARY.md before pasting anything into the paper.")


# =============================================================================
# Kaggle driver
# =============================================================================


def _ensure_base() -> str:
    """Find the training module, or fetch it from GitHub. No manual upload."""
    # a Kaggle Dataset attached to the notebook
    for root in Path("/kaggle/input").glob("*") if Path("/kaggle/input").exists() else []:
        c = root / BASE_NAME
        if c.exists():
            print(f"  base module from attached dataset: {c}")
            return str(c)
    local = WORK / BASE_NAME
    if local.exists():
        print(f"  base module already present: {local}")
        return str(local)
    print(f"  fetching {BASE_NAME} from GitHub ...")
    try:
        urllib.request.urlretrieve(BASE_URL, str(local))
    except Exception as e:
        raise SystemExit(
            f"Could not download the training module ({e}).\n"
            f"Turn Internet ON in the Settings panel, or attach it as a Dataset.")
    print(f"  saved to {local} ({local.stat().st_size/1024:.0f} KB)")
    return str(local)


def _ensure_faiss():
    try:
        import faiss  # noqa
    except ImportError:
        print("  installing faiss-cpu (needed by E7) ...")
        os.system(f"{sys.executable} -m pip install -q faiss-cpu")


def run(tier="smoke", only=None, out=None, verify=True):
    """Run a chunk.

        run("smoke")                        ~20 min, validates everything
        run("standard", ["E1","E2"])        alpha + client sweeps
        run("standard", ["E3","E4"])        ablations
        run("standard", ["E8"])             matched baselines
        run("standard", ["E5","E6","E7"])   variance, cost, retrieval
        run("standard")                     everything in one session
    """
    base = _ensure_base()
    _ensure_faiss()
    _patch_transformers()

    if verify:
        mf = load_base(base)
        want_text = TIERS[tier]["text_model"]
        if not verify_encoders(mf, [want_text]):
            raise SystemExit(
                "\nStopping: the text encoder is not loading pretrained weights, so\n"
                "these results would be meaningless. Fix that before running.")

    out = out or str(WORK / ("results_smoke.json" if tier == "smoke" else "results_v2.json"))
    main(base_py=base, tier=tier, out=out, only=only)
    print(f"\n  results at {out}")
    print("  /kaggle/working persists for this session; Save Version to keep it.")
    return out


def build_tables(results=None, outdir=None):
    """Turn the results into LaTeX tables, figures and SUMMARY.md."""
    results = results or str(WORK / "results_v2.json")
    outdir = outdir or str(WORK / "paper_assets")
    if not Path(results).exists():
        raise FileNotFoundError(f"{results} not found - run the chunks first.")
    sys.argv = ["omnimed", "--results", results, "--outdir", outdir]
    _tables_main()
    print("\n" + "=" * 70)
    s = Path(outdir) / "SUMMARY.md"
    if s.exists():
        print(s.read_text(encoding="utf8"))
    import shutil as _sh
    _sh.make_archive(str(WORK / "paper_assets"), "zip", outdir)
    print(f"\n  zipped -> {WORK}/paper_assets.zip")
    print("  Save Version, then grab it from the notebook's Output tab.")
    return outdir


def full_run():
    """Every chunk in order. Kaggle sessions are long enough to attempt this."""
    for only in (["E1", "E2"], ["E3", "E4"], ["E8"], ["E5", "E6", "E7"]):
        print("\n" + "#" * 70)
        print("# CHUNK", only)
        print("#" * 70)
        run("standard", only, verify=False)
    build_tables()


def _banner():
    print("=" * 70)
    print("OmniMed-FL experiment suite - Kaggle build")
    print("=" * 70)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    else:
        print("!! NO GPU. Settings -> Accelerator -> GPU, then rerun this cell.")
    if ON_KAGGLE:
        print("working dir:", WORK, "(persists for the session)")
    try:
        urllib.request.urlopen("https://raw.githubusercontent.com", timeout=5)
        print("internet: on")
    except Exception:
        print("!! internet appears OFF - Settings -> Internet -> On")
    print("""
Run these one at a time:

    run("smoke")                        # ~20 min  - do this first
    run("standard", ["E1","E2"])        # ~2-3 h   - alpha + client sweeps
    run("standard", ["E3","E4"])        # ~2-3 h   - ablations
    run("standard", ["E8"])             # ~2-3 h   - matched baselines
    run("standard", ["E5","E6","E7"])   # ~1-2 h   - variance, cost, retrieval
    build_tables()                      # LaTeX tables + SUMMARY.md

or  full_run()   to attempt all of it in one session.

Results stay in /kaggle/working. Click Save Version before the session ends,
or they are lost. Finished work is skipped if you re-run a chunk.
""")


if IN_NOTEBOOK:
    _banner()
