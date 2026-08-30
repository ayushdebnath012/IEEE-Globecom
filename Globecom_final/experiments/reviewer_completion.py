"""Reviewer-completion experiments for OmniMed-FL.

This module adds only experiments explicitly requested by the reviewers:

* operational federated comparison of eight lightweight fusion operators;
* a client-count x Dirichlet-heterogeneity systems grid;
* a fully-random-encoder initialization control;
* a matched one-shot hard-voting ensemble following the FedMME protocol; and
* a missing-text stress test following the published P-FIN equations.

Every invocation writes one atomic JSON record.  Separate processes can therefore
run independent keys without racing on a shared result store.  Records include
the exact base-model, runner, cache and helper hashes needed for provenance.

The FedMME and P-FIN experiments are matched *method-level adaptations*, not
claims of exact reproduction: the common corpus, DistilBERT/ViT encoders and
24-local-epoch budget are deliberately held fixed for a fair comparison.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

import omnimed_experiments as ox


FUSION_TYPES = [
    "concat",
    "attention",
    "gated",
    "clip",
    "flamingo",
    "blip2",
    "coca",
    "unified_io",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf8")
    tmp.replace(path)


def set_reproducibility(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs and optionally request deterministic CUDA kernels.

    The historical reviewer suite did not force deterministic algorithms.  New
    matched runs therefore default to the same setting, while recording it in
    every result.  ``--deterministic`` is available for a clean rerun suite.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


class ReviewerFusionModel(nn.Module):
    """The eight lightweight fusion rules with a distinct decoder-B operator.

    The historical registry used the same one-layer decoder implementation for
    two labels.  This wrapper preserves seven operators and makes decoder B a
    genuinely distinct two-query, two-layer decoder.  Names in the paper remain
    descriptive (Decoder A/B, token encoder, and so on); they are not presented
    as implementations of the similarly named foundation models.
    """

    FUSION_DIM = {
        "concat": 768,
        "attention": 256,
        "gated": 256,
        "clip": 512,
        "flamingo": 256,
        "blip2": 256,
        "coca": 384,
        "unified_io": 256,
    }

    def __init__(
        self,
        mf,
        text_model_name: str,
        vision_model_name: str,
        num_labels: int = 5,
        text_hidden: int = 256,
        vision_hidden: int = 512,
        fusion_type: str = "concat",
        dropout: float = 0.1,
        use_pretrained: bool = True,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.text_hidden = text_hidden
        self.vision_hidden = vision_hidden
        self.text_encoder = mf.LightweightTextClassifier(
            text_model_name, num_labels, text_hidden, dropout, use_pretrained
        )
        self.vision_encoder = mf.LightweightVisionClassifier(
            vision_model_name, num_labels, vision_hidden, dropout, use_pretrained
        )
        for head in (self.text_encoder.classifier, self.vision_encoder.classifier):
            for parameter in head.parameters():
                parameter.requires_grad = False
        self._build_fusion(fusion_type, text_hidden, vision_hidden, dropout)
        fusion_out = self.FUSION_DIM[fusion_type]
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_out, fusion_out // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_out // 2, num_labels),
        )

    def _build_fusion(self, kind: str, td: int, vd: int, dropout: float) -> None:
        if kind == "concat":
            self.proj_t = nn.Linear(td, 384)
            self.proj_v = nn.Linear(vd, 384)
        elif kind == "attention":
            self.proj_t = nn.Linear(td, 256)
            self.proj_v = nn.Linear(vd, 256)
            self.attn = nn.MultiheadAttention(256, 4, dropout=dropout, batch_first=True)
        elif kind == "gated":
            self.proj_t = nn.Linear(td, 256)
            self.proj_v = nn.Linear(vd, 256)
            self.gate = nn.Sequential(nn.Linear(512, 256), nn.Sigmoid())
        elif kind == "clip":
            self.proj_t = nn.Linear(td, 256)
            self.proj_v = nn.Linear(vd, 256)
        elif kind == "flamingo":
            self.proj_t = nn.Linear(td, 256)
            self.proj_v = nn.Linear(vd, 256)
            self.decoder_a = nn.TransformerDecoderLayer(
                256, nhead=4, dim_feedforward=512, dropout=dropout, batch_first=True
            )
        elif kind == "blip2":
            self.proj_t = nn.Linear(td, 256)
            self.proj_v = nn.Linear(vd, 256)
            self.query_tokens = nn.Parameter(torch.zeros(1, 2, 256))
            nn.init.normal_(self.query_tokens, std=0.02)
            layer = nn.TransformerDecoderLayer(
                256, nhead=4, dim_feedforward=512, dropout=dropout, batch_first=True
            )
            self.decoder_b = nn.TransformerDecoder(layer, num_layers=2)
        elif kind == "coca":
            self.proj_t = nn.Linear(td, 384)
            self.proj_v = nn.Linear(vd, 384)
            self.cross = nn.MultiheadAttention(384, 4, dropout=dropout, batch_first=True)
        elif kind == "unified_io":
            self.proj_t = nn.Linear(td, 256)
            self.proj_v = nn.Linear(vd, 256)
            layer = nn.TransformerEncoderLayer(
                256, nhead=4, dim_feedforward=512, dropout=dropout, batch_first=True
            )
            self.token_encoder = nn.TransformerEncoder(layer, num_layers=2)
        else:
            raise ValueError(f"unknown fusion operator: {kind}")

    def _fuse(self, ht: torch.Tensor, hv: torch.Tensor) -> torch.Tensor:
        ft, fv = self.proj_t(ht), self.proj_v(hv)
        if self.fusion_type == "concat":
            return torch.cat([ft, fv], dim=-1)
        if self.fusion_type == "attention":
            out, _ = self.attn(ft.unsqueeze(1), fv.unsqueeze(1), fv.unsqueeze(1))
            return ft + out.squeeze(1)
        if self.fusion_type == "gated":
            return ft + self.gate(torch.cat([ft, fv], dim=-1)) * fv
        if self.fusion_type == "clip":
            return torch.cat([ft, fv], dim=-1)
        if self.fusion_type == "flamingo":
            return self.decoder_a(ft.unsqueeze(1), fv.unsqueeze(1)).squeeze(1)
        if self.fusion_type == "blip2":
            queries = self.query_tokens.expand(ft.size(0), -1, -1) + ft.unsqueeze(1)
            memory = fv.unsqueeze(1)
            return self.decoder_b(queries, memory).mean(dim=1)
        if self.fusion_type == "coca":
            out, _ = self.cross(ft.unsqueeze(1), fv.unsqueeze(1), fv.unsqueeze(1))
            return out.squeeze(1)
        if self.fusion_type == "unified_io":
            return self.token_encoder(torch.stack([ft, fv], dim=1)).mean(dim=1)
        raise AssertionError("unreachable")

    def forward(self, input_ids, attention_mask=None, pixel_values=None, **kwargs):
        ht = self.text_encoder.get_text_features(input_ids, attention_mask)
        hv = self.vision_encoder.get_image_features(pixel_values)
        hidden = self._fuse(ht, hv)
        return {"logits": self.classifier(self.dropout(hidden)), "hidden": hidden}


class BoundReviewerFusionModel(ReviewerFusionModel):
    """Factory-compatible wrapper; ``mf`` is installed by ``bind_model``."""

    _mf = None

    def __init__(self, **kwargs):
        if self._mf is None:
            raise RuntimeError("BoundReviewerFusionModel.bind_model() was not called")
        super().__init__(self._mf, **kwargs)

    @classmethod
    def bind_model(cls, mf):
        cls._mf = mf
        return cls


class BoundRandomBackboneFusionModel(ReviewerFusionModel):
    """Same DistilBERT/ViT architecture as the public-start arm, randomized.

    Passing ``use_pretrained=False`` to the legacy base changes the text and
    image backbones to a small custom Transformer and residual CNN.  That would
    confound architecture with initialization.  This control instead constructs
    the same Hugging Face DistilBERT/ViT modules and then reinitializes only their
    weights from their native configurations; the projection, fusion, and task
    heads are already random under the common seed.
    """

    _mf = None

    def __init__(self, **kwargs):
        if self._mf is None:
            raise RuntimeError("BoundRandomBackboneFusionModel.bind_model() was not called")
        kwargs["use_pretrained"] = True
        super().__init__(self._mf, **kwargs)
        backbones = (
            self.text_encoder.transformer,
            self.vision_encoder.backbone,
        )
        if any(backbone is None for backbone in backbones):
            raise RuntimeError("random-backbone control requires DistilBERT and ViT modules")
        from transformers import AutoModel

        # Build fresh modules from the *same* checkpoint configurations.  This
        # is more reliable than calling init_weights() on loaded models because
        # recent Transformers releases mark checkpoint tensors as initialized
        # and deliberately skip them during later initialization passes.
        self.text_encoder.transformer = AutoModel.from_config(
            copy.deepcopy(backbones[0].config)
        )
        self.vision_encoder.backbone = AutoModel.from_config(
            copy.deepcopy(backbones[1].config)
        )

    @classmethod
    def bind_model(cls, mf):
        cls._mf = mf
        return cls


def model_kwargs(tier: dict, fusion: str = "concat", use_pretrained: bool = True) -> dict:
    return {
        "text_model_name": tier["text_model"],
        "vision_model_name": tier["vision_model"],
        "num_labels": 5,
        "fusion_type": fusion,
        "use_pretrained": use_pretrained,
    }


def setup(args):
    set_reproducibility(args.seed, args.deterministic)
    ox._patch_transformers()
    ox._patch_datasets()
    mf = ox.load_base(Path(args.base))
    ox._patch_image_loader(mf)
    tier = copy.deepcopy(ox.TIERS["standard"])
    cfg = mf.Config(
        batch_size=tier["batch_size"],
        max_samples_per_class=tier["samples_per_class"],
        fed_rounds=tier["fed_rounds"],
        local_epochs=tier["local_epochs"],
        epochs=tier["central_epochs"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("reviewer-completion experiments require a CUDA GPU")
    data = ox.build_data(mf, cfg, Path(args.cache))
    tokenizer = mf.get_text_tokenizer(tier["text_model"], cfg.max_seq_length)
    train_ds, val_ds, train_loader, val_loader = ox.make_mm_loaders(
        mf, data, cfg, tokenizer, balanced=True
    )
    BoundReviewerFusionModel.bind_model(mf)
    BoundRandomBackboneFusionModel.bind_model(mf)
    return mf, tier, cfg, device, data, tokenizer, train_ds, val_ds, train_loader, val_loader


def train_federated(args, mf, tier, cfg, device, train_ds, val_loader) -> dict:
    if args.task == "fusion":
        cls = BoundReviewerFusionModel
        kwargs = model_kwargs(tier, args.fusion, True)
    elif args.task == "init":
        cls = (
            BoundRandomBackboneFusionModel
            if args.init_mode == "random"
            else mf.MultiModalClassifier
        )
        kwargs = model_kwargs(tier, "concat", True)
    else:
        cls = mf.MultiModalClassifier
        kwargs = model_kwargs(tier, "concat", True)

    if args.task == "init" and args.init_mode == "pooled":
        _, _, pooled_loader, _ = ox.make_mm_loaders(
            mf, args._data, cfg, args._tokenizer, balanced=False
        )
        set_reproducibility(args.seed, args.deterministic)
        oracle = cls(**kwargs)
        _, warm_state = ox.centralized_train(
            mf,
            oracle,
            pooled_loader,
            val_loader,
            cfg,
            device,
            "multimodal",
            epochs=tier["central_epochs"],
            seed=args.seed,
            log_prefix=f"[oracle s={args.seed}] ",
        )
    else:
        warm_state = None

    result = ox.federated_train_ex(
        mf,
        cls,
        kwargs,
        train_ds,
        val_loader,
        cfg,
        device,
        "multimodal",
        alpha=args.alpha,
        num_clients=args.clients,
        rounds=tier["fed_rounds"],
        local_epochs=tier["local_epochs"],
        warm_start_state=warm_state,
        seed=args.seed,
        log_prefix=f"[{args.task} s={args.seed} a={args.alpha} K={args.clients}] ",
    )
    sizes = [int(value) for value in result.get("client_sizes", [])]
    # The audited core runner omits local updates for shards smaller than four
    # examples.  Preserve nominal K for its communication formula but expose the
    # number that actually trained so high-skew scalability plots cannot hide it.
    result["nominal_clients"] = int(args.clients)
    result["active_clients"] = sum(size >= 4 for size in sizes)
    result["skipped_client_ids"] = [
        index for index, size in enumerate(sizes) if size < 4
    ]
    return result


@torch.no_grad()
def predict_probabilities(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device),
        )
        probabilities.append(torch.softmax(out["logits"], dim=-1).cpu().numpy())
        labels.append(batch["labels"].argmax(dim=-1).cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def plurality_with_soft_ties(probability_sets: Sequence[np.ndarray]) -> np.ndarray:
    probs = np.stack(probability_sets, axis=0)
    hard = probs.argmax(axis=-1)
    output = np.empty(hard.shape[1], dtype=np.int64)
    mean_probs = probs.mean(axis=0)
    for index in range(hard.shape[1]):
        counts = np.bincount(hard[:, index], minlength=probs.shape[-1])
        candidates = np.flatnonzero(counts == counts.max())
        if len(candidates) == 1:
            output[index] = candidates[0]
        else:
            output[index] = candidates[np.argmax(mean_probs[index, candidates])]
    return output


def train_fedmme(args, mf, tier, cfg, device, train_ds, val_loader) -> dict:
    """Matched one-shot hard-voting ensemble following the FedMME protocol."""

    set_reproducibility(args.seed, args.deterministic)
    splits = mf.split_data_non_iid(train_ds, args.clients, args.alpha)
    states: List[dict] = []
    per_client_f1: List[float] = []
    per_client_seconds: List[float] = []
    peak_mib = 0.0
    kwargs = model_kwargs(tier, "concat", True)
    trainable_bytes = None
    matched_epochs = tier["fed_rounds"] * tier["local_epochs"]
    total_epochs = args.fedmme_epochs if args.fedmme_epochs is not None else matched_epochs
    if total_epochs < 1:
        raise ValueError(f"--fedmme-epochs must be positive, got {total_epochs}")

    for client_id, indices in enumerate(splits):
        set_reproducibility(args.seed * 1000 + client_id, args.deterministic)
        local_ds = Subset(train_ds, indices)
        local_labels = [train_ds[i]["labels"].argmax().item() for i in indices]
        loader = mf.create_balanced_dataloader(local_ds, local_labels, cfg.batch_size, 5)
        model = mf.MultiModalClassifier(**kwargs).to(device)
        if trainable_bytes is None:
            trainable_bytes = ox.trainable_bytes(model, dtype_bytes=4)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        criterion = mf.CombinedLoss(num_classes=5, diversity_weight=1.0)
        with ox.GPUMem() as meter:
            model.train()
            for _ in range(total_epochs):
                for batch in loader:
                    optimizer.zero_grad()
                    out = ox._forward(model, batch, device, "multimodal")
                    loss = criterion(out["logits"], batch["labels"].to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
        probabilities, labels = predict_probabilities(model, val_loader, device)
        pred = probabilities.argmax(axis=1)
        per_client_f1.append(float(f1_score(labels, pred, average="macro", zero_division=0)))
        states.append(mf.clone_state_dict_to_cpu(model))
        per_client_seconds.append(float(meter.seconds))
        peak_mib = max(peak_mib, float(meter.peak_mib))
        del model, optimizer, criterion, loader, local_ds
        torch.cuda.empty_cache()
        gc.collect()

    probability_sets: List[np.ndarray] = []
    labels = None
    for state in states:
        model = mf.MultiModalClassifier(**kwargs).to(device)
        model.load_state_dict(state)
        probs, labels = predict_probabilities(model, val_loader, device)
        probability_sets.append(probs)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    ensemble = plurality_with_soft_ties(probability_sets)
    macro_f1 = float(f1_score(labels, ensemble, average="macro", zero_division=0))
    accuracy = float(accuracy_score(labels, ensemble))
    diversity = float(len(np.unique(ensemble)) / 5.0)
    payload = int(trainable_bytes or 0)
    return {
        "f1": macro_f1,
        "accuracy": accuracy,
        "diversity": diversity,
        "per_client_f1": per_client_f1,
        "client_sizes": [len(x) for x in splits],
        "local_epochs": total_epochs,
        "matched_budget_epochs": matched_epochs,
        "budget": ("native" if args.fedmme_epochs is not None else "matched"),
        "ensemble": "equal hard plurality; mean-softmax tie break",
        "upload_bytes_per_client": payload,
        "total_comm_bytes_bidirectional": 2 * args.clients * payload,
        "total_comm_bytes_upload_only": args.clients * payload,
        "wall_seconds": float(sum(per_client_seconds)),
        "per_client_seconds": per_client_seconds,
        "peak_mib": peak_mib,
    }


def provenance(args, base: Path, cache: Path, result: dict) -> dict:
    script = Path(__file__).resolve()
    helper = script.with_name("pfin_matched.py")
    if args.task == "fedmme":
        schedule = {
            "rounds": 1,
            "local_epochs": int(result.get("local_epochs", 24)),
            "communication_schedule": "one upload per independently trained client model",
        }
    else:
        schedule = {
            "rounds": 8,
            "local_epochs": 3,
            "communication_schedule": "one download and upload per client per round",
        }
    return {
        "task": args.task,
        "key": result_key(args),
        "parameters": {
            "seed": args.seed,
            "alpha": args.alpha,
            "clients": args.clients,
            "fusion": args.fusion if args.task == "fusion" else None,
            "init_mode": args.init_mode if args.task == "init" else None,
            "pfin_mode": args.pfin_mode if args.task == "pfin" else None,
            "deterministic": args.deterministic,
        },
        "protocol": {
            **schedule,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "precision": "FP32",
            "initialization": (
                "same DistilBERT/ViT configurations with random encoder weights and random heads"
                if args.task == "init" and args.init_mode == "random"
                else "public encoders + random task heads"
            ),
            "data": "controlled_v2_real4_synthetic_covid_template_text",
        },
        "provenance": {
            "base_model_sha256": sha256_file(base),
            "core_runner_sha256": sha256_file(Path(ox.__file__).resolve()),
            "reviewer_runner_sha256": sha256_file(script),
            "pfin_helper_sha256": sha256_file(helper) if helper.exists() else None,
            "data_cache_sha256": sha256_file(cache),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "deterministic_algorithms_enforced": torch.are_deterministic_algorithms_enabled(),
        },
        "result": result,
    }


def result_key(args) -> str:
    if args.task == "fusion":
        return f"fusion={args.fusion}|alpha={args.alpha}|K={args.clients}|seed={args.seed}"
    if args.task == "grid":
        return f"alpha={args.alpha}|K={args.clients}|seed={args.seed}"
    if args.task == "init":
        return f"init={args.init_mode}|alpha={args.alpha}|K={args.clients}|seed={args.seed}"
    if args.task == "fedmme":
        # The matched-budget arm keeps its original key so previously validated
        # records stay byte-identical; a native-budget run gets a distinct key.
        if args.fedmme_epochs is None:
            return f"fedmme_style|alpha={args.alpha}|K={args.clients}|seed={args.seed}"
        return (
            f"fedmme_native|epochs={args.fedmme_epochs}|alpha={args.alpha}"
            f"|K={args.clients}|seed={args.seed}"
        )
    if args.task == "pfin":
        return f"pfin={args.pfin_mode}|alpha={args.alpha}|K={args.clients}|seed={args.seed}"
    raise ValueError(args.task)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["fusion", "grid", "init", "fedmme", "pfin"])
    parser.add_argument("--base", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--fusion", choices=FUSION_TYPES, default="concat")
    parser.add_argument("--init-mode", choices=["public", "random", "pooled"], default="public")
    parser.add_argument(
        "--pfin-mode",
        choices=["zero", "deterministic", "probabilistic", "probabilistic_uq"],
        default="probabilistic_uq",
    )
    parser.add_argument(
        "--fedmme-epochs",
        type=int,
        default=None,
        help=(
            "local epochs for the FedMME arm; omit for the matched "
            "fed_rounds*local_epochs budget, or pass 100 for the source paper's "
            "native client budget"
        ),
    )
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base, cache, out = Path(args.base), Path(args.cache), Path(args.out)
    (
        mf,
        tier,
        cfg,
        device,
        data,
        tokenizer,
        train_ds,
        val_ds,
        train_loader,
        val_loader,
    ) = setup(args)
    # Used only by the pooled-initialization arm without expanding the public API.
    args._data, args._tokenizer = data, tokenizer

    if args.task in {"fusion", "grid", "init"}:
        result = train_federated(args, mf, tier, cfg, device, train_ds, val_loader)
    elif args.task == "fedmme":
        result = train_fedmme(args, mf, tier, cfg, device, train_ds, val_loader)
    else:
        from pfin_matched import run_pfin_federated

        result = run_pfin_federated(
            mf=mf,
            ox=ox,
            tier=tier,
            cfg=cfg,
            device=device,
            train_ds=train_ds,
            val_loader=val_loader,
            alpha=args.alpha,
            num_clients=args.clients,
            rounds=tier["fed_rounds"],
            local_epochs=tier["local_epochs"],
            seed=args.seed,
            mode=args.pfin_mode,
        )

    payload = provenance(args, base, cache, result)
    atomic_json(out, payload)
    summary = {"key": payload["key"], "f1": result.get("f1"), "out": str(out)}
    if args.task == "pfin":
        summary["forced_missing_text_f1"] = result.get("forced_missing_text_f1")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
