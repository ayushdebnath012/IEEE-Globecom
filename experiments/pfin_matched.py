"""P-FIN-style components for the OmniMed matched comparison.

This module implements the feature-level components described by Shahid et al.
for a *matched adaptation* to the OmniMed image--text experiment.  It is not an
exact reproduction of the authors' full training pipeline: image and text
encoders, classifiers, client data loaders, and the federated training loop are
deliberately supplied by the caller.

The probabilistic imputer models a 256-dimensional text feature conditioned on
a 256-dimensional image feature with a heteroscedastic diagonal Gaussian.  A
deterministic FIN with the same Transformer trunk is provided for the matched
ablation.  Server aggregation weights follow the published Fed-UQ-Avg equation
without repository-specific boosts, floors, or warm-up schedules.
"""

from __future__ import annotations

import copy
import gc
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


DEFAULT_FEATURE_DIM = 256
DEFAULT_BETA = 0.5
DEFAULT_RHO = 0.6
DEFAULT_TEMPERATURE = 0.2


@dataclass
class FINOutput:
    """Output shared by the probabilistic and deterministic FIN variants.

    ``features`` is the representation intended for downstream fusion.  For a
    :class:`GaussianPFIN`, it is the uncertainty-gated mean by default.  For a
    :class:`DeterministicFIN`, it is the point estimate itself.

    ``logvar`` and ``variance`` are ``None`` for the deterministic ablation.
    ``sample_uncertainty`` is the mean predicted variance over feature
    dimensions for each sample and is zero for the deterministic ablation.
    """

    features: Tensor
    mean: Tensor
    logvar: Optional[Tensor]
    variance: Optional[Tensor]
    gate: Tensor
    sample_uncertainty: Tensor

    @property
    def mean_uncertainty(self) -> Tensor:
        """Return the scalar mean uncertainty for a client batch."""

        return self.sample_uncertainty.mean()


class _FeatureTransformer(nn.Module):
    """Shared two-token Transformer trunk used by both matched FIN variants."""

    def __init__(
        self,
        input_dim: int = DEFAULT_FEATURE_DIM,
        output_dim: int = DEFAULT_FEATURE_DIM,
        hidden_dim: int = DEFAULT_FEATURE_DIM,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide hidden_dim")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.query_token = nn.Parameter(torch.empty(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        nn.init.normal_(self.query_token, mean=0.0, std=0.02)

    def _encode(self, image_features: Tensor) -> Tensor:
        if image_features.ndim != 2:
            raise ValueError(
                "image_features must have shape [batch, feature_dim], got "
                f"{tuple(image_features.shape)}"
            )
        if image_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} image features, got "
                f"{image_features.shape[-1]}"
            )

        projected = self.input_projection(image_features).unsqueeze(1)
        query = self.query_token.expand(image_features.shape[0], -1, -1)
        encoded = self.transformer(torch.cat((query, projected), dim=1))
        return encoded[:, 0, :]


class _OutputMLP(nn.Sequential):
    """Two-layer output head used for the mean and log-variance estimates."""

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )


class GaussianPFIN(_FeatureTransformer):
    """Gaussian image-to-text P-FIN for the paper-faithful matched adaptation.

    The architecture is a two-layer, four-head Transformer by default.  Its
    query-token representation feeds separate MLP heads for the conditional
    mean and log variance.  The downstream text feature is gated as
    ``mean * sigmoid(-logvar)``, matching the published uncertainty gate
    ``sigmoid(-log(sigma^2))``.

    The caller should first project and L2-normalize image and target text
    features to ``input_dim`` and ``output_dim`` respectively (256 by default).
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_FEATURE_DIM,
        output_dim: int = DEFAULT_FEATURE_DIM,
        hidden_dim: int = DEFAULT_FEATURE_DIM,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        min_logvar: float = -10.0,
        max_logvar: float = 10.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        if min_logvar >= max_logvar:
            raise ValueError("min_logvar must be smaller than max_logvar")
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.mean_head = _OutputMLP(hidden_dim, output_dim)
        self.logvar_head = _OutputMLP(hidden_dim, output_dim)

    def forward(self, image_features: Tensor, apply_gate: bool = True) -> FINOutput:
        """Impute text features and return Gaussian uncertainty diagnostics."""

        hidden = self._encode(image_features)
        mean = self.mean_head(hidden)
        logvar = self.logvar_head(hidden).clamp(
            min=self.min_logvar,
            max=self.max_logvar,
        )
        variance = logvar.exp()
        gate = uncertainty_gate(logvar)
        features = mean * gate if apply_gate else mean
        sample_uncertainty = variance.mean(dim=-1)
        return FINOutput(
            features=features,
            mean=mean,
            logvar=logvar,
            variance=variance,
            gate=gate,
            sample_uncertainty=sample_uncertainty,
        )

    def impute(self, image_features: Tensor, gated: bool = True) -> Tensor:
        """Return only the imputed feature used by a fusion classifier."""

        return self.forward(image_features, apply_gate=gated).features


class DeterministicFIN(_FeatureTransformer):
    """Point-estimate FIN ablation for the P-FIN matched comparison.

    This variant deliberately shares the probabilistic model's input
    projection, query token, two-layer/four-head Transformer, and mean head.
    It removes only uncertainty estimation and gating, so MSE-trained results
    isolate the contribution of the Gaussian P-FIN components.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_FEATURE_DIM,
        output_dim: int = DEFAULT_FEATURE_DIM,
        hidden_dim: int = DEFAULT_FEATURE_DIM,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.mean_head = _OutputMLP(hidden_dim, output_dim)

    def forward(self, image_features: Tensor, apply_gate: bool = True) -> FINOutput:
        """Return a deterministic text-feature estimate.

        ``apply_gate`` is accepted for API compatibility and has no effect.
        """

        del apply_gate
        mean = self.mean_head(self._encode(image_features))
        gate = torch.ones_like(mean)
        sample_uncertainty = torch.zeros(
            mean.shape[0],
            dtype=mean.dtype,
            device=mean.device,
        )
        return FINOutput(
            features=mean,
            mean=mean,
            logvar=None,
            variance=None,
            gate=gate,
            sample_uncertainty=sample_uncertainty,
        )

    def impute(self, image_features: Tensor, gated: bool = True) -> Tensor:
        """Return the deterministic imputed text feature."""

        return self.forward(image_features, apply_gate=gated).features


def uncertainty_gate(logvar: Tensor) -> Tensor:
    """Compute the paper's feature-wise gate ``sigmoid(-log(sigma^2))``."""

    if not torch.is_floating_point(logvar):
        raise TypeError("logvar must be a floating-point tensor")
    return torch.sigmoid(-logvar)


def beta_nll_loss(
    mean: Tensor,
    logvar: Tensor,
    target: Tensor,
    beta: float = DEFAULT_BETA,
    reduction: str = "mean",
) -> Tensor:
    """Return heteroscedastic Gaussian beta-NLL with stopped beta weights.

    For each feature dimension this computes

    ``stopgrad(variance**beta) * 0.5 * (logvar + error**2 / variance)``.

    ``beta=0.5`` is the matched-adaptation default reported by P-FIN.  With
    ``reduction='none'`` the result has the same shape as ``mean``.
    """

    if mean.shape != logvar.shape or mean.shape != target.shape:
        raise ValueError(
            "mean, logvar, and target must have identical shapes; got "
            f"{tuple(mean.shape)}, {tuple(logvar.shape)}, {tuple(target.shape)}"
        )
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    # Clamp only inside the loss so externally reported model uncertainty is
    # unchanged.  The model already bounds its default output to this interval.
    safe_logvar = logvar.clamp(min=-10.0, max=10.0)
    variance = safe_logvar.exp()
    gaussian_nll = 0.5 * (
        safe_logvar + (target - mean).square() / variance
    )
    beta_weight = variance.detach().pow(beta)
    loss = beta_weight * gaussian_nll

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def imputation_loss(
    output: FINOutput,
    target: Tensor,
    beta: float = DEFAULT_BETA,
    reduction: str = "mean",
) -> Tensor:
    """Dispatch to beta-NLL or MSE for matched probabilistic/deterministic FINs."""

    if output.mean.shape != target.shape:
        raise ValueError(
            "output mean and target must have identical shapes; got "
            f"{tuple(output.mean.shape)} and {tuple(target.shape)}"
        )
    if output.logvar is not None:
        return beta_nll_loss(
            output.mean,
            output.logvar,
            target,
            beta=beta,
            reduction=reduction,
        )
    return F.mse_loss(output.mean, target, reduction=reduction)


ArrayLike = Union[Sequence[float], Tensor]


def fed_uq_weights(
    client_sizes: ArrayLike,
    mean_uncertainties: ArrayLike,
    rho: float = DEFAULT_RHO,
    temperature: float = DEFAULT_TEMPERATURE,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Compute paper-faithful Fed-UQ-Avg client weights.

    The returned weights implement

    ``W_k = (1-rho) * n_k/sum(n) + rho * softmax(-u_k/T)``,

    where ``u_k`` is client ``k``'s mean imputation variance.  The matched
    defaults are ``rho=0.6`` and ``T=0.2``.  No multimodal-client boost,
    minimum-weight floor, or warm-up schedule is applied.
    """

    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")

    if isinstance(mean_uncertainties, Tensor):
        device = mean_uncertainties.device
    elif isinstance(client_sizes, Tensor):
        device = client_sizes.device
    else:
        device = torch.device("cpu")

    sizes = torch.as_tensor(client_sizes, dtype=dtype, device=device).flatten()
    uncertainties = torch.as_tensor(
        mean_uncertainties,
        dtype=dtype,
        device=device,
    ).flatten()

    if sizes.numel() == 0:
        raise ValueError("at least one client is required")
    if sizes.shape != uncertainties.shape:
        raise ValueError(
            "client_sizes and mean_uncertainties must have the same length"
        )
    if not torch.isfinite(sizes).all() or not torch.isfinite(uncertainties).all():
        raise ValueError("sizes and uncertainties must be finite")
    if (sizes <= 0).any():
        raise ValueError("every participating client must have a positive size")
    if (uncertainties < 0).any():
        raise ValueError("mean uncertainties must be non-negative variances")

    data_weights = sizes / sizes.sum()
    confidence_weights = torch.softmax(-uncertainties / temperature, dim=0)
    weights = (1.0 - rho) * data_weights + rho * confidence_weights
    # Floating-point roundoff is the only reason this final normalization can
    # differ from one; retaining it makes state-dict aggregation exact.
    return weights / weights.sum()


def build_fin(mode: str = "gaussian", **kwargs: object) -> nn.Module:
    """Construct a matched P-FIN variant by name.

    Accepted probabilistic names are ``'gaussian'`` and ``'pfin'``; accepted
    point-estimate names are ``'deterministic'`` and ``'fin'``.
    """

    normalized = mode.strip().lower().replace("-", "_")
    if normalized in {"gaussian", "pfin", "probabilistic"}:
        return GaussianPFIN(**kwargs)
    if normalized in {"deterministic", "fin", "point"}:
        return DeterministicFIN(**kwargs)
    raise ValueError(
        "unknown FIN mode; expected gaussian/pfin or deterministic/fin, got "
        f"{mode!r}"
    )


_RUN_MODES = {
    "zero",
    "deterministic",
    "probabilistic",
    "probabilistic_uq",
}

_METHOD_LABELS = {
    "zero": "Zero-fill + FedAvg (matched missing-text control)",
    "deterministic": "Deterministic FIN + FedAvg (matched adaptation)",
    "probabilistic": "Gaussian P-FIN-style + FedAvg (matched adaptation, concat fusion)",
    "probabilistic_uq": (
        "Gaussian P-FIN-style + Fed-UQ-Avg (matched adaptation, concat fusion)"
    ),
}


class _MatchedPFINClassifier(nn.Module):
    """Image--text classifier used only by :func:`run_pfin_federated`.

    The public pretrained encoders are provided by the OmniMed base module.
    Their requested 256-dimensional outputs are L2-normalized before either
    concatenation or image-to-text imputation.  The model therefore adapts the
    P-FIN feature-level method to the same encoder family and five-class task.
    The common OmniMed concatenation classifier replaces the original method's
    bidirectional cross-modal attention, so this is explicitly a P-FIN-style
    matched adaptation rather than an exact reproduction.
    """

    def __init__(
        self,
        mf: object,
        text_model_name: str,
        vision_model_name: str,
        mode: str,
        num_labels: int = 5,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in _RUN_MODES:
            raise ValueError(f"unsupported matched P-FIN mode: {mode!r}")

        self.mode = mode
        self.num_labels = num_labels
        self.feature_dim = feature_dim
        self.text_encoder = mf.LightweightTextClassifier(
            model_name=text_model_name,
            num_labels=num_labels,
            hidden_dim=feature_dim,
            dropout=dropout,
            use_pretrained=True,
        )
        self.vision_encoder = mf.LightweightVisionClassifier(
            model_name=vision_model_name,
            num_labels=num_labels,
            hidden_dim=feature_dim,
            dropout=dropout,
            use_pretrained=True,
        )

        # These task heads are bypassed; only the public encoders and their
        # 256-dimensional projections feed the matched fusion classifier.
        for unused_head in (
            self.text_encoder.classifier,
            self.vision_encoder.classifier,
        ):
            for parameter in unused_head.parameters():
                parameter.requires_grad = False

        # Construct the common classification path before the method-specific
        # FIN.  Consequently all arms receive identical encoder/classifier
        # initialization when run with the same seed.
        self.classifier = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_labels),
        )

        if mode == "zero":
            self.fin: Optional[nn.Module] = None
        elif mode == "deterministic":
            self.fin = DeterministicFIN(
                input_dim=feature_dim,
                output_dim=feature_dim,
                hidden_dim=feature_dim,
                num_layers=2,
                num_heads=4,
                dropout=dropout,
            )
        else:
            self.fin = GaussianPFIN(
                input_dim=feature_dim,
                output_dim=feature_dim,
                hidden_dim=feature_dim,
                num_layers=2,
                num_heads=4,
                dropout=dropout,
            )

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return a 256-dimensional L2-normalized observed-text feature."""

        feature = self.text_encoder.get_text_features(input_ids, attention_mask)
        return F.normalize(feature, p=2, dim=-1)

    def encode_image(self, pixel_values: Tensor) -> Tensor:
        """Return a 256-dimensional L2-normalized image feature."""

        feature = self.vision_encoder.get_image_features(pixel_values)
        return F.normalize(feature, p=2, dim=-1)

    def classify_features(self, image_feature: Tensor, text_feature: Tensor) -> Tensor:
        """Classify a concatenated pair of aligned 256-dimensional features."""

        return self.classifier(torch.cat((image_feature, text_feature), dim=-1))

    def missing_text_feature(
        self,
        image_feature: Tensor,
        detach_imputer: bool = False,
    ) -> tuple[Tensor, Optional[FINOutput]]:
        """Return zero-filled or FIN-imputed text for an image-only client.

        During image-only local training the imputer is intentionally detached:
        only clients possessing paired image--text observations optimize its
        reconstruction objective.  The direct image branch remains trainable.
        """

        if self.fin is None:
            return torch.zeros_like(image_feature), None
        if detach_imputer:
            with torch.no_grad():
                fin_output = self.fin(image_feature.detach())
            return fin_output.features.detach(), fin_output
        fin_output = self.fin(image_feature)
        return fin_output.features, fin_output

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        force_missing_text: bool = False,
        detach_imputer: bool = False,
    ) -> Dict[str, object]:
        """Run full-text or forced-missing-text classification."""

        if pixel_values is None:
            raise ValueError("pixel_values are required")
        image_feature = self.encode_image(pixel_values)
        fin_output: Optional[FINOutput] = None
        if force_missing_text:
            text_feature, fin_output = self.missing_text_feature(
                image_feature,
                detach_imputer=detach_imputer,
            )
        else:
            if input_ids is None or attention_mask is None:
                raise ValueError(
                    "input_ids and attention_mask are required when text is observed"
                )
            text_feature = self.encode_text(input_ids, attention_mask)

        return {
            "logits": self.classify_features(image_feature, text_feature),
            "image_features": image_feature,
            "text_features": text_feature,
            "fin_output": fin_output,
        }


def _deterministic_client_mask(num_clients: int, seed: int) -> List[bool]:
    """Choose exactly three multimodal clients without changing global RNG state."""

    if num_clients != 5:
        raise ValueError(
            "the matched P-FIN stress test is defined for K=5 clients "
            "(3 multimodal and 2 image-only)"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(seed) + 0x5046494E) % (2**63 - 1))
    selected = set(torch.randperm(num_clients, generator=generator)[:3].tolist())
    return [client_id in selected for client_id in range(num_clients)]


def _classification_metrics(
    predictions: Tensor,
    labels: Tensor,
    num_labels: int,
) -> Dict[str, float]:
    """Compute accuracy, macro F1, and predicted-class diversity."""

    predictions = predictions.to(dtype=torch.long, device="cpu").flatten()
    labels = labels.to(dtype=torch.long, device="cpu").flatten()
    if predictions.numel() == 0 or predictions.shape != labels.shape:
        return {"f1": 0.0, "accuracy": 0.0, "diversity": 0.0}

    encoded = labels * num_labels + predictions
    confusion = torch.bincount(
        encoded,
        minlength=num_labels * num_labels,
    ).reshape(num_labels, num_labels).to(torch.float64)
    true_positive = confusion.diag()
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    class_f1 = torch.where(
        denominator > 0,
        2.0 * true_positive / denominator,
        torch.zeros_like(denominator),
    )
    return {
        "f1": float(class_f1.mean().item()),
        "accuracy": float(true_positive.sum().item() / predictions.numel()),
        "diversity": float(predictions.unique().numel() / num_labels),
    }


def _evaluate_matched_model(
    model: _MatchedPFINClassifier,
    dataloader: DataLoader,
    device: torch.device,
    force_missing_text: bool,
) -> Dict[str, float]:
    """Evaluate one global model without validation-time parameter selection."""

    model.eval()
    predictions: List[Tensor] = []
    labels: List[Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device),
                force_missing_text=force_missing_text,
            )
            predictions.append(output["logits"].argmax(dim=-1).cpu())
            labels.append(batch["labels"].argmax(dim=-1).cpu())

    if not predictions:
        return {"f1": 0.0, "accuracy": 0.0, "diversity": 0.0}
    return _classification_metrics(
        torch.cat(predictions),
        torch.cat(labels),
        model.num_labels,
    )


def _mean_client_imputation_variance(
    model: _MatchedPFINClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Measure a client's mean Gaussian imputation variance after local training."""

    if not isinstance(model.fin, GaussianPFIN):
        return 0.0
    model.eval()
    variance_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in dataloader:
            image_feature = model.encode_image(batch["pixel_values"].to(device))
            fin_output = model.fin(image_feature)
            batch_count = int(fin_output.sample_uncertainty.numel())
            variance_sum += float(fin_output.sample_uncertainty.sum().item())
            sample_count += batch_count
    return variance_sum / max(sample_count, 1)


def _sample_size_weights(client_sizes: Sequence[int]) -> Tensor:
    """Return standard FedAvg data-proportional client weights."""

    sizes = torch.as_tensor(client_sizes, dtype=torch.float64)
    if sizes.numel() == 0 or (sizes <= 0).any():
        raise ValueError("every matched P-FIN client must have at least one sample")
    return sizes / sizes.sum()


def _aggregate_trainable_state(
    global_model: nn.Module,
    local_states: Sequence[Dict[str, Tensor]],
    weights: Tensor,
    trainable_keys: Sequence[str],
    fin_weights: Optional[Tensor] = None,
) -> None:
    """Aggregate trainable state, using paired-client weights for FIN tensors."""

    if len(local_states) != int(weights.numel()):
        raise ValueError("one aggregation weight is required for every client state")
    if fin_weights is not None and len(local_states) != int(fin_weights.numel()):
        raise ValueError("one FIN aggregation weight is required for every client state")
    global_state = global_model.state_dict()
    with torch.no_grad():
        for key in trainable_keys:
            target = global_state[key]
            aggregate = torch.zeros_like(target, dtype=torch.float32, device="cpu")
            key_weights = fin_weights if key.startswith("fin.") and fin_weights is not None else weights
            for local_state, weight in zip(local_states, key_weights.tolist()):
                aggregate.add_(local_state[key], alpha=float(weight))
            target.copy_(aggregate.to(device=target.device, dtype=target.dtype))


def run_pfin_federated(
    mf: object,
    ox: object,
    tier: dict,
    cfg: object,
    device: Union[str, torch.device],
    train_ds: object,
    val_loader: DataLoader,
    alpha: float,
    num_clients: int,
    rounds: int,
    local_epochs: int,
    seed: int,
    mode: str,
) -> Dict[str, object]:
    """Run the K=5 client-level missing-text matched P-FIN stress test.

    This is a P-FIN-style *matched adaptation*, not an exact reproduction of
    P-FIN's original datasets, backbones, bidirectional fusion, or multi-label
    task. Every arm uses
    the same OmniMed samples, Dirichlet shards, public encoders, 256-dimensional
    L2 projections, classifier, optimizer settings, local budget, client mask,
    and seed.  Exactly three clients retain paired image--text observations and
    two are treated as image-only for the entire run.

    Supported ``mode`` values are:

    ``zero``
        Zero-filled text features with sample-size FedAvg.
    ``deterministic``
        MSE-trained deterministic FIN with sample-size FedAvg.
    ``probabilistic``
        Gaussian beta-NLL P-FIN with sample-size FedAvg.
    ``probabilistic_uq``
        The same Gaussian P-FIN with published Fed-UQ-Avg weights
        (rho=0.6, T=0.2).

    Full-text and forced-missing-text validation are both reported.  The
    communication estimate follows the rest of the OmniMed suite: one FP32
    trainable-state download and upload per participating client and round,
    plus one FP32 uncertainty scalar for Fed-UQ-Avg.
    """

    normalized_mode = mode.strip().lower().replace("-", "_")
    if normalized_mode not in _RUN_MODES:
        raise ValueError(
            f"unsupported mode {mode!r}; expected one of {sorted(_RUN_MODES)}"
        )
    if num_clients != 5:
        raise ValueError("run_pfin_federated requires num_clients=5")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if rounds <= 0 or local_epochs <= 0:
        raise ValueError("rounds and local_epochs must be positive")
    if len(train_ds) == 0:
        raise ValueError("train_ds must not be empty")
    if "text_model" not in tier or "vision_model" not in tier:
        raise KeyError("tier must define text_model and vision_model")

    run_device = torch.device(device)
    ox.set_seed(int(seed))

    # Splitting occurs immediately after the seed reset and before constructing
    # a method-specific model.  Re-running any arm with this seed therefore
    # produces exactly the same shards.
    client_splits = mf.split_data_non_iid(train_ds, num_clients, alpha)
    client_sizes = [len(indices) for indices in client_splits]
    if any(size <= 0 for size in client_sizes):
        raise RuntimeError(
            "the realized Dirichlet split produced an empty client; choose a "
            "different seed rather than assigning an artificial sample"
        )
    multimodal_mask = _deterministic_client_mask(num_clients, int(seed))
    multimodal_clients = [i for i, available in enumerate(multimodal_mask) if available]
    image_only_clients = [i for i, available in enumerate(multimodal_mask) if not available]

    client_class_hist: List[List[int]] = []
    for indices in client_splits:
        histogram = [0] * 5
        for sample_index in indices:
            label = int(train_ds[sample_index]["labels"].argmax().item())
            histogram[label] += 1
        client_class_hist.append(histogram)

    global_model = _MatchedPFINClassifier(
        mf=mf,
        text_model_name=tier["text_model"],
        vision_model_name=tier["vision_model"],
        mode=normalized_mode,
        num_labels=5,
        feature_dim=DEFAULT_FEATURE_DIM,
        dropout=float(tier.get("pfin_dropout", 0.1)),
    ).to(run_device)

    trainable_keys = [
        name for name, parameter in global_model.named_parameters()
        if parameter.requires_grad and parameter.is_floating_point()
    ]
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in global_model.parameters()
        if parameter.requires_grad
    )
    total_parameter_count = sum(
        parameter.numel() for parameter in global_model.parameters()
    )
    payload_bytes = trainable_parameter_count * 4
    fin_loss_weight = float(tier.get("pfin_loss_weight", 1.0))
    if fin_loss_weight < 0.0:
        raise ValueError("pfin_loss_weight must be non-negative")

    history: Dict[str, List[object]] = {
        "round_full_text_f1": [],
        "round_forced_missing_text_f1": [],
        "round_full_text_accuracy": [],
        "round_forced_missing_text_accuracy": [],
        "round_full_text_diversity": [],
        "round_forced_missing_text_diversity": [],
        "client_mean_uncertainty": [],
        "aggregation_weights": [],
        "fin_aggregation_weights": [],
        "round_seconds": [],
        "round_peak_mib": [],
    }

    for round_index in range(rounds):
        local_states: List[Dict[str, Tensor]] = []
        client_uncertainties: List[float] = []

        with ox.GPUMem() as measurement:
            for client_id, indices in enumerate(client_splits):
                # The explicit reset keeps shards, balanced batches, and dropout
                # streams reproducible when method arms are launched separately.
                local_seed = (
                    int(seed) * 1_000_003
                    + round_index * 10_007
                    + client_id * 101
                    + 17
                )
                ox.set_seed(local_seed)

                local_dataset = Subset(train_ds, indices)
                local_labels = [
                    int(train_ds[index]["labels"].argmax().item())
                    for index in indices
                ]
                train_loader = mf.create_balanced_dataloader(
                    local_dataset,
                    local_labels,
                    cfg.batch_size,
                    5,
                )
                uncertainty_loader = DataLoader(
                    local_dataset,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                )

                local_model = copy.deepcopy(global_model).to(run_device)
                local_model.train()
                is_multimodal = multimodal_mask[client_id]
                if not is_multimodal and local_model.fin is not None:
                    local_model.fin.eval()

                optimizer_parameters = [
                    parameter
                    for name, parameter in local_model.named_parameters()
                    if parameter.requires_grad
                    and (is_multimodal or not name.startswith("fin."))
                ]
                optimizer = torch.optim.AdamW(
                    optimizer_parameters,
                    lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                )

                for _ in range(local_epochs):
                    for batch in train_loader:
                        optimizer.zero_grad()
                        labels = batch["labels"].to(run_device).argmax(dim=-1)
                        if is_multimodal:
                            output = local_model(
                                input_ids=batch["input_ids"].to(run_device),
                                attention_mask=batch["attention_mask"].to(run_device),
                                pixel_values=batch["pixel_values"].to(run_device),
                                force_missing_text=False,
                            )
                            loss = F.cross_entropy(output["logits"], labels)
                            if local_model.fin is not None:
                                fin_output = local_model.fin(
                                    output["image_features"].detach()
                                )
                                reconstruction = imputation_loss(
                                    fin_output,
                                    output["text_features"].detach(),
                                    beta=DEFAULT_BETA,
                                )
                                loss = loss + fin_loss_weight * reconstruction
                        else:
                            output = local_model(
                                pixel_values=batch["pixel_values"].to(run_device),
                                force_missing_text=True,
                                detach_imputer=True,
                            )
                            loss = F.cross_entropy(output["logits"], labels)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(optimizer_parameters, 1.0)
                        optimizer.step()

                client_uncertainties.append(
                    _mean_client_imputation_variance(
                        local_model,
                        uncertainty_loader,
                        run_device,
                    )
                )
                local_state = {
                    key: local_model.state_dict()[key].detach().cpu().float().clone()
                    for key in trainable_keys
                }
                local_states.append(local_state)

                del optimizer, local_model, train_loader, uncertainty_loader, local_dataset
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            if normalized_mode == "probabilistic_uq":
                aggregation_weights = fed_uq_weights(
                    client_sizes,
                    client_uncertainties,
                    rho=DEFAULT_RHO,
                    temperature=DEFAULT_TEMPERATURE,
                )
            else:
                aggregation_weights = _sample_size_weights(client_sizes)

            # FIN is optimized only where paired image--text observations exist.
            # Renormalize its server weights over those clients so unchanged
            # image-only copies do not dilute the imputer update.
            multimodal_weight_mask = torch.as_tensor(
                multimodal_mask,
                dtype=aggregation_weights.dtype,
                device=aggregation_weights.device,
            )
            fin_aggregation_weights = aggregation_weights * multimodal_weight_mask
            fin_weight_sum = fin_aggregation_weights.sum()
            if fin_weight_sum <= 0:
                raise RuntimeError("at least one multimodal client is required")
            fin_aggregation_weights = fin_aggregation_weights / fin_weight_sum

            _aggregate_trainable_state(
                global_model,
                local_states,
                aggregation_weights,
                trainable_keys,
                fin_weights=fin_aggregation_weights,
            )
            del local_states
            gc.collect()

        full_text_metrics = _evaluate_matched_model(
            global_model,
            val_loader,
            run_device,
            force_missing_text=False,
        )
        forced_missing_metrics = _evaluate_matched_model(
            global_model,
            val_loader,
            run_device,
            force_missing_text=True,
        )

        history["round_full_text_f1"].append(full_text_metrics["f1"])
        history["round_forced_missing_text_f1"].append(
            forced_missing_metrics["f1"]
        )
        history["round_full_text_accuracy"].append(full_text_metrics["accuracy"])
        history["round_forced_missing_text_accuracy"].append(
            forced_missing_metrics["accuracy"]
        )
        history["round_full_text_diversity"].append(full_text_metrics["diversity"])
        history["round_forced_missing_text_diversity"].append(
            forced_missing_metrics["diversity"]
        )
        history["client_mean_uncertainty"].append(client_uncertainties)
        history["aggregation_weights"].append(
            [float(weight) for weight in aggregation_weights.tolist()]
        )
        history["fin_aggregation_weights"].append(
            [float(weight) for weight in fin_aggregation_weights.tolist()]
        )
        history["round_seconds"].append(float(measurement.seconds))
        history["round_peak_mib"].append(float(measurement.peak_mib))

        print(
            f"    [P-FIN matched {normalized_mode}] round "
            f"{round_index + 1}/{rounds} full-F1={full_text_metrics['f1']:.4f} "
            f"missing-F1={forced_missing_metrics['f1']:.4f} "
            f"{measurement.seconds:.1f}s {measurement.peak_mib:.0f}MiB"
        )

    final_full = _evaluate_matched_model(
        global_model,
        val_loader,
        run_device,
        force_missing_text=False,
    )
    final_missing = _evaluate_matched_model(
        global_model,
        val_loader,
        run_device,
        force_missing_text=True,
    )

    uq_scalar_bytes = (
        4 * num_clients * rounds
        if normalized_mode == "probabilistic_uq"
        else 0
    )
    total_communication_bytes = (
        2 * num_clients * rounds * payload_bytes + uq_scalar_bytes
    )
    peak_values = [float(value) for value in history["round_peak_mib"]]
    peak_mib = max(peak_values) if peak_values else float("nan")

    result: Dict[str, object] = {
        "method": _METHOD_LABELS[normalized_mode],
        "mode": normalized_mode,
        "matched_adaptation": True,
        "fusion_substitution": (
            "L2-normalized feature concatenation + MLP replaces the published "
            "bidirectional cross-modal attention"
        ),
        "f1": final_full["f1"],
        "primary_f1": final_missing["f1"],
        "accuracy": final_full["accuracy"],
        "diversity": final_full["diversity"],
        "full_text_f1": final_full["f1"],
        "full_text_accuracy": final_full["accuracy"],
        "full_text_diversity": final_full["diversity"],
        "forced_missing_text_f1": final_missing["f1"],
        "forced_missing_text_accuracy": final_missing["accuracy"],
        "forced_missing_text_diversity": final_missing["diversity"],
        "full_text_metrics": final_full,
        "forced_missing_text_metrics": final_missing,
        "client_modality_mask": [
            "multimodal" if available else "image_only"
            for available in multimodal_mask
        ],
        "multimodal_clients": multimodal_clients,
        "image_only_clients": image_only_clients,
        "client_sizes": client_sizes,
        "client_class_hist": client_class_hist,
        "history": history,
        "alpha": float(alpha),
        "num_clients": num_clients,
        "nominal_clients": num_clients,
        "active_clients": num_clients,
        "skipped_client_ids": [],
        "rounds": rounds,
        "local_epochs": local_epochs,
        "seed": int(seed),
        "feature_dim": DEFAULT_FEATURE_DIM,
        "beta": DEFAULT_BETA if "probabilistic" in normalized_mode else None,
        "rho": DEFAULT_RHO if normalized_mode == "probabilistic_uq" else None,
        "temperature": (
            DEFAULT_TEMPERATURE
            if normalized_mode == "probabilistic_uq"
            else None
        ),
        "fin_loss_weight": fin_loss_weight,
        "trainable_params": trainable_parameter_count,
        "total_params": total_parameter_count,
        "upload_bytes_per_client_per_round": payload_bytes,
        "uq_scalar_bytes": uq_scalar_bytes,
        "total_comm_bytes": total_communication_bytes,
        "wall_seconds": float(sum(history["round_seconds"])),
        "peak_mib": peak_mib,
        "peak_gpu_allocated_mib": peak_mib,
        "measurement_scope": (
            "sequential local training, client uncertainty scan, aggregation, and "
            "peak allocated GPU memory; validation and process CPU RSS excluded"
        ),
    }

    del global_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result


__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_RHO",
    "DEFAULT_TEMPERATURE",
    "DeterministicFIN",
    "FINOutput",
    "GaussianPFIN",
    "beta_nll_loss",
    "build_fin",
    "fed_uq_weights",
    "imputation_loss",
    "run_pfin_federated",
    "uncertainty_gate",
]
