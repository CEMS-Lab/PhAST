"""Self-training augmentation for damage rollout prediction (#56).

Pipeline: after the teacher-forced phase, run the model autoregressively,
filter rollout snapshots by a quality criterion (e.g. PDE residual below
threshold or confidence above threshold), and append them to the training
set with an inverse-age decay weight ``w = 1 / (1 + age)``.

References
----------
- Yarowsky (1995) — bootstrapping for word-sense disambiguation
- Lee (2013)      — pseudo-label deep learning
- Sohn et al. (2020) — FixMatch (consistency + confidence filtering)
- Schwarzer et al. (2019) — temporal downsampling + self-training (rollout
  predictions added back into the training set; the source for #56)

Standalone scaffold — not wired into a training loop. All ops are
float64-compatible and autograd-friendly where applicable; the filtering
step is wrapped in ``torch.no_grad()`` because pseudo-labels enter the
graph as constants (the gradient flow during retraining comes from the
standard supervised loss, not from the labelling step).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


@dataclass
class SelfTrainingConfig:
    """Knobs for the self-training augmentation pass."""

    quality_threshold: float = 0.9
    """For confidence mode: minimum predicted confidence in [0, 1].
    For residual mode: maximum allowed PDE residual (inputs above are dropped).
    """
    use_residual_filter: bool = False
    """If True, filter by a user-supplied PDE-residual callable (Schwarzer 2019);
    else filter by output confidence (Lee 2013).
    """
    use_entropy_confidence: bool = True
    """When in confidence mode and outputs are class-prob: if True use
    ``1 - entropy / log(C)``, else use max-prob. Ignored for scalar outputs.
    """
    max_pseudo_per_pass: int = 10000
    """Cap on number of pseudo-labelled samples appended per pass."""
    age_decay: bool = True
    """If True, weight pseudo samples by ``1 / (1 + age)`` (Schwarzer 2019)."""


def _confidence_from_probs(
    probs: torch.Tensor, use_entropy: bool
) -> torch.Tensor:
    """(n, C) probabilities -> (n,) confidence scalar in [0, 1]."""
    if use_entropy:
        eps = torch.tensor(1e-12, dtype=probs.dtype, device=probs.device)
        clamped = probs.clamp(min=eps)
        entropy = -(clamped * clamped.log()).sum(dim=-1)
        n_classes = probs.shape[-1]
        log_c = torch.log(
            torch.tensor(float(n_classes), dtype=probs.dtype, device=probs.device)
        )
        return 1.0 - entropy / log_c
    return probs.max(dim=-1).values


def filter_by_quality(
    scores: torch.Tensor,
    config: SelfTrainingConfig,
) -> torch.Tensor:
    """Boolean mask (n,) selecting samples that pass the quality filter.

    ``scores`` is *confidence* (higher better, threshold is lower bound) when
    ``use_residual_filter=False``, else *residual* (lower better, threshold
    is upper bound).
    """
    if config.use_residual_filter:
        return scores <= config.quality_threshold
    return scores >= config.quality_threshold


def inverse_age_weights(
    ages: torch.Tensor,
) -> torch.Tensor:
    """Inverse-age decay weights ``w = 1 / (1 + age)`` (Schwarzer 2019)."""
    return 1.0 / (1.0 + ages.to(dtype=torch.float64))


def pseudo_label_step(
    model_predict: Callable[[torch.Tensor], torch.Tensor],
    unlabelled_inputs: torch.Tensor,
    config: SelfTrainingConfig,
    residual_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one pseudo-labelling pass.

    Parameters
    ----------
    model_predict
        Callable returning model output for a batch of inputs. May return
        scalar regression output of shape (n, ...) or class probabilities
        of shape (n, C).
    unlabelled_inputs
        (n, ...) batch of unlabelled features (e.g. autoregressive rollout
        snapshots that were not seen during teacher-forced training).
    config
        Filter knobs.
    residual_fn
        Required when ``config.use_residual_filter=True``. Maps
        ``(inputs, predictions) -> (n,)`` per-sample residual scalars.

    Returns
    -------
    (filtered_inputs, pseudo_labels, scores)
        with ``n_kept <= n``. ``scores`` are the confidence/residual values
        used for filtering; useful for downstream weighting.
    """
    with torch.no_grad():
        preds = model_predict(unlabelled_inputs)
        if config.use_residual_filter:
            if residual_fn is None:
                raise ValueError("residual_fn required when use_residual_filter=True")
            scores = residual_fn(unlabelled_inputs, preds)
        else:
            if preds.dim() >= 2 and preds.shape[-1] > 1:
                scores = _confidence_from_probs(preds, config.use_entropy_confidence)
            else:
                # Scalar regression: treat distance from {0, 1} as a
                # binary-confidence proxy (damage ∈ [0, 1] in PF problems).
                p = preds.reshape(preds.shape[0], -1).mean(dim=-1)
                scores = (p - 0.5).abs() * 2.0
        mask = filter_by_quality(scores, config)
        if int(mask.sum().item()) > config.max_pseudo_per_pass:
            kept_idx = torch.nonzero(mask, as_tuple=True)[0]
            if config.use_residual_filter:
                topk = torch.topk(
                    -scores[kept_idx], k=config.max_pseudo_per_pass
                ).indices
            else:
                topk = torch.topk(
                    scores[kept_idx], k=config.max_pseudo_per_pass
                ).indices
            new_mask = torch.zeros_like(mask)
            new_mask[kept_idx[topk]] = True
            mask = new_mask
    if preds.dim() >= 2 and preds.shape[-1] > 1 and not config.use_residual_filter:
        pseudo_labels = preds.argmax(dim=-1)
    else:
        pseudo_labels = preds
    return unlabelled_inputs[mask], pseudo_labels[mask], scores[mask]


def merge_with_labelled(
    labelled_x: torch.Tensor,
    labelled_y: torch.Tensor,
    pseudo_x: torch.Tensor,
    pseudo_y: torch.Tensor,
    pseudo_ages: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate labelled + pseudo into a single combined dataset.

    Pseudo samples are appended last so their per-sample weights can be
    addressed via slicing. Real samples receive weight 1.0; pseudo samples
    receive ``1 / (1 + age)`` if ``pseudo_ages`` is provided, else 1.0.
    """
    x = torch.cat([labelled_x, pseudo_x], dim=0)
    y = torch.cat([labelled_y, pseudo_y], dim=0)
    real_w = torch.ones(labelled_x.shape[0], dtype=torch.float64)
    pseudo_w = (
        inverse_age_weights(pseudo_ages)
        if pseudo_ages is not None
        else torch.ones(pseudo_x.shape[0], dtype=torch.float64)
    )
    weights = torch.cat([real_w, pseudo_w], dim=0)
    return x, y, weights
