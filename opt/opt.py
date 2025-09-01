"""
AdamW optimizer builder with sane param-grouping.

Features
- Decoupled weight decay (AdamW) with two groups:
  1) decay: all weights except normalization layers and biases
  2) no_decay: normalization params and biases (weight_decay=0.0)
- Optional fused AdamW when available (PyTorch ≥ 2.0, CUDA).

Usage
-------
opt = build_adamw(model, lr=1e-3, weight_decay=0.01)
"""

from __future__ import annotations
import torch
import torch.nn as nn


_NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.GroupNorm, nn.LayerNorm,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)


def build_adamw(
    model: nn.Module,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    fused: bool | None = None,
) -> torch.optim.Optimizer:
    """
    Create AdamW with bias/norm excluded from weight decay.

    Args:
        model: nn.Module whose parameters will be optimized.
        lr: learning rate.
        weight_decay: L2 weight decay applied only to the "decay" group.
        betas: Adam betas.
        eps: Adam epsilon.
        fused: if None, auto-enable when supported on CUDA; otherwise use given flag.

    Returns:
        torch.optim.AdamW instance with two param groups.
    """
    no_decay_params = set()
    for m in model.modules():
        if isinstance(m, _NORM_TYPES):
            for p in m.parameters(recurse=False):
                no_decay_params.add(p)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p in no_decay_params or name.endswith(".bias")) else decay).append(p)

    groups = [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    kwargs = dict(lr=float(lr), betas=tuple(betas), eps=float(eps))
    if fused is not None and hasattr(torch.optim.AdamW, "fused"):
        kwargs["fused"] = bool(fused)
    elif hasattr(torch.optim.AdamW, "fused"):
        kwargs["fused"] = torch.cuda.is_available()

    return torch.optim.AdamW(groups, **kwargs)


__all__ = ["build_adamw"]
