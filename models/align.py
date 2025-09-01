"""
Alignment and optional cheat-feature injection.

Inputs:
- fused_sino : (B, C_in, X, A)   # from Fusion
- cheat_xy   : (B, Cc,  X, Y)    # from CheatEnc2D (optional)

Output:
- aligned    : (B, C_out, X, A)

Design:
- Optional resampling of cheat XY -> XA via bilinear interpolate.
- 1×1 projections to match channels, then gated additive injection.
- Residual 3×3 refinement. Gate is learnable in [0,1] via sigmoid.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual2D(nn.Module):
    """Minimal residual 2D block with mandatory skip-connection."""

    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, k, padding=p, bias=False)
        self.gn1 = nn.GroupNorm(1, hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, out_ch, k, padding=p, bias=False)
        self.gn2 = nn.GroupNorm(1, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self._init()

    def _init(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="linear")
        if isinstance(self.skip, nn.Conv2d):
            nn.init.kaiming_normal_(self.skip.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return self.act(y + self.skip(x))


class ALign(nn.Module):
    """
    Align fused sino features and optionally inject cheat features.

    Args:
        in_ch:  channels of fused_sino.
        out_ch: output channels (defaults to in_ch).
        cheat_in_ch: channels of cheat_xy (0 disables cheat path).
        k: kernel size for residual refinement (odd).
    """

    def __init__(self, in_ch: int, out_ch: Optional[int] = None, cheat_in_ch: int = 0, k: int = 3):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch

        self.proj_in = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.has_cheat = cheat_in_ch > 0
        self.cheat_proj = nn.Conv2d(cheat_in_ch, out_ch, 1, bias=False) if self.has_cheat else None

        self.gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(gate) ∈ (0,1), starts closed
        self.refine = Residual2D(out_ch, out_ch, out_ch, k)

        if isinstance(self.proj_in, nn.Conv2d):
            nn.init.kaiming_normal_(self.proj_in.weight, nonlinearity="linear")
        if self.cheat_proj is not None:
            nn.init.kaiming_normal_(self.cheat_proj.weight, nonlinearity="linear")

    def forward(self, fused_sino: torch.Tensor, cheat_xy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            fused_sino: (B, C_in, X, A)
            cheat_xy  : (B, Cc,  X, Y) or None
        """
        x = self.proj_in(fused_sino)  # -> (B, C_out, X, A)

        if self.has_cheat and cheat_xy is not None:
            # Geometry-agnostic alignment XY -> XA via interpolation.
            xa = F.interpolate(cheat_xy, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
            xa = self.cheat_proj(xa)  # -> (B, C_out, X, A)
            g = self.gate.sigmoid()
            x = x + g * xa

        x = self.refine(x)
        return x


__all__ = ["ALign", "Residual2D"]
