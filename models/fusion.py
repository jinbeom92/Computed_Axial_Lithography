"""
Feature fusion for sino-domain encoders.

Inputs:
- feat_1d : (B, C1, X, A) from Enc1D
- feat_2d : (B, C2, X, A) from Enc2D

Output:
- fused   : (B, C_out, X, A)

Design:
- Channel-wise concat -> 1×1 projection -> residual 3×3 refinement.
- Per-branch learnable scalars (alpha) softly weight each source before concat.
- TorchScript-friendly, no unnecessary control flow.
"""

from __future__ import annotations
import torch
import torch.nn as nn


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
        return self.act(y + self.skip(x))  # mandatory skip


class Fusion(nn.Module):
    """
    Fuse Enc1D and Enc2D features in sino (X,A) domain.

    Args:
        c1: channels from Enc1D.
        c2: channels from Enc2D.
        out_ch: output channels after fusion.
        k: kernel size for residual refinement (odd).
    """

    def __init__(self, c1: int, c2: int, out_ch: int = 128, k: int = 3):
        super().__init__()
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))
        self.proj = nn.Conv2d(c1 + c2, out_ch, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(1, out_ch)
        self.act = nn.ReLU(inplace=True)
        self.refine = Residual2D(out_ch, out_ch, out_ch, k)
        self._init()

    def _init(self) -> None:
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")

    def forward(self, feat_1d: torch.Tensor, feat_2d: torch.Tensor) -> torch.Tensor:
        # (B, C1, X, A), (B, C2, X, A)
        x = torch.cat((self.alpha1 * feat_1d, self.alpha2 * feat_2d), dim=1)
        x = self.act(self.gn(self.proj(x)))
        x = self.refine(x)
        return x


__all__ = ["Fusion", "Residual2D"]
