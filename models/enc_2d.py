"""
2D encoders for z-sliced inputs with mandatory residual skips.

Sino encoder:
- Input:  (B, 1, X, A)  # one (X,A) map per sample (z-slice upstream)
- Output: (B, C_out, X, A)

Cheat voxel encoder:
- Input:  (B, 1, X, Y)  # one (X,Y) map per sample (z-slice upstream)
- Output: (B, C_out, X, Y)

Design:
- Two Residual2D blocks with RELU activations.
- GroupNorm with 1 group (robust w.r.t. channel counts, TorchScript-friendly).
- Kaiming initialization for stable training.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class Residual2D(nn.Module):
    """Minimal residual 2D block with mandatory skip-connection."""

    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=k, padding=p, bias=False)
        self.gn1 = nn.GroupNorm(1, hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, out_ch, kernel_size=k, padding=p, bias=False)
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


class Enc2D(nn.Module):
    """
    2D encoder for sino (X,A) maps.

    Args:
        out_ch: output channels.
        hidden_ch: hidden channels inside residual blocks.
        k: square kernel size (odd recommended).
    Input:
        x: (B, 1, X, A)
    Output:
        feat: (B, out_ch, X, A)
    """

    def __init__(self, out_ch: int = 64, hidden_ch: int = 64, k: int = 5):
        super().__init__()
        self.block1 = Residual2D(1, hidden_ch, hidden_ch, k)
        self.block2 = Residual2D(hidden_ch, hidden_ch, out_ch, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


class CheatEnc2D(nn.Module):
    """
    Cheat 2D encoder for voxel (X,Y) maps.

    Args:
        out_ch: output channels.
        hidden_ch: hidden channels inside residual blocks.
        k: square kernel size (odd recommended).
    Input:
        x: (B, 1, X, Y)
    Output:
        feat: (B, out_ch, X, Y)
    """

    def __init__(self, out_ch: int = 64, hidden_ch: int = 64, k: int = 5):
        super().__init__()
        self.block1 = Residual2D(1, hidden_ch, hidden_ch, k)
        self.block2 = Residual2D(hidden_ch, hidden_ch, out_ch, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


__all__ = ["Enc2D", "CheatEnc2D", "Residual2D"]
