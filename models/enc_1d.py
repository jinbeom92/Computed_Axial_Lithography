"""
1D Encoder over X for each angle slice of sino(X,A,Z).

Pipeline (per z-slice, batched):
- Input: sino of shape (B, 1, X, A). (Channel-first; 1 channel required.)
- Reshape: treat each angle a independently => (B*A, 1, X).
- Apply shared 1D Conv stack with residual skip on X.
- Restore layout to (B, C_out, X, A) for downstream fusion/align.

Notes:
- Strictly follows: z-slice (X,A) -> slice along A -> A tensors (X) -> 1D conv.
- TorchScript friendly.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class Residual1D(nn.Module):
    """Minimal residual block for 1D sequences along X."""

    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, k: int = 5, gn_groups: int = 8):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, hidden_ch, k, padding=pad, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=max(1, min(gn_groups, hidden_ch)), num_channels=hidden_ch)
        self.conv2 = nn.Conv1d(hidden_ch, out_ch, k, padding=pad, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=max(1, min(gn_groups, out_ch)), num_channels=out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

        self._init()

    def _init(self) -> None:
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
        if isinstance(self.skip, nn.Conv1d):
            nn.init.kaiming_normal_(self.skip.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act(self.gn1(y))
        y = self.conv2(y)
        y = self.gn2(y)
        return self.act(y + self.skip(x))  # skip-connection is mandatory


class Enc1D(nn.Module):
    """
    1D encoder that processes sino lines along X independently for each angle.

    Args:
        out_ch: output channels per (X) line after 1D encoding.
        hidden_ch: hidden channels inside residual blocks.
        k: convolution kernel size (odd recommended).
    Input:
        x: (B, 1, X, A)  # one z-slice per batch item
    Output:
        feat: (B, out_ch, X, A)
    """

    def __init__(self, out_ch: int = 64, hidden_ch: int = 64, k: int = 5):
        super().__init__()
        # Two residual blocks; simple and stable without unnecessary control flow.
        self.block1 = Residual1D(1, hidden_ch, hidden_ch, k)
        self.block2 = Residual1D(hidden_ch, hidden_ch, out_ch, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, X, A)
        b, c, xlen, alen = x.shape
        xa = x.permute(0, 3, 1, 2).reshape(b * alen, c, xlen)  # (B*A, 1, X)
        xa = self.block1(xa)
        xa = self.block2(xa)                                   # (B*A, C_out, X)
        feat = xa.reshape(b, alen, xa.shape[1], xlen).permute(0, 2, 3, 1)  # (B, C_out, X, A)
        return feat


__all__ = ["Enc1D"]
