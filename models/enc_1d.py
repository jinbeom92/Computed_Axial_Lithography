from __future__ import annotations
import torch
import torch.nn as nn


class Residual1D(nn.Module):
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
        return self.act(y + self.skip(x))


class Enc1D(nn.Module):
    def __init__(self, out_ch: int = 64, hidden_ch: int = 64, k: int = 5):
        super().__init__()
        self.block1 = Residual1D(1, hidden_ch, hidden_ch, k)
        self.block2 = Residual1D(hidden_ch, hidden_ch, out_ch, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, xlen, alen = x.shape
        xa = x.permute(0, 3, 1, 2).reshape(b * alen, c, xlen)
        xa = self.block1(xa)
        xa = self.block2(xa)
        feat = xa.reshape(b, alen, xa.shape[1], xlen).permute(0, 2, 3, 1)
        return feat


__all__ = ["Enc1D"]
