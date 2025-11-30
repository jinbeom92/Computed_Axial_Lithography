from __future__ import annotations
import torch
import torch.nn as nn


class Residual2D(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=k, padding=p, bias=False)
        self.gn1 = nn.GroupNorm(1, hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.gn2 = nn.GroupNorm(1, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        # self.act = nn.ReLU(inplace=True)
        self.act = nn.PReLU()
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


class Enc2D(nn.Module):
    def __init__(self, out_ch: int = 64, hidden_ch: int = 64, k: int = 5):
        super().__init__()
        self.block1 = Residual2D(1, hidden_ch, hidden_ch, k)
        self.block2 = Residual2D(hidden_ch, hidden_ch, out_ch, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


class CheatEnc2D(nn.Module):
    def __init__(self, out_ch: int = 64, hidden_ch: int = 64, k: int = 5):
        super().__init__()
        self.block1 = Residual2D(1, hidden_ch, hidden_ch, k)
        self.block2 = Residual2D(hidden_ch, hidden_ch, out_ch, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


__all__ = ["Enc2D", "CheatEnc2D", "Residual2D"]
