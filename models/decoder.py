from __future__ import annotations
import torch
import torch.nn as nn


class Residual2D(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, k, padding=p, bias=False)
        self.gn1 = nn.GroupNorm(1, hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, out_ch, k, padding=p, bias=False)
        self.gn2 = nn.GroupNorm(1, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        # self.act = nn.ReLU(inplace=True)
        self.act = nn.PReLU()
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="linear")
        if isinstance(self.skip, nn.Conv2d):
            nn.init.kaiming_normal_(self.skip.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return self.act(y + self.skip(x))


class DecoderSino(nn.Module):
    def __init__(self, in_ch: int, hidden: int | None = None, *, hidden_ch: int | None = None, k: int = 3):
        super().__init__()
        h = hidden if hidden is not None else (hidden_ch if hidden_ch is not None else 128)
        self.block1 = Residual2D(in_ch, h, h, k)
        self.block2 = Residual2D(h, h, h, k)
        self.head   = nn.Conv2d(h, 1, kernel_size=1, bias=True)
        self.pos    = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return self.pos(x)

Decoder = DecoderSino

__all__ = ["DecoderSino", "Decoder", "Residual2D"]
