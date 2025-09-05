from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual2D(nn.Module):
    """Residual 2D block with GroupNorm and mandatory skip."""
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, k, padding=p, bias=False)
        self.gn1 = nn.GroupNorm(1, hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, out_ch, k, padding=p, bias=False)
        self.gn2 = nn.GroupNorm(1, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
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
    Non-interpolating aligner with optional cheat injection (TorchScript-safe).

    Policy
    ------
    - No XY->XA resampling. Cheat stays on its grid.
    - Reduce along Y (mean) -> (B,Cc,X,1), replicate across A.
    - Concatenate [sino, σ(gate)*cheat_broadcast] then 1×1 projection.

    TorchScript note
    ----------------
    Guard calls with `self.cat_proj is not None` so the type is refined
    from Optional[Conv2d] → Conv2d within the branch.
    """
    def __init__(self, in_ch: int, out_ch: Optional[int] = None, cheat_in_ch: int = 0, k: int = 3):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch

        self.proj_in = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        # Optional proj for cheat concat; None when cheat_in_ch == 0
        self.cat_proj: Optional[nn.Conv2d] = (
            nn.Conv2d(out_ch + cheat_in_ch, out_ch, 1, bias=False) if cheat_in_ch > 0 else None
        )
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.refine = Residual2D(out_ch, out_ch, out_ch, k)
        self._announced: bool = torch.jit.Attribute(False, bool)

        if isinstance(self.proj_in, nn.Conv2d):
            nn.init.kaiming_normal_(self.proj_in.weight, nonlinearity="linear")
        if self.cat_proj is not None:
            nn.init.kaiming_normal_(self.cat_proj.weight, nonlinearity="linear")

    def forward(self, fused_sino: torch.Tensor, cheat_xy: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self._announced:
            mode = "script" if torch.jit.is_scripting() else "eager/trace"
            cheat_enabled = self.cat_proj is not None
            cheat_active  = cheat_enabled and (cheat_xy is not None)
            print("[Align]", "mode=", mode, "cheat_enabled=", cheat_enabled, "cheat_active=", cheat_active)
            self._announced = True

        x = self.proj_in(fused_sino)

        if (self.cat_proj is not None) and (cheat_xy is not None):
            c = cheat_xy.mean(dim=3, keepdim=True)
            c = c.expand(-1, -1, -1, x.shape[3])
            y = torch.cat((x, self.gate.sigmoid() * c), dim=1)
            x = self.cat_proj(y)

        x = self.refine(x)
        return x