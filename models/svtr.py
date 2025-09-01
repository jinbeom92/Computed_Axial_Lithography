"""
SVTR: Sino-domain encoders → Fusion → ALign → Decoder → (sino_opt, recon_opt)

Inputs
------
- sino_xa : (B, 1, X, A)        # one z-slice batch in sino plane
- cheat_xy: (B, 1, X, Y) | None # optional cheat encoder input (train-only)

Outputs
-------
- sino_opt : (B, 1, X, A)       # optimized sinogram (non-negative)
- recon_opt: (B, 1, H, H)       # H ≈ X (FBP back-projection result)

Notes
-----
- 1D/2D encoders run in the sino (X,A) domain, then fused and aligned.
- Decoder enforces sino_opt ≥ 0 (Softplus head).
- FBP uses physics/bp.fbp2d (Hamming by default), fully autograd-friendly.
- A 5D helper forward_zstack processes (B,1,X,A,Z) → (B,1,X,A,Z),(B,1,H,H,Z).
"""

from __future__ import annotations
from pathlib import Path
import importlib.util
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn

# ----- local dynamic imports (file names starting with digits) -----
def _load_local(module_filename: str, module_name: str):
    p = Path(__file__).with_name(module_filename)
    spec = importlib.util.spec_from_file_location(module_name, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, f"Failed to load {p}"
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

_enc1d = _load_local("enc_1d.py", "enc1d")
_enc2d = _load_local("enc_2d.py", "enc2d")
_fusion = _load_local("fusion.py", "fusion")
_align  = _load_local("align.py",  "align")
_decoder= _load_local("decoder.py","decoder")

Enc1D, Enc2D, CheatEnc2D = _enc1d.Enc1D, _enc2d.Enc2D, _enc2d.CheatEnc2D
Fusion, ALign, Decoder = _fusion.Fusion, _align.ALign, _decoder.DecoderSino

# FBP
from physics.bp import fbp2d, fbp3d


class SVTR(nn.Module):
    """
    End-to-end sino→recon model.

    Args:
        c1d:       Enc1D out channels.
        c2d:       Enc2D out channels.
        fuse_out:  Fusion output channels.
        align_out: ALign output channels (None→fuse_out).
        cheat:     enable cheat path.
        cheat_out: CheatEnc2D out channels (used only if cheat=True).
        dec_hidden: decoder inner channels.
        bp_filter: FBP filter ('hamming' default).
        bp_out:    output size H for FBP (None→X).
    """

    def __init__(
        self,
        c1d: int = 64,
        c2d: int = 64,
        fuse_out: int = 128,
        align_out: Optional[int] = None,
        *,
        cheat: bool = False,
        cheat_out: int = 64,
        dec_hidden: int = 128,
        bp_filter: str = "hamming",
        bp_out: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.enc1d   = Enc1D(out_ch=c1d)
        self.enc2d   = Enc2D(out_ch=c2d)
        self.fusion  = Fusion(c1=c1d, c2=c2d, out_ch=fuse_out)
        self.cheat   = cheat
        self.cheat_enc = CheatEnc2D(out_ch=cheat_out) if cheat else None
        align_out = fuse_out if align_out is None else align_out
        self.align   = ALign(in_ch=fuse_out, out_ch=align_out, cheat_in_ch=(cheat_out if cheat else 0))
        self.decoder = Decoder(in_ch=align_out, hidden_ch=dec_hidden)
        self.bp_filter = bp_filter
        self.bp_out    = bp_out
        self.skip_alpha = nn.Parameter(torch.tensor(1.0))

    def _forward_4d(self, sino_xa: torch.Tensor, cheat_xy: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core pipeline for one z-slice batch: (B,1,X,A)[,(B,1,X,Y)] → (sino_opt,recon_opt)."""
        f1 = self.enc1d(sino_xa)                 # (B,C1,X,A)
        f2 = self.enc2d(sino_xa)                 # (B,C2,X,A)
        f  = self.fusion(f1, f2)                 # (B,F,X,A)
        cfeat = self.cheat_enc(cheat_xy) if (self.cheat and cheat_xy is not None) else None
        a  = self.align(f, cfeat)                # (B,Ao,X,A)
        sino_opt = self.decoder(a)               # (B,1,X,A)  ≥ 0
        sino_pred = self.decoder(a)
        sino_opt  = torch.clamp(sino_pred + self.skip_alpha * sino_xa, min=0.0)
        recon_opt = fbp2d(sino_opt, output_size=self.bp_out, filter_name=self.bp_filter)
        recon_opt = recon_opt.clamp(0, 1)
        return sino_opt, recon_opt

    def forward(self, sino_xa: torch.Tensor, cheat_xy: Optional[torch.Tensor] = None):
        """
        If sino_xa is 4D (B,1,X,A) → returns (B,1,X,A),(B,1,H,H).
        If sino_xa is 5D (B,1,X,A,Z) → returns (B,1,X,A,Z),(B,1,H,H,Z).
        """
        if sino_xa.dim() == 4:
            return self._forward_4d(sino_xa, cheat_xy)

        # 5D fast path (vectorized over Z)
        assert sino_xa.dim() == 5 and sino_xa.size(1) == 1, "Expected (B,1,X,A,Z)"
        B, _, X, A, Z = sino_xa.shape
        s4 = sino_xa.permute(0, 4, 1, 2, 3).reshape(B * Z, 1, X, A)

        c4 = torch.jit.annotate(Optional[torch.Tensor], None)
        if self.cheat and (cheat_xy is not None):
            _, _, Xv, Yv, Zv = cheat_xy.shape
            assert Zv == Z and Xv == X, "cheat_xy shape mismatch"
            c4 = cheat_xy.permute(0, 4, 1, 2, 3).reshape(B * Z, 1, Xv, Yv)

        so4, _ = self._forward_4d(s4, c4)
        sino_opt_5d = so4.reshape(B, Z, 1, X, A).permute(0, 2, 3, 4, 1)  # (B,1,X,A,Z)
        recon_5d = fbp3d(sino_opt_5d, output_size=self.bp_out, filter_name=self.bp_filter)
        recon_5d = recon_5d.clamp(0,1)
        return sino_opt_5d, recon_5d
