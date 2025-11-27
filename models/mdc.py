from __future__ import annotations
from pathlib import Path
import importlib.util
from typing import Optional, Tuple

import torch
import torch.nn as nn
from physics.bp import fbp2d, fbp3d

def _load_local(module_filename: str, module_name: str):
    p = Path(__file__).with_name(module_filename)
    spec = importlib.util.spec_from_file_location(module_name, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, f"Failed to load {p}"
    spec.loader.exec_module(mod)
    return mod

_enc1d = _load_local("enc_1d.py", "enc1d")
_enc2d = _load_local("enc_2d.py", "enc2d")
_fusion = _load_local("fusion.py", "fusion")
_align  = _load_local("align.py",  "align")
_decoder= _load_local("decoder.py","decoder")

Enc1D, Enc2D, CheatEnc2D = _enc1d.Enc1D, _enc2d.Enc2D, _enc2d.CheatEnc2D
Fusion, ALign, Decoder = _fusion.Fusion, _align.ALign, _decoder.DecoderSino


from physics.bp import fbp2d, fbp3d


class MDC(nn.Module):
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
        bp_filter: str = "None",
        bp_out: Optional[int] = None,
        bp_angle_chunk: int = 0
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
        self.skip_alpha = nn.Parameter(torch.tensor(0.0))
        self.bp_angle_chunk = int(bp_angle_chunk)

    def _forward_4d(self, sino_xa: torch.Tensor, cheat_xy: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        f1 = self.enc1d(sino_xa)
        f2 = self.enc2d(sino_xa)
        f  = self.fusion(f1, f2)
        
        cfeat = torch.jit.annotate(Optional[torch.Tensor], None)
        
        if (self.cheat_enc is not None) and (cheat_xy is not None):
            cfeat = self.cheat_enc(cheat_xy)
        a  = self.align(f, cfeat)
        sino_opt = self.decoder(a)

        recon_opt = fbp2d(sino_opt, output_size=self.bp_out, filter_name=self.bp_filter, angle_chunk=self.bp_angle_chunk)
        recon_opt = recon_opt / torch.amax(recon_opt, dim=(2, 3), keepdim=True)
        return sino_opt, recon_opt

    def forward(self, sino_xa: torch.Tensor, cheat_xy: Optional[torch.Tensor] = None):
        if sino_xa.dim() == 4:
            return self._forward_4d(sino_xa, cheat_xy)

        assert sino_xa.dim() == 5 and sino_xa.size(1) == 1
        B, _, X, A, Z = sino_xa.shape
        s4 = sino_xa.permute(0, 4, 1, 2, 3).reshape(B * Z, 1, X, A)
        c4 = torch.jit.annotate(Optional[torch.Tensor], None)
        
        if self.cheat and (cheat_xy is not None):
            _, _, Xv, Yv, Zv = cheat_xy.shape
            assert Zv == Z and Xv == X
            c4 = cheat_xy.permute(0, 4, 1, 2, 3).reshape(B * Z, 1, Xv, Yv)

        so4, _ = self._forward_4d(s4, c4)
        sino_opt_5d = so4.reshape(B, Z, 1, X, A).permute(0, 2, 3, 4, 1)
        recon_5d = fbp3d(sino_opt_5d, output_size=self.bp_out, filter_name=self.bp_filter, angle_chunk=self.bp_angle_chunk)
        recon_5d = recon_5d / torch.amax(recon_5d, dim=(2, 3), keepdim=True)
        return sino_opt_5d, recon_5d
