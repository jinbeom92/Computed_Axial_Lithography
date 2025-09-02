"""
Torch inverse Radon (Filtered Back-Projection) with F.grid_sample (Hamming).

- TorchScript/Trace friendly: no keyword-only args, no math.log2, no tensor->python conversions in hot paths,
  and NO direct 'device=' passing into torch factory funcs during scripting.
- skimage iradon parity: Hamming filter, π/(2·A) scaling.  (ref) radon_transform.py
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn.functional as F


# --------- helpers ---------
def _next_pow2_for_fft(x: int) -> int:
    """min 64, and at least next pow2 of (2*x) using bit shifts (JIT-safe)."""
    n = int(x)
    target = max(1, 2 * n)
    v = 64
    while v < target:
        v <<= 1
    return v

def _fftshift_1d_safeslice(x: torch.Tensor) -> torch.Tensor:
    """1D fftshift via slicing (no torch.roll, trace-safe)."""
    n = x.shape[-1]
    h = n // 2
    return torch.cat((x[..., h:], x[..., :h]), dim=-1)

def _fourier_filter(size: int, filter_name: str, ref: torch.Tensor) -> torch.Tensor:
    """
    Construct Fourier filter like skimage._get_fourier_filter.
    Build on CPU in float32, then move to ref.device/ref.dtype at the end.
    Returns: (size, 1)
    """
    # ---- build in CPU float32 (JIT-safe; no device kwarg) ----
    half = size // 2
    n1 = torch.arange(1, half + 1, 2, dtype=torch.int64)     # CPU
    n2 = torch.arange(half - 1, 0, -2, dtype=torch.int64)    # CPU
    n = torch.cat([n1, n2]).to(torch.float32)                # CPU

    f = torch.zeros(size, dtype=torch.float32)               # CPU
    f[0] = 0.25
    f[1::2] = -1.0 / (math.pi * n) ** 2

    fourier_filter = 2.0 * torch.real(torch.fft.fft(f))      # CPU float32

    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        omega = math.pi * torch.fft.fftfreq(size, dtype=torch.float32)  # CPU
        # start from index 1; avoid divide-by-zero
        ff = fourier_filter.clone()
        ff[1:] = ff[1:] * (torch.sin(omega[1:]) / (omega[1:] + 1e-12))
        fourier_filter = ff
    elif filter_name == "cosine":
        freq = torch.linspace(0.0, math.pi, size, dtype=torch.float32)  # CPU
        cosine = _fftshift_1d_safeslice(torch.sin(freq))
        fourier_filter = fourier_filter * cosine
    elif filter_name == "hamming":
        # build window on CPU to avoid device kwarg in factory
        win = torch.hamming_window(size, periodic=False, dtype=torch.float32)  # CPU
        fourier_filter = fourier_filter * _fftshift_1d_safeslice(win)
    elif filter_name == "hann":
        win = torch.hann_window(size, periodic=False, dtype=torch.float32)     # CPU
        fourier_filter = fourier_filter * _fftshift_1d_safeslice(win)
    elif filter_name == "None":
        fourier_filter[:] = 1.0
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    # ---- move to ref's device/dtype at the end ----
    return fourier_filter.to(device=ref.device, dtype=ref.dtype).view(size, 1)


def _filter_projections(sino: torch.Tensor, filter_name: str = "None") -> torch.Tensor:
    """
    sino: (B,1,X,A) → frequency-domain filtering (ramp default) → (B,1,X,A)
    """
    if sino.ndim != 4 or sino.shape[1] != 1:
        raise RuntimeError("sino must be (B,1,X,A)")
    B, X, A = sino.shape[0], sino.shape[2], sino.shape[3]

    padX = _next_pow2_for_fft(int(X))
    if padX > X:
        sino_p = F.pad(sino, (0, 0, 0, padX - X))
    else:
        sino_p = sino

    # build filter like skimage, then convert to sino dtype/device
    Ff = _fourier_filter(padX, filter_name, ref=sino).view(1, 1, padX, 1)
    proj = torch.fft.fft(sino_p, dim=2) * Ff
    radon_filtered = torch.fft.ifft(proj, dim=2).real[:, :, :X, :]
    return radon_filtered


def _build_sampling_grid(Xdet: int, A: int, out_size: Optional[int], ref: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Build grid sampling all angles at once.
    Returns:
      grid: (1, H, H*A, 2), align_corners=True normalized coords
      H   : output size
    """
    H = int(Xdet if out_size is None else out_size)
    device, dtype = ref.device, ref.dtype

    center_xy = 0.5 * float(H - 1)
    # create on CPU then move; avoid 'device=' in factory during scripting
    yy = torch.arange(H, dtype=torch.float32) - center_xy
    xx = torch.arange(H, dtype=torch.float32) - center_xy
    yy = yy.to(dtype=dtype, device=device)
    xx = xx.to(dtype=dtype, device=device)

    xpr, ypr = torch.meshgrid(yy, xx, indexing="ij")  # (H,H), device/dtype=ref

    if A > 1:
        theta_deg = torch.arange(A, dtype=torch.float32) * (180.0 / float(A))  # CPU
    else:
        theta_deg = torch.zeros(1, dtype=torch.float32)                         # CPU
    theta = (theta_deg * (math.pi / 180.0)).to(dtype=dtype, device=device)
    c, s = torch.cos(theta), torch.sin(theta)  # (A,)

    # t = y cosθ - x sinθ  → (H,H,A)
    t = ypr[..., None] * c[None, None, :] - xpr[..., None] * s[None, None, :]

    center = 0.5 * float(Xdet - 1)
    t_idx  = t + center
    det_den = float(max(1, Xdet - 1))
    t_norm = 2.0 * (t_idx / det_den) - 1.0

    if A > 1:
        a_idx = torch.arange(A, dtype=torch.float32)
        a_norm = 2.0 * (a_idx / float(A - 1)) - 1.0
    else:
        a_norm = torch.zeros(1, dtype=torch.float32)
    a_norm = a_norm.to(dtype=dtype, device=device)

    a_grid = a_norm.view(1, 1, A).expand(H, H, A)  # (H,H,A)
    grid = torch.stack((a_grid, t_norm), dim=-1).reshape(1, H, H * A, 2)
    return grid, H


# --------- public API (TorchScript-friendly signatures) ---------
def fbp2d(sino: torch.Tensor, output_size: Optional[int] = None, filter_name: str = "hamming") -> torch.Tensor:
    """
    FBP for one z-slice batch.
    Args:
        sino        : (B,1,X,A)
        output_size: None → X
        filter_name: hamming | ramp | hann | cosine | shepp-logan | None
    Returns:
        recon_2d    : (B,1,H,H)
    """
    if sino.ndim != 4 or sino.shape[1] != 1:
        raise RuntimeError("sino must be (B,1,X,A)")
    B, X, A = sino.shape[0], sino.shape[2], sino.shape[3]

    radon_filtered = _filter_projections(sino, filter_name=filter_name)      # (B,1,X,A)
    grid, H = _build_sampling_grid(int(X), int(A), output_size, ref=sino)   # (1,H,H*A,2)
    grid = grid.expand(B, -1, -1, -1)

    samp = F.grid_sample(radon_filtered, grid, mode="bilinear", padding_mode="zeros", align_corners=True)  # (B,1,H,H*A)
    samp = samp.view(B, 1, H, H, A).sum(dim=-1)

    scale = math.pi / (2.0 * float(max(1, A)))  # skimage iradon scaling
    recon = samp * scale
    
    # circle mask
    yy = torch.arange(H, dtype=torch.float32) - 0.5 * float(H - 1)
    xx = torch.arange(H, dtype=torch.float32) - 0.5 * float(H - 1)
    r  = 0.5 * float(H - 1)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    mask = ((Y**2 + X**2) <= (r * r)).to(dtype=samp.dtype, device=samp.device)
    recon = (samp * (math.pi / (2.0 * float(max(1, A))))) * mask
    
    return recon


def fbp3d(sino_5d: torch.Tensor, output_size: Optional[int] = None, filter_name: str = "hamming") -> torch.Tensor:
    """
    Vectorized Z-stack FBP.
    sino_5d: (B,1,X,A,Z) → (B,1,H,H,Z)
    """
    if sino_5d.ndim != 5 or sino_5d.shape[1] != 1:
        raise RuntimeError("sino must be (B,1,X,A,Z)")
    B, X, A, Z = sino_5d.shape[0], sino_5d.shape[2], sino_5d.shape[3], sino_5d.shape[4]
    x4 = sino_5d.permute(0, 4, 1, 2, 3).reshape(B * Z, 1, X, A)
    r4 = fbp2d(x4, output_size=output_size, filter_name=filter_name)
    H = int(r4.size(2))
    r5 = r4.reshape(B, Z, 1, H, H).permute(0, 2, 3, 4, 1)
    return r5
