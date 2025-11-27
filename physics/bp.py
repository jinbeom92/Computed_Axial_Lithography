from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn.functional as F

_FILTER_CACHE = {}
_XY_CACHE     = {}
_ANORM_CACHE  = {}

def _dev_key(t: torch.Tensor) -> tuple[str, str]:
    return (str(t.device), str(t.dtype))

def _next_pow2_for_fft(x: int) -> int:
    n = int(x)
    target = max(1, 2 * n)
    v = 64
    while v < target:
        v <<= 1
    return v

def _fftshift_1d_safeslice(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    h = n // 2
    return torch.cat((x[..., h:], x[..., :h]), dim=-1)

def _fourier_filter(size: int, filter_name: str, ref: torch.Tensor) -> torch.Tensor:
    half = size // 2
    n1 = torch.arange(1, half + 1, 2, dtype=torch.int64)
    n2 = torch.arange(half - 1, 0, -2, dtype=torch.int64)
    n = torch.cat([n1, n2]).to(torch.float32)

    f = torch.zeros(size, dtype=torch.float32)
    f[0] = 0.25
    f[1::2] = -1.0 / (math.pi * n) ** 2

    fourier_filter = 2.0 * torch.real(torch.fft.fft(f))

    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        omega = math.pi * torch.fft.fftfreq(size, dtype=torch.float32)
        ff = fourier_filter.clone()
        ff[1:] = ff[1:] * (torch.sin(omega[1:]) / (omega[1:] + 1e-12))
        fourier_filter = ff
    elif filter_name == "cosine":
        freq = torch.linspace(0.0, math.pi, size, dtype=torch.float32)
        cosine = _fftshift_1d_safeslice(torch.sin(freq))
        fourier_filter = fourier_filter * cosine
    elif filter_name == "hamming":
        win = torch.hamming_window(size, periodic=False, dtype=torch.float32)
        fourier_filter = fourier_filter * _fftshift_1d_safeslice(win)
    elif filter_name == "hann":
        win = torch.hann_window(size, periodic=False, dtype=torch.float32)
        fourier_filter = fourier_filter * _fftshift_1d_safeslice(win)
    elif filter_name == "None":
        fourier_filter[:] = 1.0
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    return fourier_filter.to(device=ref.device, dtype=ref.dtype).view(size, 1)


def _build_sampling_grid(Xdet: int, A: int, out_size: Optional[int], ref: torch.Tensor) -> tuple[torch.Tensor, int]:
    H = int(Xdet if out_size is None else out_size)
    device, dtype = ref.device, ref.dtype

    center_xy = 0.5 * float(H - 1)
    yy = torch.arange(H, dtype=torch.float32) - center_xy
    xx = torch.arange(H, dtype=torch.float32) - center_xy
    yy = yy.to(dtype=dtype, device=device)
    xx = xx.to(dtype=dtype, device=device)

    xpr, ypr = torch.meshgrid(yy, xx, indexing="ij")

    if A > 1:
        theta_deg = torch.arange(A, dtype=torch.float32) * (180.0 / float(A))
    else:
        theta_deg = torch.zeros(1, dtype=torch.float32)
    theta = (theta_deg * (math.pi / 180.0)).to(dtype=dtype, device=device)
    c, s = torch.cos(theta), torch.sin(theta)

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

    a_grid = a_norm.view(1, 1, A).expand(H, H, A)
    grid = torch.stack((a_grid, t_norm), dim=-1).reshape(1, H, H * A, 2)
    return grid, H

def _fourier_filter_cached(size: int, filter_name: str, ref: torch.Tensor) -> torch.Tensor:
    if torch.jit.is_scripting():
        return _fourier_filter(size, filter_name, ref)
    key = (int(size), str(filter_name),) + _dev_key(ref)
    filt = _FILTER_CACHE.get(key)
    if filt is None:
        filt = _fourier_filter(size, filter_name, ref)
        _FILTER_CACHE[key] = filt
    return filt

def _xy_mesh_cached(H: int, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.jit.is_scripting():
        center = 0.5 * float(H - 1)
        yy = torch.arange(H, dtype=torch.float32) - center
        xx = torch.arange(H, dtype=torch.float32) - center
        ypr = yy.to(dtype=ref.dtype, device=ref.device).view(H, 1).expand(H, H)
        xpr = xx.to(dtype=ref.dtype, device=ref.device).view(1, H).expand(H, H)
        return ypr, xpr

    key = (int(H),) + _dev_key(ref)
    packed = _XY_CACHE.get(key)
    if packed is None:
        center = 0.5 * float(H - 1)
        yy = torch.arange(H, dtype=torch.float32) - center
        xx = torch.arange(H, dtype=torch.float32) - center
        ypr = yy.to(dtype=ref.dtype, device=ref.device).view(H, 1).expand(H, H)
        xpr = xx.to(dtype=ref.dtype, device=ref.device).view(1, H).expand(H, H)
        packed = torch.stack((ypr, xpr), dim=0)
        _XY_CACHE[key] = packed
    return packed[0], packed[1]

def _anorm_full_cached(A: int, ref: torch.Tensor) -> torch.Tensor:
    if torch.jit.is_scripting():
        if A > 1:
            a_idx = torch.arange(A, dtype=torch.float32)
            return (2.0 * (a_idx / float(A - 1)) - 1.0).to(dtype=ref.dtype, device=ref.device)
        return torch.zeros(1, dtype=ref.dtype, device=ref.device)

    key = (int(A),) + _dev_key(ref)
    a = _ANORM_CACHE.get(key)
    if a is None:
        if A > 1:
            a_idx = torch.arange(A, dtype=torch.float32)
            a = (2.0 * (a_idx / float(A - 1)) - 1.0).to(dtype=ref.dtype, device=ref.device)
        else:
            a = torch.zeros(1, dtype=ref.dtype, device=ref.device)
        _ANORM_CACHE[key] = a
    return a

def _filter_projections(sino: torch.Tensor, filter_name: str = "None") -> torch.Tensor:
    if sino.ndim != 4 or sino.shape[1] != 1:
        raise RuntimeError("sino must be (B,1,X,A)")
    B, X, A = sino.shape[0], sino.shape[2], sino.shape[3]

    padX = _next_pow2_for_fft(int(X))
    if padX > X:
        sino_p = F.pad(sino, (0, 0, 0, padX - X))
    else:
        sino_p = sino

    Ff = _fourier_filter_cached(padX, filter_name, ref=sino).view(1, 1, padX, 1)
    proj = torch.fft.fft(sino_p, dim=2) * Ff
    radon_filtered = torch.fft.ifft(proj, dim=2).real[:, :, :X, :]
    return radon_filtered

def fbp2d(
    sino: torch.Tensor,
    output_size: Optional[int] = None,
    filter_name: str = "None",
    angle_chunk: int = 0,
) -> torch.Tensor:

    if sino.ndim != 4 or sino.shape[1] != 1:
        raise RuntimeError("sino must be (B,1,X,A)")
    B, Xdet, A = sino.shape[0], sino.shape[2], sino.shape[3]

    radon_filtered = _filter_projections(sino, filter_name=filter_name)
    H = int(Xdet if output_size is None else output_size)

    if angle_chunk <= 0 or angle_chunk >= A:
        grid, _ = _build_sampling_grid(int(Xdet), int(A), output_size, ref=sino)
        grid = grid.expand(B, -1, -1, -1)
        samp = F.grid_sample(radon_filtered, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        samp = samp.view(B, 1, H, H, A).sum(dim=-1)
    else:
        ypr, xpr = _xy_mesh_cached(H, ref=sino)
        a_full = _anorm_full_cached(int(A), ref=sino)
        det_center = 0.5 * float(Xdet - 1)
        det_den = float(max(1, Xdet - 1))
        scale = math.pi / (2.0 * float(max(1, A)))

        samp_acc = torch.zeros(B, 1, H, H, dtype=radon_filtered.dtype, device=radon_filtered.device)
        a0 = 0
        while a0 < A:
            cnt = int(angle_chunk) if (a0 + angle_chunk) <= A else int(A - a0)
            if A > 1:
                idx = torch.arange(cnt, dtype=torch.float32) + float(a0)
                theta = (idx * (180.0 / float(A))).to(dtype=radon_filtered.dtype, device=radon_filtered.device) * (math.pi / 180.0)
            else:
                theta = torch.zeros(1, dtype=radon_filtered.dtype, device=radon_filtered.device)
            c = torch.cos(theta)
            s = torch.sin(theta)

            t = ypr[..., None] * c.view(1, 1, -1) - xpr[..., None] * s.view(1, 1, -1)
            t_idx  = t + det_center
            t_norm = 2.0 * (t_idx / det_den) - 1.0

            a_chunk = a_full[a0:a0+cnt]
            a_grid  = a_chunk.view(1, 1, cnt).expand(H, H, cnt)

            grid = torch.stack((a_grid, t_norm), dim=-1).reshape(1, H, H * cnt, 2)
            grid = grid.expand(B, -1, -1, -1)

            samp = F.grid_sample(radon_filtered, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            samp_acc = samp_acc + samp.view(B, 1, H, H, cnt).sum(dim=-1)
            a0 = a0 + cnt

        samp = samp_acc

    scale = math.pi / (2.0 * float(max(1, A)))
    recon = samp * scale

    yy = torch.arange(H, dtype=torch.float32) - 0.5 * float(H - 1)
    xx = torch.arange(H, dtype=torch.float32) - 0.5 * float(H - 1)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    mask = ((Y**2 + X**2) <= ((0.5 * float(H - 1)) ** 2)).to(dtype=samp.dtype, device=samp.device)
    recon = recon * mask
    recon = recon.rot90(-1, dims=(2,3))
    recon = torch.flip(recon, dims=[3])
    return recon


def fbp3d(sino_5d: torch.Tensor, output_size: Optional[int] = None, filter_name: str = "None", angle_chunk: int = 0) -> torch.Tensor:
    if sino_5d.ndim != 5 or sino_5d.shape[1] != 1:
        raise RuntimeError("sino must be (B,1,X,A,Z)")
    B, X, A, Z = sino_5d.shape[0], sino_5d.shape[2], sino_5d.shape[3], sino_5d.shape[4]
    x4 = sino_5d.permute(0, 4, 1, 2, 3).reshape(B * Z, 1, X, A)
    r4 = fbp2d(x4, output_size=output_size, filter_name=filter_name, angle_chunk=angle_chunk)
    H = int(r4.size(2))
    r5 = r4.reshape(B, Z, 1, H, H).permute(0, 2, 3, 4, 1)
    return r5
