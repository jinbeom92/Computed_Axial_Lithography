"""
TorchScript inference for SVTR (sino → optim. sino & recon).

Usage
-----
python infer.py \
  --sino_dir data/sino \
  --voxel_dir data/voxel \
  --ckpt results/ckpt/best_script.pt \
  --out_dir results/infer

I/O
---
- Input  : <case>_sino.npy with shape (X, A, Z), <case>_voxel.npy with shape (X, Y, Z)
- Output : out_dir/<case>/sino_opt.npy      (X, A, Z)
          out_dir/<case>/recon_opt.npy     (H, H, Z)
          out_dir/<case>/sino_mid.png      mid-Z slice visualization
          out_dir/<case>/recon_mid.png     mid-Z slice visualization

Notes
-----
- Loads TorchScript module saved as results/ckpt/best_script.pt.
- Tries 5D fast path (B,1,X,A,Z); if the checkpoint was traced on 4D only,
  it falls back to per-slice 4D inference automatically.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

# Matplotlib is only used for saving two PNGs per case.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_5d(x_np: np.ndarray) -> torch.Tensor:
    # (X, A, Z) or (X, Y, Z) -> (1, 1, X, A/Y, Z) float32
    x = np.ascontiguousarray(x_np, dtype=np.float32)
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    return t


def _save_mid_png(img2d: np.ndarray, out_path: Path, title: str) -> None:
    h, w = int(img2d.shape[0]), int(img2d.shape[1])
    fig = plt.figure(figsize=(max(3, w/256), max(3, h/256)), dpi=256)
    ax = fig.add_subplot(111)
    ax.imshow(img2d, origin="lower")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path)
    plt.close(fig)


def _infer_5d(model, sino_5d: torch.Tensor, cheat_5d: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    # Expected return: (1,1,X,A,Z), (1,1,H,H,Z)
    with torch.no_grad():
        return model(sino_5d, cheat_5d)


def _infer_slice_loop(model, sino_5d: torch.Tensor, cheat_5d: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    # Fallback when checkpoint was traced on 4D only.
    # Loop over Z, call (1,1,X,A) → stack back to 5D.
    b, c, X, A, Z = sino_5d.shape
    so_list = []
    ro_list = []
    z = 0
    while z < Z:
        s4 = sino_5d[:, :, :, :, z]  # (1,1,X,A)
        c4 = None
        if cheat_5d is not None:
            c4 = cheat_5d[:, :, :, :, z]  # (1,1,X,Y)
        with torch.no_grad():
            so4, ro4 = model(s4, c4)      # (1,1,X,A), (1,1,H,H)
        so_list.append(so4)
        ro_list.append(ro4)
        z += 1
    so = torch.stack(so_list, dim=-1)       # (1,1,X,A,Z)
    r = torch.stack(ro_list, dim=-1)        # (1,1,H,H,Z)
    return so, r


def run_case(model, device: torch.device, sino_path: Path, voxel_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    sino_np = np.load(sino_path)  # (X, A, Z)
    voxel_np = np.load(voxel_path)  # (X, Y, Z)
    if sino_np.ndim != 3:
        raise ValueError(f"Invalid sino ndim for {sino_path.name}: {sino_np.shape}")
    if voxel_np.ndim != 3:
        raise ValueError(f"Invalid voxel ndim for {voxel_path.name}: {voxel_np.shape}")
    if int(sino_np.shape[2]) != int(voxel_np.shape[2]):
        raise ValueError(f"Z mismatch: sino Z={sino_np.shape[2]} vs voxel Z={voxel_np.shape[2]} for {sino_path.stem}")

    sino_5d = _to_5d(sino_np).to(device)    # (1,1,X,A,Z)
    cheat_5d = _to_5d(voxel_np).to(device)  # (1,1,X,Y,Z)

    try:
        sino_opt_5d, recon_opt_5d = _infer_5d(model, sino_5d, cheat_5d)
    except Exception:
        # 5D fast path unsupported by traced graph → slice loop
        sino_opt_5d, recon_opt_5d = _infer_slice_loop(model, sino_5d, cheat_5d)

    # Save NPZ (drop batch & channel dims)
    sino_opt = sino_opt_5d.squeeze(0).squeeze(0).detach().cpu().numpy()     # (X, A, Z)
    recon_opt = recon_opt_5d.squeeze(0).squeeze(0).detach().cpu().numpy()   # (H, H, Z)

    np.save(out_dir / "sino_opt.npy", sino_opt)
    np.save(out_dir / "recon_opt.npy", recon_opt)

    # Mid-Z visualizations
    mid = int(sino_opt.shape[2] // 2)
    _save_mid_png(sino_opt[:, :, mid], out_dir / "sino_mid.png", "sino_opt @ mid-Z")
    _save_mid_png(recon_opt[:, :, mid], out_dir / "recon_mid.png", "recon_opt @ mid-Z")


def find_pairs(sino_dir: Path, voxel_dir: Path, sfx_sino: str = "_sino.npy", sfx_voxel: str = "_voxel.npy") -> list[tuple[str, Path, Path]]:
    pairs = []
    for s_path in sorted(sino_dir.glob(f"*{sfx_sino}")):
        stem = s_path.name[:-len(sfx_sino)]
        v_path = voxel_dir / f"{stem}{sfx_voxel}"
        if not v_path.exists():
            raise FileNotFoundError(f"Missing voxel for case '{stem}': {v_path}")
        pairs.append((stem, s_path, v_path))
    if len(pairs) == 0:
        raise FileNotFoundError(f"No pairs found in {sino_dir} and {voxel_dir}")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sino_dir", type=str, default="data/sino")
    ap.add_argument("--voxel_dir", type=str, default="data/voxel")
    ap.add_argument("--ckpt", type=str, default="results/ckpt/best_script.pt")
    ap.add_argument("--out_dir", type=str, default="results/infer")
    ap.add_argument("--device", type=str, default="auto")  # auto|cuda|cpu
    args = ap.parse_args()

    sino_dir = Path(args.sino_dir)
    voxel_dir = Path(args.voxel_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = torch.jit.load(args.ckpt, map_location=device).eval()
    torch.set_grad_enabled(False)

    pairs = find_pairs(sino_dir, voxel_dir)
    i = 0
    while i < len(pairs):
        stem, s_path, v_path = pairs[i]
        case_out = out_root / stem
        print(f"[infer] case={stem}")
        run_case(model, device, s_path, v_path, case_out)
        i += 1


if __name__ == "__main__":
    main()