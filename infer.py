from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_5d(x_np: np.ndarray) -> torch.Tensor:
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
    with torch.no_grad():
        return model(sino_5d, cheat_5d)


def _infer_slice_loop(model, sino_5d: torch.Tensor, cheat_5d: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    b, c, X, A, Z = sino_5d.shape
    so_list = []
    ro_list = []
    z = 0
    while z < Z:
        s4 = sino_5d[:, :, :, :, z]
        c4 = None
        if cheat_5d is not None:
            c4 = cheat_5d[:, :, :, :, z]
        with torch.no_grad():
            so4, ro4 = model(s4, c4)
        so_list.append(so4)
        ro_list.append(ro4)
        z += 1
    so = torch.stack(so_list, dim=-1)
    r = torch.stack(ro_list, dim=-1)
    return so, r


def run_case(model, device: torch.device, sino_path: Path, voxel_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    sino_np = np.load(sino_path)
    voxel_np = np.load(voxel_path)
    if sino_np.ndim != 3:
        raise ValueError(f"Invalid sino ndim for {sino_path.name}: {sino_np.shape}")
    if voxel_np.ndim != 3:
        raise ValueError(f"Invalid voxel ndim for {voxel_path.name}: {voxel_np.shape}")
    if int(sino_np.shape[2]) != int(voxel_np.shape[2]):
        raise ValueError(f"Z mismatch: sino Z={sino_np.shape[2]} vs voxel Z={voxel_np.shape[2]} for {sino_path.stem}")

    sino_5d = _to_5d(sino_np).to(device)
    cheat_5d = _to_5d(voxel_np).to(device)

    try:
        sino_opt_5d, recon_opt_5d = _infer_5d(model, sino_5d, cheat_5d)
    except Exception:
        sino_opt_5d, recon_opt_5d = _infer_slice_loop(model, sino_5d, cheat_5d)

    sino_opt = sino_opt_5d.squeeze(0).squeeze(0).detach().cpu().numpy()
    recon_opt = recon_opt_5d.squeeze(0).squeeze(0).detach().cpu().numpy()

    np.save(out_dir / "sino_opt.npy", sino_opt)
    np.save(out_dir / "recon_opt.npy", recon_opt)

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
    ap.add_argument("--device", type=str, default="auto")
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