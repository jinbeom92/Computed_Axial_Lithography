import argparse, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Robust imports: package vs flat files ---
from losses.ssim import SSIMLoss
from losses.mse import MSELoss
from losses.ec   import ContrastLoss


def load_model(jit_path: Path, device: torch.device) -> torch.jit.ScriptModule:
    if not jit_path.exists():
        alt = jit_path.parent / "best_script.pt"
        if alt.exists():
            jit_path = alt
    print(f"[load] TorchScript model: {jit_path}")
    model = torch.jit.load(str(jit_path), map_location=device)
    model.eval()
    return model


def find_pair_paths(sino_dir: Path, voxel_dir: Path):
    pairs = []
    for sp in sorted(sino_dir.glob("*_sino.npy")):
        cid = sp.name[:-9]  # drop '_sino.npy'
        vp = voxel_dir / f"{cid}_voxel.npy"
        if vp.exists():
            pairs.append((cid, sp, vp))
        else:
            print(f"[warn] voxel not found for {cid}: {vp}")
    return pairs


def to_tensor(arr: np.ndarray, device, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)


@torch.no_grad()
def infer_case(
    model: torch.jit.ScriptModule,
    cid: str,
    sino_path: Path,
    voxel_path: Path,
    out_dir: Path,
    device: torch.device,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
):
    # ---- load ----
    sino  = np.load(sino_path)   # (X,A,Z)
    voxel = np.load(voxel_path)  # (X,Y,Z)
    assert sino.ndim == 3 and voxel.ndim == 3
    X, A, Z = sino.shape
    Xv, Y, Zv = voxel.shape
    assert X == Xv and Z == Zv, "X/Z mismatch between sino and voxel"

    # ---- pre-allocate full volumes (Numpy) ----
    so_vol = np.empty((X, A, Z), dtype=np.float32)   # sino_opt
    ro_vol = None                                    # recon_opt (H,H,Z), H unknown until first slice

    # ---- per-z inference (4D call) ----
    for z in range(Z):
        s_z = to_tensor(sino[:, :, z], device).unsqueeze(0).unsqueeze(0)  # (1,1,X,A)

        try:
            so_z, ro_z = model(s_z)          # (1,1,X,A), (1,1,H,H)
        except TypeError:
            so_z, ro_z = model(s_z, None)

        if clamp_min is not None or clamp_max is not None:
            lo = clamp_min if clamp_min is not None else -float("inf")
            hi = clamp_max if clamp_max is not None else  float("inf")
            ro_z = ro_z.clamp(min=lo, max=hi)

        # ---- move to CPU 2D slices & assign into (…,Z) ----
        so2d = so_z[0, 0].detach().cpu().numpy().astype(np.float32)  # (X,A)
        ro2d = ro_z[0, 0].detach().cpu().numpy().astype(np.float32)  # (H,H)

        if ro_vol is None:
            H = int(ro2d.shape[0])
            ro_vol = np.empty((H, H, Z), dtype=np.float32)

        so_vol[:, :, z] = so2d
        ro_vol[:, :, z] = ro2d

    # ---- sanity check & save (entire Z) ----
    assert so_vol.shape == (X, A, Z)
    assert ro_vol.shape[2] == Z
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{cid}_sino_opt.npy",  so_vol)
    np.save(out_dir / f"{cid}_recon_opt.npy", ro_vol)

    # ---- metrics on full volumes (vectorized over Z as batch axis) ----
    # Convert (H,H,Z)/(X,Y,Z) -> [Z,1,H,W]
    R_hat_4d = torch.from_numpy(ro_vol).permute(2, 0, 1).unsqueeze(1).to(device=device, dtype=torch.float32)
    V_gt_4d  = torch.from_numpy(voxel).permute(2, 0, 1).unsqueeze(1).to(device=device, dtype=torch.float32)
    V_gt_4d  = (V_gt_4d == 1.0).to(torch.float32) if not V_gt_4d.is_floating_point() else (V_gt_4d > 0.5).to(torch.float32)

    # Instantiate losses once (TorchScript-friendly, torch-only)
    crit_ssim = SSIMLoss(boundary_value=0.8).to(device)
    crit_mse  = MSELoss(boundary_value=0.8).to(device)
    crit_ec   = ContrastLoss().to(device)

    ssim = float(1.0 - crit_ssim(R_hat_4d, V_gt_4d).item())
    mse  = float(crit_mse(R_hat_4d, V_gt_4d).item())
    ec   = float(1.0 - 2.0 * crit_ec(R_hat_4d, V_gt_4d).item())  # loss=(1-EC)*0.5 → EC=1-2*loss

    # ---- preview PNGs (mid z only) ----
    mid = Z // 2
    plt.figure(figsize=(6,5))
    plt.imshow(so_vol[:, :, mid], cmap="CMRmap", origin="lower", aspect="auto")
    plt.title(f"{cid} | sino_opt z={mid}"); plt.colorbar(); plt.tight_layout()
    plt.savefig(out_dir / f"{cid}_sino_opt_z{mid:03d}.png", dpi=150); plt.close()

    H = ro_vol.shape[0]
    plt.figure(figsize=(5,5))
    plt.imshow(ro_vol[:, :, mid], cmap="CMRmap", origin="lower")
    plt.title(f"{cid} | recon_opt z={mid} (H={H})"); plt.colorbar(); plt.tight_layout()
    plt.savefig(out_dir / f"{cid}_recon_opt_z{mid:03d}.png", dpi=150); plt.close()

    # ---- metrics.json ----
    metrics = {
        "case": cid, "Z": int(Z), "X": int(X), "A": int(A), "Y": int(Y), "H": int(H),
        "ssim": round(ssim, 6), "mse":  round(mse,  6), "ec":   round(ec,   6),
    }
    with open(out_dir / f"{cid}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[done] {cid} saved: sino_opt{so_vol.shape}, recon_opt{ro_vol.shape}  "
          f"SSIM={ssim:.4f} MSE={mse:.4e} EC={ec:.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="results/ckpt/best_script.pt")
    ap.add_argument("--sino_dir", type=str, default="data/test_sino")
    ap.add_argument("--voxel_dir", type=str, default="data/test_voxel")
    ap.add_argument("--out_dir", type=str, default="results/infer")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--clamp_min", type=float, default=None)
    ap.add_argument("--clamp_max", type=float, default=None)
    args = ap.parse_args()

    sino_dir = Path(args.sino_dir)
    voxel_dir = Path(args.voxel_dir)
    if not voxel_dir.exists():
        alt = Path("data/test_voxel")  # fix typo: 'text_voxel' -> 'test_voxel'
        if alt.exists():
            print(f"[info] using voxel_dir fallback: {alt}")
            voxel_dir = alt

    out_dir = Path(args.out_dir)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )

    model = load_model(Path(args.model), device)
    pairs = find_pair_paths(sino_dir, voxel_dir)
    assert len(pairs) > 0, f"No pairs found in {sino_dir} and {voxel_dir}"

    agg = []
    for cid, sp, vp in pairs:
        m = infer_case(model, cid, sp, vp, out_dir, device, args.clamp_min, args.clamp_max)
        agg.append(m)

    if agg:
        avg = {k: float(np.mean([d[k] for d in agg])) for k in ["ssim","mse","ec"]}
        with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump({"num_cases": len(agg), "avg": avg}, f, ensure_ascii=False, indent=2)
        print(f"[summary] cases={len(agg)}  SSIM={avg['ssim']:.4f}  "
              f"MSE={avg['mse']:.4e}  EC={avg['ec']:.4f}")


if __name__ == "__main__":
    main()
