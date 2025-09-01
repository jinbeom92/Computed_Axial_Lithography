from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple
import copy, warnings
import csv
import matplotlib.pyplot as plt

import yaml
import torch
import torch.nn as nn
from torch.jit._trace import TracerWarning
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

# -----------------------------
# Imports (package- or flat-structure both supported)
# -----------------------------
from dataset.dataset import ZSlicePairDataset
from models.svtr import SVTR
from opt.opt import build_adamw
from losses.ssim import SSIMLoss
from losses.mse import MSELoss
from losses.ec import ContrastLoss


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)
    
def as_inside_mask(voxel: torch.Tensor) -> torch.Tensor:
    """
    Convert arbitrary GT labels to a binary inside mask in {0,1}.
    - If float: threshold at 0.5
    - If integer: treat value==1 as inside, others as outside (robust to {0,1} or {1,2})
    Returns: float32 mask with same shape as input.
    """
    if torch.is_floating_point(voxel):
        return (voxel > 0.5).to(torch.float32)
    # integer labels: 1 = inside, anything else = outside
    return (voxel == 1).to(torch.float32)


def make_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val loaders.
    If cfg['train']['val_copy'] == True, use the *same* dataset for both
    train and val (via two Subset views). Otherwise, split by val_split.
    """
    ds = ZSlicePairDataset(
        sino_dir=cfg["data"]["sino_dir"],
        voxel_dir=cfg["data"]["voxel_dir"]
    )

    dl_args = dict(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # --- copy mode: val = copy of train (same indices, separate loaders) ---
    if bool(cfg["train"].get("val_copy", False)):
        idx_all = list(range(len(ds)))
        train_ds = Subset(ds, idx_all)
        val_ds   = Subset(ds, idx_all)
        return (
            DataLoader(train_ds, shuffle=False, **dl_args),
            DataLoader(val_ds,   shuffle=False, **dl_args),
        )

    # --- default: split by val_split, but never allow empty splits ---
    n_total   = len(ds)
    val_split = float(cfg["train"]["val_split"])
    n_val     = max(1, min(n_total - 1, int(round(n_total * val_split)))) if n_total >= 2 else 0
    n_train   = n_total - n_val

    if n_train == 0 or n_val == 0:
        # fallback: copy mode if split would be empty
        idx_all = list(range(n_total))
        train_ds = Subset(ds, idx_all)
        val_ds   = Subset(ds, idx_all)
    else:
        g = torch.Generator().manual_seed(int(cfg["train"]["seed"]))
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    return (
        DataLoader(train_ds, shuffle=False, **dl_args),
        DataLoader(val_ds,   shuffle=False, **dl_args),
    )

def append_metrics_csv(csv_path: Path, epoch: int, tr: Dict[str, float], va: Dict[str, float]) -> None:
    """
    Append a row of metrics into a CSV file. Creates header if file doesn't exist.
    Columns: epoch, train_loss, val_loss, train_ssim, val_ssim, train_mse, val_mse, train_ec, val_ec
    """
    header = ["epoch","train_loss","val_loss","train_ssim","val_ssim","train_mse","val_mse","train_ec","val_ec"]
    row = [epoch, tr["loss"], va["loss"], tr["ssim"], va["ssim"], tr["mse"], va["mse"], tr["ec"], va["ec"]]
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        # format floats consistently
        formatted = [f"{x:.6f}" if isinstance(x, float) else x for x in row]
        w.writerow(formatted)

class LiveBPViewer:
    """
    Minimal live viewer for backprojection (recon_opt) using matplotlib.
    Keeps a single figure and updates it per epoch. Safe to call repeatedly.
    """
    def __init__(self):
        self.fig = None
        self.ax = None
        self.im = None
        plt.ion()  # interactive mode

    def update(self, img_tensor: torch.Tensor, title: str = "BP Preview") -> None:
        # img_tensor: (H, W) on any device
        img = img_tensor.detach().float().cpu().numpy()
        if self.fig is None:
            self.fig, self.ax = plt.subplots(num="BP (Backprojection) Preview", figsize=(5, 5))
            self.im = self.ax.imshow(img, cmap="CMRmap", origin="lower")
            self.ax.set_title(title)
            self.fig.colorbar(self.im, ax=self.ax)
        else:
            self.im.set_data(img)
            # auto-rescale color range for visibility
            self.im.set_clim(vmin=float(img.min()), vmax=float(img.max()))
            self.ax.set_title(title)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # non-blocking UI refresh

def build_model(cfg: Dict) -> SVTR:
    mcfg = cfg["model"]
    bpcfg = cfg["bp"]
    return SVTR(
        c1d=mcfg["c1d"],
        c2d=mcfg["c2d"],
        fuse_out=mcfg["fusion_out"],
        align_out=mcfg["align_out"],
        cheat=mcfg["cheat"]["enabled"],
        cheat_out=mcfg["cheat"]["out_ch"],
        dec_hidden=mcfg["decoder_hidden"],
        bp_filter=bpcfg["filter"],
        bp_out=bpcfg["output_size"],
    )


def build_losses(cfg: Dict, device: torch.device) -> Dict[str, nn.Module]:
    # Optional SSIM params in cfg; fallback to sensible defaults
    vw = float(cfg.get("losses", {}).get("void_weight", 0.0))
    ssim_cfg = cfg.get("ssim", {})
    ssim_loss = SSIMLoss(
        window_size=int(ssim_cfg.get("window_size", 11)),
        sigma=float(ssim_cfg.get("sigma", 1.5)),
        data_range=float(ssim_cfg.get("data_range", 1.0)),
        K1=float(ssim_cfg.get("K1", 0.01)),
        K2=float(ssim_cfg.get("K2", 0.03)),
        boundary_value=float(ssim_cfg.get("boundary_value", 0.8)),
        void_weight=vw,
    ).to(device)

    # MSE uses the same boundary_value by default
    mse_loss = MSELoss(boundary_value=float(ssim_cfg.get("boundary_value", 0.8)), void_weight=vw).to(device)

    ec_loss = ContrastLoss(void_weight=vw).to(device)

    return {"ssim": ssim_loss, "mse": mse_loss, "ec": ec_loss}


def to_device(batch: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    return batch["sino"].to(device), batch["voxel"].to(device)


def run_epoch(
    model: SVTR,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    cfg: Dict,
    device: torch.device,
    train: bool,
    losses: Dict[str, nn.Module],
) -> Dict[str, float]:
    model.train(train)
    w = cfg["loss_weights"]
    use_cheat = cfg["model"]["cheat"]["enabled"] and (train or not cfg["model"]["cheat"]["train_only"])

    meter = {"loss": 0.0, "ssim": 0.0, "mse": 0.0, "ec": 0.0}
    n_seen = 0

    crit_ssim = losses["ssim"]
    crit_mse  = losses["mse"]
    crit_ec   = losses["ec"]

    pbar = tqdm(loader, desc="Train" if train else "Val", leave=False)
    for batch in pbar:
        sino, voxel_raw = to_device(batch, device)
        voxel = as_inside_mask(voxel_raw)
        cheat_in = voxel if use_cheat else None

        sino_opt, recon_opt = model(sino, cheat_in)  # (B,1,X,A), (B,1,H,H)

        # Losses
        l_ssim = crit_ssim(recon_opt, voxel)  # (1 - SSIM)
        l_mse  = crit_mse(recon_opt, voxel)
        l_ec   = crit_ec(recon_opt, voxel)    # returns loss = (1 - EC) * 0.5

        total = w["w_ssim"] * l_ssim + w["w_mse"] * l_mse + w["w_ec"] * l_ec

        if train:
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()

        B = sino.size(0)
        n_seen += B
        meter["loss"] += total.item() * B
        meter["ssim"] += (1.0 - l_ssim.item()) * B                    # report SSIM (higher better)
        meter["mse"]  += l_mse.item() * B
        meter["ec"]   += (1.0 - 2.0 * l_ec.item()) * B                # recover EC from loss=(1-EC)*0.5

        pbar.set_postfix(
            loss=meter["loss"] / n_seen,
            ssim=meter["ssim"] / n_seen,
            mse=meter["mse"] / n_seen,
            ec=meter["ec"] / n_seen,
        )

    for k in meter:
        meter[k] /= max(1, n_seen)
    return meter


def export_torchscript(model, example_sino, example_voxel):
    model.eval()
    try:
        scripted = torch.jit.script(model)
        print("[TS] script export OK")
        return scripted
    except Exception as e:
        print(f"[TS] script failed → fallback to trace(strict=False): {e}")
        warnings.filterwarnings("ignore", category=TracerWarning)
        m = copy.deepcopy(model).cpu().eval()
        ex_s = example_sino.detach().cpu()
        ex_v = example_voxel.detach().cpu() if example_voxel is not None else None
        scripted = torch.jit.trace(m, (ex_s, ex_v), strict=False)
        return scripted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = torch.device("cuda" if (cfg["train"]["device"] == "auto" and torch.cuda.is_available()) else cfg["train"]["device"])

    # Data
    train_loader, val_loader = make_loaders(cfg)

    # Model & Optimizer
    model = build_model(cfg).to(device)
    opt = build_adamw(
        model,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
        eps=cfg["optim"]["eps"],
        fused=None,
    )

    # Losses
    losses = build_losses(cfg, device)

    # Train
    best_val = float("inf")
    ckpt_dir = Path(cfg["save"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = ckpt_dir / "train_log.csv"
    viewer = LiveBPViewer()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr = run_epoch(model, train_loader, opt, cfg, device, train=True,  losses=losses)
        va = run_epoch(model, val_loader,   opt, cfg, device, train=False, losses=losses)

        print(f"[Epoch {epoch:03d}] "
              f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
              f"train_ssim={tr['ssim']:.4f} val_ssim={va['ssim']:.4f} "
              f"train_mse={tr['mse']:.4f} val_mse={va['mse']:.4f} "
              f"train_ec={tr['ec']:.4f} val_ec={va['ec']:.4f}")
        
        append_metrics_csv(csv_path, epoch, tr, va)

        # Save TorchScript (last)
        with torch.no_grad():
            try:
                ex = next(iter(val_loader))
            except StopIteration:
                ex = next(iter(train_loader))
            ex_sino = ex["sino"].to(device)
            ex_vox  = ex["voxel"].to(device) if cfg["model"]["cheat"]["enabled"] else None
            
            try:
                _, ro = model(ex_sino, ex_vox)
                viewer.update(ro[0, 0], title=f"BP Recon (epoch {epoch})")
            except Exception as e:
                print(f"[viz] skip BP preview: {e}")

            scripted = export_torchscript(model, ex_sino, ex_vox)
            scripted.save(str(ckpt_dir / "last_script.pt"))

        # Save best by val loss (fix: use 'loss' key, not a non-existent 'val_ssim')
        if va["loss"] < best_val:
            best_val = va["loss"]
            scripted.save(str(ckpt_dir / "best_script.pt"))

if __name__ == "__main__":
    main()
