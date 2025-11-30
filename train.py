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
from dataset.dataset import ZSlicePairDataset
from models.mdc import MDC
from opt.opt import build_adamw
from losses.ssim import SSIMLoss
from losses.mse import MSELoss
from losses.ec import ContrastLoss
from losses.psf import PSFLoss
from losses.tv import TVLoss
from losses.gdl import GDLLoss



def load_cfg(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)
    
def as_inside_mask(voxel: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(voxel):
        return (voxel > 0.5).to(torch.float32)
    return (voxel == 1).to(torch.float32)


def make_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    ds = ZSlicePairDataset(
        sino_dir=cfg["data"]["sino_dir"],
        voxel_dir=cfg["data"]["voxel_dir"]
    )
    nw = int(cfg["data"]["num_workers"])
    dl_args = dict(
        batch_size=cfg["data"]["batch_size"],
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)) if nw > 0 else None,
    )

    if bool(cfg["train"].get("val_copy", False)):
        idx_all = list(range(len(ds)))
        train_ds = Subset(ds, idx_all)
        val_ds   = Subset(ds, idx_all)
        return (
            DataLoader(train_ds, shuffle=False, **dl_args),
            DataLoader(val_ds,   shuffle=False, **dl_args),
        )

    n_total   = len(ds)
    val_split = float(cfg["train"]["val_split"])
    n_val     = max(1, min(n_total - 1, int(round(n_total * val_split)))) if n_total >= 2 else 0
    n_train   = n_total - n_val

    if n_train == 0 or n_val == 0:
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
    header = ["epoch","train_loss","val_loss","train_ssim","val_ssim","train_mse","val_mse","train_ec","val_ec"]
    row = [epoch, tr["loss"], va["loss"], tr["ssim"], va["ssim"], tr["mse"], va["mse"], tr["ec"], va["ec"]]
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        formatted = [f"{x:.6f}" if isinstance(x, float) else x for x in row]
        w.writerow(formatted)

class LiveBPViewer:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.im = None
        plt.ion()

    def update(self, img_tensor: torch.Tensor, title: str = "BP Preview") -> None:
        img = img_tensor.detach().float().cpu().numpy()
        if self.fig is None:
            self.fig, self.ax = plt.subplots(num="BP (Backprojection) Preview", figsize=(5, 5))
            self.im = self.ax.imshow(img, cmap="CMRmap", origin="lower")
            self.ax.set_title(title)
            self.fig.colorbar(self.im, ax=self.ax)
        else:
            self.im.set_data(img)
            self.im.set_clim(vmin=float(img.min()), vmax=float(img.max()))
            self.ax.set_title(title)

        self.fig.canvas.draw_idle()
        plt.pause(1)

def build_model(cfg: Dict) -> MDC:
    mcfg = cfg["model"]
    bpcfg = cfg["bp"]
    return MDC(
        c1d=mcfg["c1d"],
        c2d=mcfg["c2d"],
        fuse_out=mcfg["fusion_out"],
        align_out=mcfg["align_out"],
        cheat=mcfg["cheat"]["enabled"],
        cheat_out=mcfg["cheat"]["out_ch"],
        dec_hidden=mcfg["decoder_hidden"],
        bp_filter=bpcfg["filter"],
        bp_out=bpcfg["output_size"],
        bp_angle_chunk=int(bpcfg.get("angle_chunk", 0)),
    )


def build_losses(cfg: Dict, device: torch.device) -> Dict[str, nn.Module]:
    vw = float(cfg.get("losses", {}).get("void_weight", 0.05))
    ssim_cfg = cfg.get("ssim", {})
    ssim_loss = SSIMLoss(
        window_size=int(ssim_cfg.get("window_size", 11)),
        sigma=float(ssim_cfg.get("sigma", 1.5)),
        data_range=float(ssim_cfg.get("data_range", 1.0)),
        K1=float(ssim_cfg.get("K1", 0.01)),
        K2=float(ssim_cfg.get("K2", 0.03)),
        boundary_value=float(ssim_cfg.get("boundary_value", 0.81)),
        void_weight=vw,
    ).to(device)

    mse_loss = MSELoss(boundary_value=float(ssim_cfg.get("boundary_value", 0.81)), void_weight=vw).to(device)

    ec_loss = ContrastLoss(void_weight=vw).to(device)
    
    psf_loss = PSFLoss(
        data_range=float(ssim_cfg.get("data_range", 1.0)),
        void_weight=vw,
    ).to(device)
    
    tv_loss = TVLoss(void_weight=vw).to(device)
    
    gdl_loss = GDLLoss(void_weight=vw).to(device)

    return {"ssim": ssim_loss, "mse": mse_loss, "ec": ec_loss, "psf": psf_loss, "tv": tv_loss, "gdl": gdl_loss}


def to_device(batch: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    return batch["sino"].to(device, non_blocking=True), batch["voxel"].to(device, non_blocking=True)


def run_epoch(
    model: MDC,
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
    crit_psf = losses["psf"]
    crit_tv = losses["tv"]
    crit_gdl = losses["gdl"]

    pbar = tqdm(loader, desc="Train" if train else "Val", leave=False)
    
    for batch in pbar:
        sino, voxel_raw = to_device(batch, device)
        voxel = as_inside_mask(voxel_raw)
        cheat_in = voxel if use_cheat else None
        sino_opt, recon_opt = model(sino, cheat_in)

        l_ssim = crit_ssim(recon_opt, voxel)
        l_mse  = crit_mse(recon_opt, voxel)
        l_ec   = crit_ec(recon_opt, voxel)
        l_psf = crit_psf(recon_opt, voxel)
        l_tv = crit_tv(recon_opt, voxel)
        l_gdl = crit_gdl(recon_opt, voxel)

        total = (w["w_ssim"] * l_ssim + w["w_mse"] * l_mse + w["w_ec"] * l_ec +
                 w.get("w_pfs",0.1) * l_psf + w.get("w_tv",0.05) * l_tv + w.get("w_gdl",0.05) * l_gdl)

        if train:
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()

        B = sino.size(0)
        n_seen += B
        meter["loss"] += total.item() * B
        meter["ssim"] += (1.0 - l_ssim.item()) * B
        meter["mse"]  += l_mse.item() * B
        meter["ec"]   += (1.0 - 2.0 * l_ec.item()) * B

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

        if example_voxel is None:
            B, _, X, A = ex_s.shape
            ex_v = torch.zeros(B, 1, X, X, dtype=ex_s.dtype)
        else:
            ex_v = example_voxel.detach().cpu()

        scripted = torch.jit.trace(m, (ex_s, ex_v), strict=False)
        return scripted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = torch.device("cuda" if (cfg["train"]["device"] == "auto" and torch.cuda.is_available()) else cfg["train"]["device"])

    train_loader, val_loader = make_loaders(cfg)

    model = build_model(cfg).to(device)
    opt = build_adamw(
        model,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
        eps=cfg["optim"]["eps"],
        fused=None,
    )

    losses = build_losses(cfg, device)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, 
        T_max=cfg["train"]["epochs"],
        eta_min=1e-6
    )

    best_val = float("inf")
    ckpt_dir = Path(cfg["save"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = ckpt_dir / "train_log.csv"
    viewer = LiveBPViewer()
    
    print(f"[Scheduler Config]")
    print(f"  Initial LR: {cfg['optim']['lr']:.2e}")
    print(f"  Min LR (eta_min): {scheduler.eta_min:.2e}")
    print(f"  T_max (total epochs): {scheduler.T_max}")
    print(f"  Schedule: CosineAnnealingLR")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr = run_epoch(model, train_loader, opt, cfg, device, train=True, losses=losses)
        va = run_epoch(model, val_loader, opt, cfg, device, train=False, losses=losses)

        print(f"[Epoch {epoch:03d}] "
              f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
              f"train_ssim={tr['ssim']:.4f} val_ssim={va['ssim']:.4f} "
              f"train_mse={tr['mse']:.4f} val_mse={va['mse']:.4f} "
              f"train_ec={tr['ec']:.4f} val_ec={va['ec']:.4f}")
        
        append_metrics_csv(csv_path, epoch, tr, va)

        current_lr_before = opt.param_groups[0]['lr']
        scheduler.step()
        current_lr_after = opt.param_groups[0]['lr']
        
        print(f"  [Scheduler] Before: {current_lr_before:.2e} → After: {current_lr_after:.2e}")

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

        try:
            scripted = export_torchscript(model, ex_sino, ex_vox)
            scripted.save(str(ckpt_dir / "last_script.pt"))
        except Exception as e:
            print(f"[export] TorchScript save failed: {e}")

        if va["loss"] < best_val:
            import numpy as np
            np.save(rf"C:\Users\enf31\Desktop\SK_Hynix_Project\results\total_recon\recon_opt_total.npy", ro.detach().cpu().numpy())
            best_val = va["loss"]
            try:
                scripted.save(str(ckpt_dir / "best_script.pt"))
            except Exception as e:
                print(f"[export] best TorchScript save failed: {e}")

if __name__ == "__main__":
    main()