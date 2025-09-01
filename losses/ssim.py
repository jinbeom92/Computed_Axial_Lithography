# /mnt/data/ssim.py
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SSIMLoss"]


class SSIMLoss(nn.Module):
    """
    TorchScript-friendly masked SSIM loss (1 - SSIM).
    - Builds a modified target: inside=1.0, 1-pixel outer ring=`boundary_value`, else=0.0.
    - SSIM is averaged only over (modified target > 0) region; if empty, uses full image.
    Shape:
        R_hat, V_gt: [B, 1, H, W]
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        K1: float = 0.01,
        K2: float = 0.03,
        boundary_value: float = 0.8,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.data_range = float(data_range)
        self.K1 = float(K1)
        self.K2 = float(K2)
        self.boundary_value = float(boundary_value)

        ker = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        ker[0, 0, 1, 1] = 0.0  # 8-neighborhood, no center
        self.register_buffer("ker", ker)

    @torch.jit.export
    def outer_ring_target_and_mask(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mask: [B,1,H,W] values >0 treated as inside.
        Returns:
            tgt_mod: [B,1,H, W] float32 in {0, boundary_value, 1}
            eval_mask: [B,1,H,W] bool, where tgt_mod > 0
        """
        m = (mask > 0.0)
        inN = m.to(torch.float32)
        cnt_in = F.conv2d(inN, self.ker, padding=1)
        outer_ring = (~m) & (cnt_in > 0)
        tgt_mod = inN.clone()
        tgt_mod.masked_fill_(outer_ring, self.boundary_value)
        eval_mask = tgt_mod > 0.0
        return tgt_mod, eval_mask

    @torch.jit.export
    def _gaussian_window(self, device: torch.device) -> torch.Tensor:
        """Return [1,1,ks,ks] normalized 2D Gaussian kernel (float32)."""
        ks = self.window_size
        sigma = self.sigma
        coords = torch.arange(ks, dtype=torch.float32, device=device) - (ks - 1.0) / 2.0
        g1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        g1d = g1d / (g1d.sum() + 1e-12)
        g2d = torch.outer(g1d, g1d)
        g2d = g2d / (g2d.sum() + 1e-12)
        return g2d.view(1, 1, ks, ks)

    def forward(self, R_hat: torch.Tensor, V_gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            R_hat: [B,1,H,W] reconstruction (any dtype, casted to float32 internally).
            V_gt : [B,1,H,W] mask-like target (values >0 are inside).
        Returns:
            Scalar loss = mean over batch of (1 - masked SSIM).
        """
        assert R_hat.dim() == 4 and V_gt.dim() == 4 and R_hat.shape == V_gt.shape, \
            "R_hat and V_gt must be [B,1,H,W] and have the same shape."
        assert R_hat.shape[1] == 1, "Channel must be 1."

        x = R_hat.to(dtype=torch.float32)
        tgt_mod, m_bool = self.outer_ring_target_and_mask(V_gt)
        y = tgt_mod.to(dtype=torch.float32, device=x.device)
        m = m_bool.to(dtype=torch.float32, device=x.device)

        # If mask is empty for a sample, use full image for that sample.
        mask_sum = m.sum(dim=(1, 2, 3), keepdim=True)
        m_eff = torch.where(mask_sum > 0.0, m, torch.ones_like(m))

        w = self._gaussian_window(x.device)
        pad = self.window_size // 2
        eps = 1e-12

        msum = F.conv2d(m_eff, w, padding=pad)

        x_mean = F.conv2d(x * m_eff, w, padding=pad) / (msum + eps)
        y_mean = F.conv2d(y * m_eff, w, padding=pad) / (msum + eps)

        x2_mean = F.conv2d((x * x) * m_eff, w, padding=pad) / (msum + eps)
        y2_mean = F.conv2d((y * y) * m_eff, w, padding=pad) / (msum + eps)
        xy_mean = F.conv2d((x * y) * m_eff, w, padding=pad) / (msum + eps)

        sigma_x2 = (x2_mean - x_mean * x_mean).clamp_min(0.0)
        sigma_y2 = (y2_mean - y_mean * y_mean).clamp_min(0.0)
        sigma_xy = xy_mean - x_mean * y_mean

        L = self.data_range
        C1 = (self.K1 * L) * (self.K1 * L)
        C2 = (self.K2 * L) * (self.K2 * L)

        num = (2.0 * x_mean * y_mean + C1) * (2.0 * sigma_xy + C2)
        den = (x_mean * x_mean + y_mean * y_mean + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map = num / (den + eps)

        ssim_mean = (ssim_map * m_eff).sum(dim=(1, 2, 3)) / (m_eff.sum(dim=(1, 2, 3)) + eps)
        loss = 1.0 - ssim_mean
        return loss.mean()
