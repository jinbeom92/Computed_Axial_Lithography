import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PSFLoss"]


class PSFLoss(nn.Module):
    def __init__(
        self,
        data_range: float = 1.0,
        K1: float = 0.01,
        K2: float = 0.03,
        window_size: int = 11,
        sigma: float = 1.5,
        void_weight: float = 0.0,
    ):
        super().__init__()
        self.data_range = float(data_range)
        self.K1 = float(K1)
        self.K2 = float(K2)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.void_weight = float(void_weight)

        ker = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        ker[0, 0, 1, 1] = 0.0
        self.register_buffer("ker", ker)

    def _gaussian_window(self, device: torch.device) -> torch.Tensor:
        ks = self.window_size
        sigma = self.sigma
        coords = torch.arange(ks, dtype=torch.float32, device=device) - (ks - 1.0) / 2.0
        g1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        g1d = g1d / (g1d.sum() + 1e-12)
        g2d = torch.outer(g1d, g1d)
        g2d = g2d / (g2d.sum() + 1e-12)
        return g2d.view(1, 1, ks, ks)

    def _mse_component(self, R_hat: torch.Tensor, V_gt: torch.Tensor) -> torch.Tensor:
        mask_ring = (V_gt > 0.0).to(torch.float32)
        gt_mod = mask_ring.clone()
        
        m = (V_gt > 0.0)
        in_n = m.to(torch.float32)
        cnt_in = F.conv2d(in_n, self.ker.to(R_hat.device, R_hat.dtype), padding=1)
        outer_ring = (~m) & (cnt_in > 0)
        gt_mod.masked_fill_(outer_ring, 0.81)

        mask = gt_mod > 0.0
        diff2 = (R_hat - gt_mod) ** 2
        num = (diff2 * mask.to(R_hat.dtype)).sum(dim=(1, 2, 3))
        den = mask.sum(dim=(1, 2, 3)).to(R_hat.dtype).clamp_min(1.0)
        base = (num / den).mean()

        if self.void_weight > 0.0:
            void = (gt_mod <= 0.0).to(dtype=R_hat.dtype, device=R_hat.device)
            vden = void.sum(dim=(1, 2, 3)).to(R_hat.dtype).clamp_min(1.0)
            l_void = ((R_hat**2) * void).sum(dim=(1, 2, 3)) / vden
            return base + self.void_weight * l_void.mean()
        
        return base

    def _ssim_component(self, R_hat: torch.Tensor, V_gt: torch.Tensor) -> torch.Tensor:
        m = (V_gt > 0.0)
        in_n = m.to(torch.float32)
        cnt_in = F.conv2d(in_n, self.ker.to(R_hat.device, R_hat.dtype), padding=1)
        outer_ring = (~m) & (cnt_in > 0)
        tgt_mod = in_n.clone()
        tgt_mod.masked_fill_(outer_ring, 0.81)
        eval_mask = tgt_mod > 0.0

        x = R_hat.to(dtype=torch.float32)
        y = tgt_mod.to(dtype=torch.float32, device=x.device)
        m_float = eval_mask.to(dtype=torch.float32, device=x.device)

        mask_sum = m_float.sum(dim=(1, 2, 3), keepdim=True)
        m_eff = torch.where(mask_sum > 0.0, m_float, torch.ones_like(m_float))

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

        base = 1.0 - ssim_mean

        if self.void_weight > 0.0:
            void = (tgt_mod <= 0).to(dtype=x.dtype, device=x.device)
            vden = void.sum(dim=(1, 2, 3)).to(x.dtype).clamp_min(1.0)
            l_void = ((x**2) * void).sum(dim=(1, 2, 3)) / vden
            return (base + self.void_weight * l_void).mean()

        return base.mean()

    def forward(self, R_hat: torch.Tensor, V_gt: torch.Tensor) -> torch.Tensor:
        assert R_hat.dim() == 4 and V_gt.dim() == 4 and R_hat.shape == V_gt.shape
        assert R_hat.shape[1] == 1

        lambda_mse = 0.5
        lambda_ssim = 0.5

        l_mse = self._mse_component(R_hat, V_gt)
        l_ssim = self._ssim_component(R_hat, V_gt)

        return lambda_mse * l_mse + lambda_ssim * l_ssim