import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TVLoss"]


class TVLoss(nn.Module):
    def __init__(self, reduction: str = "mean", beta: float = 1.0, void_weight: float = 0.0):
        super().__init__()
        self.reduction = reduction
        self.beta = float(beta)
        self.void_weight = float(void_weight)

        ker = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        ker[0, 0, 1, 1] = 0.0
        self.register_buffer("ker", ker)

    def forward(self, R_hat: torch.Tensor, V_gt: torch.Tensor) -> torch.Tensor:
        assert R_hat.dim() == 4 and V_gt.dim() == 4 and R_hat.shape == V_gt.shape
        assert R_hat.shape[1] == 1

        x = R_hat

        diff_x = x[:, :, :-1, :] - x[:, :, 1:, :]
        diff_y = x[:, :, :, :-1] - x[:, :, :, 1:]

        tv_x = torch.sqrt(diff_x ** 2 + 1e-10)
        tv_y = torch.sqrt(diff_y ** 2 + 1e-10)

        tv_norm_x = tv_x.sum(dim=(2, 3))
        tv_norm_y = tv_y.sum(dim=(2, 3))

        base_tv = tv_norm_x.mean() + tv_norm_y.mean()

        if self.void_weight > 0.0:
            m = (V_gt > 0.0)
            in_n = m.to(torch.float32)
            cnt_in = F.conv2d(in_n, self.ker.to(x.device, x.dtype), padding=1)
            outer_ring = (~m) & (cnt_in > 0)
            gt_mod = in_n.clone()
            gt_mod.masked_fill_(outer_ring, 0.81)
            void = (gt_mod <= 0.0).to(dtype=x.dtype, device=x.device)
            
            void_tv = (torch.abs(diff_x[:, :, :, :]) * void[:, :, :-1, :]).sum(dim=(2, 3))
            void_tv = void_tv + (torch.abs(diff_y[:, :, :, :]) * void[:, :, :, :-1]).sum(dim=(2, 3))
            
            void_den = void.sum(dim=(2, 3)).clamp_min(1.0)
            l_void = void_tv.sum(dim=1) / void_den.sum(dim=1).clamp_min(1.0)
            
            return base_tv + self.void_weight * l_void.mean()

        return base_tv
