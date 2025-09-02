import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MSELoss"]


class MSELoss(nn.Module):

    def __init__(self, boundary_value: float = 0.8, void_weight: float = 0.0):
        super().__init__()
        self.boundary_value = float(boundary_value)
        self.void_weight = float(void_weight)
        ker = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        ker[0, 0, 1, 1] = 0.0  # 8-neighborhood, no center
        self.register_buffer("ker", ker)

    @torch.jit.export
    def outer_ring_target(self, mask: torch.Tensor) -> torch.Tensor:

        m = (mask > 0.0)
        inN = m.to(torch.float32)
        cnt_in = F.conv2d(inN, self.ker, padding=1)
        outer_ring = (~m) & (cnt_in > 0)
        gt_mod = inN.clone()
        # Fill the 1-pixel outer ring
        gt_mod.masked_fill_(outer_ring, self.boundary_value)
        return gt_mod

    def forward(self, R_hat: torch.Tensor, V_gt: torch.Tensor) -> torch.Tensor:

        assert R_hat.dim() == 4 and V_gt.dim() == 4 and R_hat.shape == V_gt.shape, \
            "R_hat and V_gt must be [B,1,H,W] and have the same shape."
        assert R_hat.shape[1] == 1, "Channel must be 1."

        gt_mod = self.outer_ring_target(V_gt).to(dtype=R_hat.dtype, device=R_hat.device)
        mask = gt_mod > 0.0

        diff2 = (R_hat - gt_mod) ** 2
        num = (diff2 * mask.to(dtype=R_hat.dtype)).sum(dim=(1, 2, 3))
        den = mask.sum(dim=(1, 2, 3)).to(dtype=R_hat.dtype).clamp_min(1.0)
        base = (num / den).mean()
        if self.void_weight > 0.0:
            ring = (gt_mod > 0) & (gt_mod < 1)
            void = (gt_mod <= 0.0).to(dtype=R_hat.dtype)
            vden = void.sum(dim=(1,2,3)).to(dtype=R_hat.dtype).clamp_min(1.0)
            l_void = ((R_hat**2) * void.to(R_hat.dtype)).sum(dim=(1,2,3)) / vden
            return base + self.void_weight * l_void.mean()
        return base
