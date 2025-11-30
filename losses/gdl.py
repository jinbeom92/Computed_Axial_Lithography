import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GDLLoss"]


class GDLLoss(nn.Module):
    def __init__(self, void_weight: float = 0.0):
        super().__init__()
        self.void_weight = float(void_weight)

        ker = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        ker[0, 0, 1, 1] = 0.0
        self.register_buffer("ker", ker)

    def _compute_gradients(self, x: torch.Tensor) -> tuple:
        grad_x = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :] + 1e-10)
        grad_y = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:] + 1e-10)
        return grad_x, grad_y

    def forward(self, R_hat: torch.Tensor, V_gt: torch.Tensor) -> torch.Tensor:
        assert R_hat.dim() == 4 and V_gt.dim() == 4 and R_hat.shape == V_gt.shape
        assert R_hat.shape[1] == 1

        pred_grad_x, pred_grad_y = self._compute_gradients(R_hat)
        
        m = (V_gt > 0.0)
        in_n = m.to(torch.float32)
        cnt_in = F.conv2d(in_n, self.ker.to(R_hat.device, R_hat.dtype), padding=1)
        outer_ring = (~m) & (cnt_in > 0)
        gt_mod = in_n.clone()
        gt_mod.masked_fill_(outer_ring, 0.81)

        target_grad_x, target_grad_y = self._compute_gradients(gt_mod)

        diff_grad_x = torch.abs(pred_grad_x - target_grad_x)
        diff_grad_y = torch.abs(pred_grad_y - target_grad_y)

        base_gdl = diff_grad_x.mean() + diff_grad_y.mean()

        if self.void_weight > 0.0:
            void = (gt_mod <= 0.0).to(dtype=R_hat.dtype, device=R_hat.device)
            
            void_diff = (diff_grad_x[:, :, :, :] * void[:, :, :-1, :]).sum(dim=(2, 3))
            void_diff = void_diff + (diff_grad_y[:, :, :, :] * void[:, :, :, :-1]).sum(dim=(2, 3))
            
            void_den = void.sum(dim=(2, 3)).clamp_min(1.0)
            l_void = void_diff.sum(dim=1) / void_den.sum(dim=1).clamp_min(1.0)
            
            return base_gdl + self.void_weight * l_void.mean()

        return base_gdl
