import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ContrastLoss"]


class ContrastLoss(nn.Module):

    def __init__(self, void_weight: float = 0.0):
        super().__init__()
        self.void_weight = float(void_weight)
        ker = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        ker[0, 0, 1, 1] = 0.0  # 8-neighborhood, no center
        self.register_buffer("ker", ker)

    def forward(self, R_hat: torch.Tensor, V_gt: torch.Tensor) -> torch.Tensor:

        assert R_hat.dim() == 4 and V_gt.dim() == 4 and R_hat.shape == V_gt.shape, \
            "R_hat and V_gt must be [B,1,H,W] and have the same shape."
        assert R_hat.shape[1] == 1, "Channel must be 1."

        x = R_hat
        m = (V_gt > 0.0)  # inside
        outN = (~m).to(dtype=x.dtype)

        ker = self.ker.to(device=x.device, dtype=x.dtype)

        # inner boundary: inside pixels with at least one outside neighbor
        cnt_out = F.conv2d(outN, ker, padding=1)
        ib = m & (cnt_out > 0)

        # maximum of outside region in 3x3 neighborhood
        x_out = x * outN
        max_out = F.max_pool2d(x_out, kernel_size=3, stride=1, padding=1)

        contrast = (x - max_out) * ib.to(dtype=x.dtype)

        num = ib.to(dtype=x.dtype).sum(dim=(2, 3))          # (B,1)
        sum_contrast = contrast.sum(dim=(2, 3))             # (B,1)

        ec = torch.where(num > 0.0, sum_contrast / num.clamp_min(1.0), torch.ones_like(sum_contrast))

        base = (1.0 - ec) * 0.5
        if self.void_weight > 0.0:
            m = (V_gt > 0.0)
            cnt_in = F.conv2d(m.to(x.dtype), self.ker.to(x.device, x.dtype), padding=1)
            ring_out = (~m) & (cnt_in > 0)
            out_strict = (~m) & (~ring_out)
            out_strict = out_strict.to(x.dtype)
            vden = out_strict.sum(dim=(2,3)).clamp_min(1.0)
            l_void = ((x**2) * out_strict).sum(dim=(2,3)) / vden
            return (base + self.void_weight * l_void).mean()
        return base.mean()
