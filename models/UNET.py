# 🧠 U-Net 모델 구조 정의

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from unet_utils import normalize_for_loss, normalize_by_max
import physics.bp as bp

class ConvBNAct(nn.Module):
    """U-Net의 기본 구성 요소: Conv2d -> BatchNorm2d -> ReLU"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act="relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DownBlock(nn.Module):
    """U-Net의 인코더(내려가는) 경로를 구성하는 블록"""
    def __init__(self, in_ch, out_ch, act="relu"):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, act=act), 
            ConvBNAct(out_ch, out_ch, act=act)
        )
        self.downsampler = ConvBNAct(out_ch, out_ch, k=3, s=2, p=1, act=act) # stride = 2
    def forward(self, x):
        skip = self.block(x) # skip connection 생성 (원본 해상도)
        downsampled = self.downsampler(skip) # Strided Conv로 다운샘플링
        return downsampled, skip

class UpBlock(nn.Module):
    """U-Net의 디코더(올라가는) 경로를 구성하는 블록"""
    def __init__(self, in_ch_x: int, in_ch_skip: int, out_ch: int, act="relu"):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch_x, in_ch_x, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ConvBNAct(in_ch_x + in_ch_skip, out_ch, act=act),
            ConvBNAct(out_ch, out_ch, act=act)
            )
    def forward(self, x, skip):
        x = self.upsample(x)
        # 채널 차원을 기준으로 두 텐서를 연결
        return self.block(torch.cat([x, skip], dim=1))



class UNET(nn.Module):
    def __init__(self, c1d, c2d, fuse_out, align_out, cheat, cheat_out, dec_hidden, bp_filter, bp_out, bp_angle_chunk,
                depth: int = 5, act: str = "relu", noise_threshold: float = 0.0):
        super().__init__()
        self.noise_threshold = noise_threshold
        base_ch = cheat_out
        print(f"✅ SVTR (Adapter) initialized.")
        print(f"  - Using 'out_ch' from config ({cheat_out}) as base_ch.")
        print(f"  - Using 'depth' from config ({depth}).")
        print(f"  - Ignoring unused params like c1d, c2d, fuse_out, etc.")

        ch = [base_ch * (2 ** i) for i in range(depth)]

        # sino
        self.enc0 = nn.Sequential(ConvBNAct(1, ch[0], act=act), ConvBNAct(ch[0], ch[0], act=act))
        self.downs = nn.ModuleList([DownBlock(ch[i], ch[i+1], act=act) for i in range(depth-1)])
        self.bott = nn.Sequential(ConvBNAct(ch[-1], ch[-1], act=act), ConvBNAct(ch[-1], ch[-1], act=act))
        self.ups = nn.ModuleList([UpBlock(ch[i], ch[i], ch[i-1], act=act) for i in range(depth-1, 0, -1)])
        

        # Fusion
        self.fusions = nn.ModuleDict()

        fusion_levels = [f'enc{i}' for i in range(depth)] + ['bott'] + [f'dec{i}' for i in range(depth-1)]

        for level in fusion_levels:
            if 'enc' in level or 'bott' in level:
                idx = int(level.replace('enc','').replace('bott', str(depth-1)))
            else: # dec
                idx = depth - 2 - int(level.replace('dec',''))
            
        
        self.head = nn.Conv2d(ch[0], 1, kernel_size=1)

    def _forward_2d_chunk(self, sino_chunk_input: torch.Tensor) :
        # (B, A, V, 1) -> (B, 1, A, V) - Conv2d를 위한 차원 변경
        sino_input = sino_chunk_input.permute(0, 3, 1, 2)

        e0 = self.enc0(sino_input)
        skips = [e0]
        x_down = e0
        for down in self.downs: x_down, skip = down(x_down); skips.append(skip)

        x_bott = self.bott(x_down)
        skips.reverse()

        x_up = x_bott
        for i, up in enumerate(self.ups):
            skip = skips[i] # .pop() 대신 인덱스로 순서에 맞게 가져옵니다.
            x_up = up(x_up, skip)
        
        sino_opt = self.head(x_up)
        sino_opt = normalize_by_max(sino_opt)
        sino_opt = sino_opt - self.noise_threshold
        sino_opt = F.relu(sino_opt)


        sino_for_fbp = sino_opt.permute(0, 1, 3, 2)
        recon_chunk = bp.fbp2d(sino_for_fbp)
        R_hat_normalized = normalize_for_loss(recon_chunk)
        sino_hat_chunk = sino_opt.permute(0, 2, 3, 1)
        return sino_hat_chunk, R_hat_normalized
    
    def forward(self, sino: torch.Tensor, cheat_in: Optional[torch.Tensor] = None):
        """
        train.py가 호출하는 메인 forward 메소드.
        'cheat_in' 인자를 받기는 하지만, 내부 로직에서 사용하지 않고 무시합니다.
        """
        # 입력 텐서의 차원 조정
        if sino.ndim == 4 and sino.shape[1] == 1:
            sino = sino.squeeze(1)
        
        # (B, V, A) -> (B, A, V, 1)
        sino_input = sino.permute(0, 2, 1).unsqueeze(-1)

        # cheat_in(gt_image_batch) 없이 내부 로직 호출
        sino_hat_chunk, R_hat_chunk = self._forward_2d_chunk(sino_input)

        # (B, A, V, 1) -> (B, V, A)
        sino_hat_slices = sino_hat_chunk.squeeze(-1).permute(0, 2, 1)
        
        # train.py가 기대하는 2개의 출력을 반환
        return sino_hat_slices, R_hat_chunk