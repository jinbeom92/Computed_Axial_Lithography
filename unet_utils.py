# 🛠️ 정규화 등 헬퍼 함수

import torch

def normalize_for_loss(tensor_batch: torch.Tensor) -> torch.Tensor:
    B = tensor_batch.shape[0]
    tensor_flat = tensor_batch.view(B, -1)
    b_min = torch.min(tensor_flat, dim=1, keepdim=True)[0].view(B,1,1,1)
    b_max = torch.max(tensor_flat, dim=1, keepdim=True)[0].view(B,1,1,1)
    epsilon = 1e-8
    return (tensor_batch - b_min) / (b_max - b_min + epsilon)

def normalize_by_max(tensor_batch: torch.Tensor) -> torch.Tensor:
    B = tensor_batch.shape[0]
    tensor_flat = tensor_batch.view(B, -1)
    
    # dim=1을 기준으로 각 샘플의 최댓값을 찾습니다.
    b_max = torch.max(tensor_flat, dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    epsilon = 1e-8 # 0으로 나누는 것을 방지
    
    # 각 샘플을 자신의 최댓값으로 나누어줍니다.
    return tensor_batch / (b_max + epsilon)