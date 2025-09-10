# 📦 데이터셋 및 데이터로더 정의

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class ZSlicePairDataset(Dataset):
    """
    여러 3D .npy 볼륨을 메모리 매핑으로 읽어와 2D 슬라이스 쌍을 반환하는 데이터셋.
    """
    def __init__(self, sino_dir, voxel_dir):
        sino_files = sorted(glob.glob(os.path.join(sino_dir, "*.npy")))
        voxel_files = sorted(glob.glob(os.path.join(voxel_dir, "*.npy")))

        if len(sino_files) != len(voxel_files) or len(sino_files) == 0:
            raise ValueError("sino/voxel 파일 수가 다르거나 파일이 없습니다.")

        self.sino_volumes = [np.load(f, mmap_mode='r') for f in sino_files]
        self.voxel_volumes = [np.load(f, mmap_mode='r') for f in voxel_files]
        
        self.slice_map = []
        for vol_idx, sino_vol in enumerate(self.sino_volumes):
            num_slices = sino_vol.shape[2]
            for slice_idx in range(num_slices):
                self.slice_map.append((vol_idx, slice_idx))
        
        print(f"총 {len(sino_files)}개의 3D 볼륨에서 {len(self.slice_map)}개의 2D 슬라이스를 발견했습니다.")

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.slice_map[idx]
        
        sino_slice = self.sino_volumes[vol_idx][:, :, slice_idx].copy()
        voxel_slice = self.voxel_volumes[vol_idx][:, :, slice_idx].copy()

        H, W = voxel_slice.shape
        center_y, center_x = (H - 1) / 2.0, (W - 1) / 2.0
        radius = min(H, W) / 2.0
        y_coords, x_coords = np.ogrid[:H, :W]
        dist_from_center_sq = (y_coords - center_y)**2 + (x_coords - center_x)**2
        mask = dist_from_center_sq > radius**2
        voxel_slice[mask] = 0.0

        sino_tensor = torch.from_numpy(sino_slice).float()
        voxel_tensor = torch.from_numpy(voxel_slice).float()

        return {
            "sino": sino_tensor,
            "voxel": voxel_tensor.unsqueeze(0)
        }