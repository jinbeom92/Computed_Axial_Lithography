from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class ZSlicePairDataset(Dataset):

    def __init__(
        self,
        sino_dir: str | Path = "dataset/sino",
        voxel_dir: str | Path = "dataset/voxel",
        sino_suffix: str = "_sino.npy",
        voxel_suffix: str = "_voxel.npy",
        dtype: torch.dtype = torch.float32,
        require_all: bool = True,
    ) -> None:
        self.sino_dir = Path(sino_dir)
        self.voxel_dir = Path(voxel_dir)
        self.sino_suffix = sino_suffix
        self.voxel_suffix = voxel_suffix
        self.dtype = dtype

        self.pairs: List[Tuple[str, Path, Path, int]] = []
        self.index: List[Tuple[int, int]] = []

        self._discover_pairs(require_all)
        self._build_index()

        if not self.pairs:
            raise FileNotFoundError(
                f"No pairs found under {self.sino_dir} and {self.voxel_dir} "
                f"with suffixes ({self.sino_suffix}, {self.voxel_suffix})."
            )

    def _discover_pairs(self, require_all: bool) -> None:
        sino_files = sorted(self.sino_dir.glob(f"*{self.sino_suffix}"))
        if not sino_files:
            return

        for s_path in sino_files:
            name = s_path.name
            if not name.endswith(self.sino_suffix):
                continue
            case_id = name[: -len(self.sino_suffix)]
            v_path = self.voxel_dir / f"{case_id}{self.voxel_suffix}"
            if not v_path.exists():
                if require_all:
                    raise FileNotFoundError(f"Missing voxel file for case '{case_id}': {v_path}")
                continue

            s_mm = np.load(s_path, mmap_mode="r")
            v_mm = np.load(v_path, mmap_mode="r")

            if s_mm.ndim != 3 or v_mm.ndim != 3:
                raise ValueError(
                    f"Invalid ndim for case '{case_id}'. "
                    f"Expected sino(X,A,Z) & voxel(X,Y,Z); got {s_mm.shape} and {v_mm.shape}."
                )

            z_s, z_v = s_mm.shape[2], v_mm.shape[2]
            if z_s != z_v:
                raise ValueError(
                    f"Z mismatch for case '{case_id}': sino Z={z_s}, voxel Z={z_v}."
                )

            self.pairs.append((case_id, s_path, v_path, z_s))

    def _build_index(self) -> None:
        for i, (_, _, _, Z) in enumerate(self.pairs):
            self.index.extend((i, z) for z in range(Z))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | str]:
        pair_idx, z = self.index[idx]
        case_id, s_path, v_path, _ = self.pairs[pair_idx]

        s_mm = np.load(s_path, mmap_mode="r")
        v_mm = np.load(v_path, mmap_mode="r")

        sino_za = s_mm[:, :, z]
        voxel_zy = v_mm[:, :, z]

        sino_arr  = np.array(sino_za,  copy=True, dtype=np.float32, order="C")
        voxel_arr = np.array(voxel_zy, copy=True, dtype=np.float32, order="C")
        sino_t  = torch.from_numpy(sino_arr).unsqueeze(0)
        voxel_t = torch.from_numpy(voxel_arr).unsqueeze(0)
        if sino_t.dtype != self.dtype:   sino_t  = sino_t.to(self.dtype)
        if voxel_t.dtype != self.dtype:  voxel_t = voxel_t.to(self.dtype)

        return {
            "sino": sino_t,
            "voxel": voxel_t,
            "z": z,
            "case": case_id,
        }

    @property
    def cases(self) -> Sequence[str]:
        return [c for c, _, _, _ in self.pairs]

    def __repr__(self) -> str:
        n_cases = len(self.pairs)
        n_items = len(self.index)
        return f"{self.__class__.__name__}(cases={n_cases}, items={n_items}, sino_dir={self.sino_dir}, voxel_dir={self.voxel_dir})"
