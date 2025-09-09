# ================== Numpy-only voxel & sino Z-slice dedup ==================
# IN  : data/original/voxel/*_voxel.npy , data/original/sino/*_sino.npy
# OUT : data/dedup/voxel/*_voxel_dedup.npy , data/dedup/sino/*_sino_dedup.npy
#
# Overview
# - This script removes duplicate Z-slices along the last axis for both voxel volumes
#   and matching sinograms.
# - "Duplicate" is defined as *exact equality* (np.array_equal) between 2D slices.
# - For each case <ID>, we:
#     1) Load voxel: (X, Y, Z)
#     2) Keep unique slices only -> save to data/dedup/voxel/<ID>_voxel_dedup.npy
#     3) Apply the same kept indices to the matching sinogram (X, A, Zs)
#        -> save to data/dedup/sino/<ID>_sino_dedup.npy
# - Z and Zs may differ; indices are clipped defensively to the sinogram’s length.
#
# Notes
# - This is an O(Z^2) comparison in the worst case (exact match check).
#   It is simple but may be slow for very large Z; consider hashing if needed.

import os, glob
import numpy as np

BASE_DIR         = "data"
ORIG_VOXEL_DIR   = os.path.join(BASE_DIR, "original", "voxel")
ORIG_SINO_DIR    = os.path.join(BASE_DIR, "original", "sino")
DEDUP_VOXEL_DIR  = os.path.join(BASE_DIR, "dedup", "voxel")
DEDUP_SINO_DIR   = os.path.join(BASE_DIR, "dedup", "sino")
os.makedirs(DEDUP_VOXEL_DIR, exist_ok=True)
os.makedirs(DEDUP_SINO_DIR,  exist_ok=True)

def dedup_z_slices_numpy_only(vol_xyz: np.ndarray):
    """
    Keep only unique Z-slices (exact duplicates removed).

    Args:
        vol_xyz (np.ndarray): 3D volume with shape (X, Y, Z).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - dedup_vol: (X, Y, Z') volume containing only unique slices
            - kept_idx : (Z',) int array of original Z-indices that were kept

    Implementation detail:
        - Two slices are considered identical if np.array_equal(sliceA, sliceB) is True.
        - This is a strict equality check; no tolerance/threshold is used.
    """
    assert vol_xyz.ndim == 3, f"expected 3D, got {vol_xyz.shape}"
    Z = vol_xyz.shape[-1]

    unique_slices = []  # list of 2D arrays kept as unique references
    kept_idx = []       # list of original z-indices corresponding to unique_slices

    for z in range(Z):
        sl = vol_xyz[..., z]
        exists = False
        # Linear scan: check if this slice matches any already kept slice
        for ref in unique_slices:
            if np.array_equal(sl, ref):  # exact match -> consider as duplicate
                exists = True
                break
        if not exists:
            unique_slices.append(sl.copy())  # store a copy of unique slice
            kept_idx.append(z)

    # Stack unique slices back along Z; if none, produce (X, Y, 0)
    dedup_vol = np.stack(unique_slices, axis=-1) if unique_slices else vol_xyz[..., :0]
    return dedup_vol, np.asarray(kept_idx, dtype=np.int32)

# Traverse original voxel files and produce 1:1 dedup outputs
in_files = sorted(glob.glob(os.path.join(ORIG_VOXEL_DIR, "*_voxel.npy")))
print(f"[INFO] found {len(in_files)} voxel files in {ORIG_VOXEL_DIR}")

for vpath in in_files:
    # Derive the case base name by removing the suffix "_voxel.npy"
    base = os.path.basename(vpath)[:-10]
    vol  = np.load(vpath)           # expected shape: (X, Y, Z)
    Z    = vol.shape[-1]

    # 1) Deduplicate voxel along Z (numpy-only, exact equality)
    dedup_vol, kept_idx = dedup_z_slices_numpy_only(vol)
    Zp = dedup_vol.shape[-1]
    out_voxel = os.path.join(DEDUP_VOXEL_DIR, f"{base}_voxel_dedup.npy")
    np.save(out_voxel, dedup_vol)

    # 2) Apply the same kept indices to the matching sinogram (keep 1:1 pairing)
    spath = os.path.join(ORIG_SINO_DIR, f"{base}_sino.npy")
    if os.path.exists(spath):
        sino = np.load(spath)       # expected shape: (X, A, Zs)
        Zs = sino.shape[-1]

        # Defensive clipping in case Zs != Z (e.g., preprocessing mismatch)
        kept_safe = kept_idx[kept_idx < Zs]
        dedup_sino = sino[..., kept_safe] if kept_safe.size else sino[..., :0]

        out_sino = os.path.join(DEDUP_SINO_DIR, f"{base}_sino_dedup.npy")
        np.save(out_sino, dedup_sino)

        print(f"[OK] {base}: voxel {Z}->{Zp} | sino {Zs}->{dedup_sino.shape[-1]}  | saved: {out_voxel}, {out_sino}")
    else:
        print(f"[OK] {base}: voxel {Z}->{Zp} | sino missing → skipped  | saved: {out_voxel}")

print("\n[SUMMARY] done.")
