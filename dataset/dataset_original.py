import os, glob, csv, traceback
import numpy as np
import trimesh
import vamtoolbox as vam
from typing import Optional

# ================== Topology / Repair Options ==================
TRUST_TOPOLOGY     = True   # If True, assume STL is watertight (recommended).
FIX_SMALL_GAPS_3D  = False  # If True, apply 3D closing to fill small gaps (default False to preserve holes).

# --- Paths ---
STL_DIR     = r"C:\Users\ICTCOC\Downloads\stl_data_t"   # Input STL directory
VOXEL_DIR   = "data_t1/original/voxel"                  # Output directory for voxel npy files
SINO_DIR    = "data_t1/original/sino"                   # Output directory for sino npy files
LOG_FILE    = "data_t1/process_log.csv"                 # Log file path
ERROR_LOG   = "data_t1/error_log.txt"                   # Error log path

RESOLUTION = 256      # XY resolution for voxelization
NUM_ANGLES = 180      # Number of projection angles
ANGLES_DEG = np.linspace(0, 180, NUM_ANGLES, endpoint=False).astype(np.float32)

# ===== Preprocessing Parameters =====
# (1) Skip small/multipart objects
MIN_PART_FACES = 8                   # Minimum number of faces to keep a mesh part
PREVIEW_RES    = min(RESOLUTION, 96) # Low-res voxelization preview for component counting
MIN_COMP_ABS   = 128                 # Minimum component size (absolute voxel count)
MIN_COMP_FRAC  = 0.005               # Minimum component size (relative, >= 0.5% of total voxels)
DEBUG_PARTS    = False               # If True, print debug info for part counts

# (2) Thin object filter
OUT_OF_PART_VALUE = 0.3              # Skip if Z / avg(X,Y) < threshold

# (3) Field of view (FOV) safety margins
FOV_MARGIN_VOX       = 8     # XY margin in voxels
FOV_SAFETY           = 2     # Safety margin on radius
DIAMETER_MARGIN_VOX  = 16    # Diameter margin (pixels)

# --- Prepare output directories and log files ---
os.makedirs(VOXEL_DIR, exist_ok=True)
os.makedirs(SINO_DIR,  exist_ok=True)

if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)
if os.path.exists(ERROR_LOG):
    os.remove(ERROR_LOG)

log_f = open(LOG_FILE, "a", newline="", encoding="utf-8")
log_w = csv.writer(log_f)
log_w.writerow(["original_filename", "voxel_path", "sino_path"])

err_f = open(ERROR_LOG, "a", encoding="utf-8")

# --- Import CUDA projector from vamtoolbox ---
try:
    from vamtoolbox.projector.Projector2DParallelCUDA import Projector2DParallelCUDAAstra
except ImportError:
    print("❌ Error: Cannot import Projector2DParallelCUDAAstra from vamtoolbox.")
    log_f.close(); err_f.close()
    raise

# ----------------- Utility Functions -----------------
def _voxel_solid_matrix(mesh: trimesh.Trimesh,
                        pitch: float = 1.0,
                        trust_topology: bool = TRUST_TOPOLOGY,
                        fix_small_gaps_3d: bool = FIX_SMALL_GAPS_3D) -> np.ndarray:
    """
    Convert STL mesh into a solid voxel volume (interior filled).
    - If trust_topology=False, attempt lightweight mesh repair.
    - Fill using winding rule to preserve topology.
    - Optionally apply 3D binary closing to fill gaps.
    Returns float32 array {0.0, 1.0}.
    """
    m = mesh.copy()
    if not trust_topology:
        try:
            import trimesh.repair as T
            T.fill_holes(m); T.fix_inversion(m)
            m.remove_degenerate_faces(); m.remove_duplicate_faces()
            m.remove_unreferenced_vertices(); m.merge_vertices()
        except Exception:
            pass

    vg = m.voxelized(pitch=pitch)
    try:
        vg = vg.fill(method='winding')
    except Exception:
        vg = vg.fill()

    solid = vg.matrix.astype(np.float32)
    if fix_small_gaps_3d:
        from scipy.ndimage import binary_closing
        solid = binary_closing(solid > 0.5, structure=np.ones((3,3,3), bool)).astype(np.float32)
    return (solid > 0.5).astype(np.float32)

def voxelize_bool_preview(mesh_or_path, xy_resolution: int) -> np.ndarray:
    """
    Quick voxelization preview (bool volume) for estimating component count.
    """
    mesh = mesh_or_path if isinstance(mesh_or_path, trimesh.Trimesh) else trimesh.load(mesh_or_path, force='mesh')
    if mesh.is_empty or max(mesh.extents[:2]) <= 1e-6:
        return np.zeros((1,1,1), dtype=bool)
    extents = mesh.extents
    scale = xy_resolution / max(extents[:2])
    z_resolution = max(1, int(round(extents[2] * scale)))
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.scale_matrix(scale))
    solid = _voxel_solid_matrix(m, pitch=1.0)
    final = np.zeros((xy_resolution, xy_resolution, z_resolution), dtype=np.float32)
    copy_slice = tuple(slice(0, min(a, b)) for a, b in zip(solid.shape, final.shape))
    final[copy_slice] = solid[copy_slice]
    return (final > 0.5)

def _count_components_binary(vol_bool: np.ndarray, min_size: int) -> int:
    """
    Count number of connected components in a binary 3D volume.
    Only count components >= min_size.
    Uses SciPy label if available, otherwise fallback BFS.
    """
    try:
        from scipy.ndimage import label
        structure = np.ones((3,3,3), dtype=bool)
        labels, n = label(vol_bool, structure=structure)
        if n == 0: return 0
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        return int((sizes >= min_size).sum())
    except Exception:
        # BFS fallback
        visited = np.zeros_like(vol_bool, dtype=bool)
        idxs = np.argwhere(vol_bool)
        count = 0
        neigh = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=int)
        X,Y,Z = vol_bool.shape
        for sx,sy,sz in idxs:
            if visited[sx,sy,sz]: continue
            stack = [(int(sx),int(sy),int(sz))]
            visited[sx,sy,sz] = True
            size = 0
            while stack:
                x,y,z = stack.pop()
                size += 1
                for dx,dy,dz in neigh:
                    nx,ny,nz = x+dx, y+dy, z+dz
                    if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z:
                        if vol_bool[nx,ny,nz] and not visited[nx,ny,nz]:
                            visited[nx,ny,nz] = True
                            stack.append((nx,ny,nz))
            if size >= min_size:
                count += 1
        return count

def _measure_r_max_xy(vol: np.ndarray) -> float:
    """Measure maximum radius in XY plane relative to center."""
    X, Y, Z = vol.shape
    idx = np.argwhere(vol > 0.5)
    if idx.size == 0: return 0.0
    cx, cy = (X-1)/2.0, (Y-1)/2.0
    dx = idx[:,0] - cx; dy = idx[:,1] - cy
    return float(np.sqrt((dx*dx + dy*dy).max()))

def _center_embed_xy(final_shape, m):
    """Embed smaller volume m into the center of a larger zero-background volume."""
    FX, FY, FZ = final_shape
    MX, MY, MZ = m.shape
    out = np.zeros((FX, FY, MZ), dtype=np.float32)
    xs = max(0, (FX - MX)//2); ys = max(0, (FY - MY)//2)
    out[xs:xs+MX, ys:ys+MY, :MZ] = m
    return out

def voxelize_radius_safe(
    stl_path: str,
    xy_resolution: int,
    margin_vox: int = 8,
    safety: int = 2,
    diam_margin_vox: Optional[int] = None,
    fudge: float = 0.98
):
    """
    STL -> voxelization (X,Y,Z) with safety margin.
    - Scale mesh so XY fits within resolution - 2*margin_vox.
    - Embed into centered volume.
    - If radius exceeds allowed limit, apply shrink factor.
    Returns (volume, info_dict).
    """
    if 2*margin_vox >= xy_resolution:
        raise ValueError("margin_vox too large for resolution.")

    mesh = trimesh.load(stl_path, force='mesh')
    if mesh.is_empty or max(mesh.extents[:2]) <= 1e-6:
        raise ValueError("Invalid mesh.")

    # First scaling
    extents = mesh.extents
    base_scale = (xy_resolution - 2*margin_vox) / max(extents[:2])
    m1 = mesh.copy()
    m1.apply_transform(trimesh.transformations.scale_matrix(base_scale))
    v1 = _voxel_solid_matrix(m1, pitch=1.0)
    vol1 = _center_embed_xy((xy_resolution, xy_resolution, v1.shape[2]), v1)

    # Radius check
    r_max = _measure_r_max_xy(vol1)
    r_allow = (xy_resolution/2.0) - safety
    rad_margin = (diam_margin_vox/2.0) if diam_margin_vox else 0.0
    r_target = r_allow - rad_margin

    info = {"scaled_once": True, "auto_shrink": False}
    if r_max > r_target:
        shrink = (r_target / max(r_max, 1e-6)) * fudge
        m2 = mesh.copy()
        m2.apply_transform(trimesh.transformations.scale_matrix(base_scale * shrink))
        v2 = _voxel_solid_matrix(m2, pitch=1.0)
        vol2 = _center_embed_xy((xy_resolution, xy_resolution, v2.shape[2]), v2)
        info["auto_shrink"] = True
        return vol2, info
    return vol1, info

def _center_of_mass_xy(vol: np.ndarray):
    """Compute center-of-mass offset in XY plane (for logging)."""
    X, Y, Z = vol.shape
    idx = np.argwhere(vol > 0.5)
    if idx.size == 0: return (0.0, 0.0, 0.0)
    cx, cy = (X-1)/2.0, (Y-1)/2.0
    meanx, meany = float(idx[:,0].mean()), float(idx[:,1].mean())
    dx, dy = meanx - cx, meany - cy
    rad = _measure_r_max_xy(vol)
    return (dx, dy, rad)

def fp_2d_stack_with_vam(target_geo, angles_deg):
    """
    Forward-project voxel volume into sino stack.
    - Uses parallel beam geometry (CUDA).
    - Generates (H, A, Z) stack: H = detector pixels, A = angles, Z = slices.
    """
    proj_geo  = vam.geometry.ProjectionGeometry(angles=angles_deg, ray_type="parallel", CUDA=True)
    projector = Projector2DParallelCUDAAstra(target_geo, proj_geo)
    vol = target_geo.array
    s_list = [projector.forward(vol[..., zi]) for zi in range(vol.shape[2])]
    return np.stack(s_list, axis=2).astype(np.float32)

# ----------------- Main Processing Loop -----------------
for stl_path in sorted(glob.glob(os.path.join(STL_DIR, "*.stl"))):
    fname = os.path.basename(stl_path)
    fname_noext = os.path.splitext(fname)[0]
    try:
        # (A) Mesh validity and multipart check
        mesh = trimesh.load(stl_path, force='mesh')
        if mesh.is_empty:
            msg = f"[SKIP] {fname} empty mesh\n"
            err_f.write(msg); err_f.flush()
            print(msg.strip()); continue

        raw_parts = [p for p in mesh.split(only_watertight=False) if p.faces.size >= MIN_PART_FACES]
        if DEBUG_PARTS:
            print(f"[CHECK] {fname}: raw_parts={len(raw_parts)}")

        # (B) Component count check (preview voxelization)
        if len(raw_parts) > 1:
            preview = voxelize_bool_preview(mesh, PREVIEW_RES)
            nnz = int(preview.sum())
            if nnz == 0:
                msg = f"[SKIP] {fname} empty after preview voxelize\n"
                err_f.write(msg); err_f.flush()
                print(msg.strip()); continue

            thr = max(1, max(MIN_COMP_ABS, int(nnz * MIN_COMP_FRAC)))
            n_comp = _count_components_binary(preview, min_size=thr)
            if DEBUG_PARTS:
                print(f"[CHECK] {fname}: preview nnz={nnz}, components={n_comp}")
            if n_comp != 1:
                msg = f"[SKIP] {fname} skip_multipart(parts={len(raw_parts)}, comps={n_comp})\n"
                err_f.write(msg); err_f.flush()
                print(msg.strip()); continue

        # (C) Voxelization with FOV margin
        voxel_matrix, finfo = voxelize_radius_safe(
            stl_path, RESOLUTION,
            margin_vox=FOV_MARGIN_VOX,
            safety=FOV_SAFETY,
            diam_margin_vox=DIAMETER_MARGIN_VOX
        )
        X, Y, Z = voxel_matrix.shape
        z_ratio = Z / ((X + Y) / 2.0)
        if z_ratio < OUT_OF_PART_VALUE:
            msg = f"[SKIP] {fname} thin object (z_ratio={z_ratio:.3f})\n"
            err_f.write(msg); err_f.flush()
            print(msg.strip()); continue

        # (D) Logging center/radius status
        dx, dy, rmax = _center_of_mass_xy(voxel_matrix)
        r_allow = (RESOLUTION/2.0) - FOV_SAFETY
        rad_margin = r_allow - rmax
        diam_margin = 2.0 * rad_margin

        print(f"[INFO]  {fname}: shape={voxel_matrix.shape}, z_ratio={z_ratio:.3f}")
        print(f"[CENTER] COM offset=({dx:+.2f}, {dy:+.2f}) vox")
        print(f"[FOV]   r_allow={r_allow:.2f}, r_max={rmax:.2f}, "
              f"rad_margin={rad_margin:.2f}, diam_margin={diam_margin:.2f}")
        if finfo.get("auto_shrink", False):
            print(f"[AUTO]  shrink_factor={finfo['shrink_factor']:.4f}, r_max_after={finfo['r_max_after']:.2f}")

        # (E) Save voxel and sino outputs
        voxel_path = os.path.join(VOXEL_DIR, f"{fname_noext}_voxel.npy")
        np.save(voxel_path, voxel_matrix)

        target_geo = vam.geometry.TargetGeometry(target=voxel_matrix)
        sino = fp_2d_stack_with_vam(target_geo, ANGLES_DEG)
        sino_path = os.path.join(SINO_DIR, f"{fname_noext}_sino.npy")
        np.save(sino_path, sino)

        log_w.writerow([fname, voxel_path, sino_path])
        log_f.flush()
        print(f"[OK] {fname} → {os.path.basename(voxel_path)}, {os.path.basename(sino_path)}")

    except Exception as e:
        msg = f"{fname}: {str(e)}\n{traceback.format_exc()}\n"
        err_f.write(msg); err_f.flush()
        print(f"[ERROR] {fname} failed. Logged.")

log_f.close()
err_f.close()
print("\nAll preprocessing complete.")
