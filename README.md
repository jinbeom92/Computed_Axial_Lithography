# MDC: Multi-Dimensional Cheat Reconstruction Network

## Overview

**MDC (Multi-Dimensional Cheat)** is a novel physics-informed tomographic reconstruction architecture that combines
1D line-profile encoding, 2D global sinogram encoding, voxel-domain hints, and a differentiable filtered back-projection (FBP) layer.
This design achieves **46.38% performance improvement over OSMO** in our experiments, while preserving spatial resolution and ensuring physical consistency.

---

## Key Features

* **Dual Encoder Design**

  * **Enc1D**: Processes each projection line (X) independently for every angle (A), capturing fine **local line-profile details**.
  * **Enc2D**: Learns **global spatial structures** directly from the (X,A) sinogram plane.

* **Fusion Module**

  * Concatenates Enc1D/Enc2D features and projects them via learnable scalars (α₁, α₂) for adaptive weighting.

* **CheatEnc2D (Voxel-domain hint)**

  * Encodes XY-plane voxel slices.
  * Reduced along Y (mean) → broadcast along A, **avoiding interpolation artifacts**.
  * Provides a stable alignment anchor without information loss.

* **ALign Module**

  * Aligns fused sinogram features with optional cheat features using non-interpolating concatenation + residual refinement.

* **Decoder**

  * Two Residual2D blocks + 1×1 projection head.
  * Enforces non-negative **sino\_opt** output.

* **Physics-informed FBP**

  * Torch-only differentiable inverse Radon (FBP) layer with Hamming filter.
  * Ensures physical consistency between predicted sinograms and reconstructions.

* **Loss Functions**

  * **MSE** with boundary soft-target (0.81).
  * **SSIM** with masked boundary emphasis.
  * **Contrast Loss (EC)** to maximize internal-external edge contrast.
  * This combination stabilizes training and reduces overshoot/halo artifacts.

---

## Processing Pipeline

1. **Input**: Sinogram `(X,A,Z)`, Voxel `(X,Y,Z)` slices.
2. **Encoders**:

   * Enc1D (local line features)
   * Enc2D (global plane features)
   * CheatEnc2D (voxel XY hints, optional)
3. **Fusion + Align**: Combine Enc1D/Enc2D → Align with optional cheat injection.
4. **Decoder**: Predict **sino\_opt ≥ 0**.
5. **FBP**: Apply differentiable FBP → output normalized recon `(H,H,Z)`.
6. **Loss**: MSE + SSIM + EC.

---

## Inference

* **TorchScript support**:
  Models are exported with `torch.jit.script` or `trace` fallback.
* **5D fast path**: Batched reconstruction `(B,1,X,A,Z)`.
* **Fallback loop**: Per-slice inference if checkpoint traced on 4D only.
* **Outputs**:

  * `sino_opt.npy` (optimized sinograms)
  * `recon_opt.npy` (final reconstructions)
  * Mid-slice PNG visualizations for quick inspection.

---

## Differentiation from U-Net and Others

* **Unlike U-Net**:

  * No down/upsampling; **stride=1, same padding** preserves resolution.
  * MDC integrates physics (FBP) directly in the pipeline, ensuring interpretability.

* **Unlike OSMO / FBPConvNet**:

  * MDC optimizes the **sinogram first**, then reconstructs, instead of post-FBP image correction.
  * Achieved **46.38% improvement over OSMO** in our benchmarks.

* **Unique Contribution**:

  * **Multi-dimensional encoding** (1D+2D)
  * **Cheat injection without interpolation**
  * **Physics-informed reconstruction with boundary-aware loss design**

---

## Results

* **Performance**: +46.38% improvement compared to OSMO.
* **Qualitative**: Sharper edges, reduced halo/ring artifacts, improved structural fidelity.
* **Efficiency**: Memory-efficient, TorchScript-ready, GPU-accelerated.

---

## Citation

If you use MDC in your work, please cite:

---
## Architecture

<img width="926" height="801" alt="image" src="https://github.com/user-attachments/assets/db3aaf65-d286-42bb-8728-6444875313dd" />

---
