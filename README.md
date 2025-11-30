# MDC: Multi-Dimensional Cheat Reconstruction Network

## Overview
MDC is a physics-informed tomographic reconstruction architecture that combines multi-dimensional encoding, voxel-domain hints, and a differentiable filtered back-projection (FBP) layer.  
It **achieves 46.38% improvement over OSMO**, while preserving spatial resolution and ensuring physical consistency.

## Key Features

- Dual Encoder Design  
  - Enc1D: Captures local line-profile details from each projection line.  
  - Enc2D: Learns global structures directly from the sinogram plane.  

- Fusion Module  
  - Combines Enc1D/Enc2D features with adaptive weighting.  

- CheatEnc2D  
  - Provides voxel-domain hints to stabilize alignment without interpolation.  

- ALign Module  
  - Aligns fused features with optional cheat injection.  

- Decoder  
  - Residual 2D blocks projecting to non-negative sinogram output.  

- Physics-informed FBP  
  - Differentiable inverse Radon layer ensures physical consistency.

## Processing Pipeline

1. Input  
   - Sinogram: (X, A, Z)  
   - Voxel: (X, Y, Z) slices  

2. Encoders  
   - Extract local (Enc1D) and global (Enc2D) features from the sinogram.  

3. Fusion + Align  
   - Fuse Enc1D/Enc2D features with optional voxel-domain cheat injection.  

4. Decoder  
   - Predict optimized sinogram: `sino_opt ≥ 0`.  

5. FBP  
   - Apply differentiable FBP → output normalized reconstruction: (H, H, Z).  

6. Loss  
   - MSE + SSIM + Contrast Loss for boundary and edge preservation.

## Inference

- TorchScript supported  
  - `torch.jit.script` / trace fallback  

- Batched 5D reconstruction  
  - Input: `(B, 1, X, A, Z)`  

- Outputs  
  - `sino_opt.npy`: optimized sinograms  
  - `recon_opt.npy`: final reconstructions  
  - Optional mid-slice visualizations

## Results

- Performance  
  - +46.38% over OSMO  

- Qualitative  
  - Sharper edges  
  - Reduced halo/ring artifacts  

- Efficiency  
  - Memory-efficient  
  - GPU-accelerated  
  - TorchScript-ready

## License & Citation

Copyright (c) 2025 **jinbeom92**  
All rights reserved. Redistribution or use without permission is prohibited.

If you use MDC in your work, please cite this repository.


---
## Architecture

<img width="926" height="801" alt="image" src="https://github.com/user-attachments/assets/db3aaf65-d286-42bb-8728-6444875313dd" />

---
