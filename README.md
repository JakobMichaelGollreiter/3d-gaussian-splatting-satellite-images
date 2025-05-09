# 3d-gaussian-splatting-satellite-images
## Overview 
This project explores the application of 3D Gaussian Splatting (3DGS) for reconstructing 3D structures from sparse satellite imagery. Using the DFC2019 Track 3 dataset, we aim to generate accurate 3D reconstructions from limited multi-view satellite images (10-20 views per scene) with varying lighting and quality conditions.
## Dataset 
We use Track 3 of the 2019 IEEE GRSS Data Fusion Contest (DFC2019), which includes:
- 110 scenes with approximately 20 multi-angle satellite images per scene
- RGB images
- RPC camera metadata
- Digital Surface Models (DSMs) as ground truth heightmaps

## Results

This video demonstrates our  approach to 3D scene reconstruction that combines Visual Geometry Grounding Transformer (VGGT) with classical bundle adjustment optimization. 
Our method leverages VGGT's powerful single-pass capability to estimate camera parameters and generate 3D point clouds, then refines these results through our specialized bundle adjustment implementation. The key innovations include:
  - Confidence-based point selection for reliable reconstruction
  - Differentiable reprojection with robust loss functions
  - Multi-stage optimization prioritizing high-confidence points
  - Preservation of physical constraints throughout optimization
Click to watch the video on youtube
[![Click to watch the video on youtube](https://img.youtube.com/vi/9ljRhNIM--o/0.jpg)](https://www.youtube.com/watch?v=9ljRhNIM--o)


## Team 
- Jeremy Fischer
- Daniel Won
- Jakob Gollreiter
