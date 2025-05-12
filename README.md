# 3D Reconstruction from Satellite Images

This repository contains the code and documentation for our paper on 3D reconstruction from sparse satellite imagery (10-20 views per scene). Link to the webpage of the project: https://jflyer45.github.io/cs280-project-webpage/

## Key Contributions
- Comparison of three reconstruction approaches:
  - 3D Gaussian Splatting with derived camera parameters
  - VGGT with bundle adjustment optimization
  - DUSt3R with optimized view selection
- Novel camera selection methodology using Fisher information theory
- Practical reconstruction from challenging satellite imagery conditions

## Methods
1. **3DGS**: Traditional Gaussian splatting adapted for satellite views
2. **VGGT+BA**: Transformer-based geometry estimation with bundle adjustment
3. **DUSt3R**: End-to-end neural reconstruction with optimal view selection

## Results
- VGGT+BA produces most geometrically accurate results
- Optimized DUSt3R offers most reliable reconstructions
- View selection improves all methods' performance

## Team 
- Jeremy Fischer
- Daniel Won
- Jakob Gollreiter
