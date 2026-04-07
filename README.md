# Multimodal AI for Ship Detection in Satellite Imagery

This project explores and compares modern YOLO architectures for ship detection using multimodal satellite imagery (Sentinel-1 SAR and Sentinel-2 Optical).

## Overview

- Dataset: 69,000+ images (SAR + optical)
- Task: ship detection in challenging maritime environments
- Framework: PyTorch + Ultralytics YOLO
- Focus: accuracy vs efficiency trade-off

## Objectives

- Replicate YOLOv10sLight baseline from Galdelli et al. (2025)
- Evaluate newer architectures (YOLOv11, YOLOv12)
- Design a lightweight model (YOLOv12sLight)
- Build a consistent benchmarking pipeline

## Key Contributions

- Implemented and benchmarked YOLOv10, YOLOv11, YOLOv12 families  
- Developed **YOLOv12sLight** via head pruning  
  → reduced compute from **21.2 → 17.0 GFLOPs (-20%)**  
- Achieved best performance with **YOLOv11m (AP50 = 0.851)**  
- Demonstrated that lightweight pruning preserves performance while improving efficiency  

## Dataset

Composite dataset (HS3-S2) combining 6 open-source sources:

- [SDDCB](https://github.com/CAESAR-Radi/SAR-Ship-Dataset)
- [SSDD](https://drive.google.com/file/d/1grDw3zbGjQKYPjOxv9-h4WSUctoUvu1O/view)
- [HRSID](https://github.com/chaozhong2010/HRSID)
- [S2 Detection](https://universe.roboflow.com/sentinel2/sentinel-2-ship_detection)
- [S2 FC](https://huggingface.co/mayrajeo/marine-vessel-detection-yolov8)
- [SDAI](https://www.kaggle.com/datasets/andrewmvd/ship-detection) 

All images converted to **single-channel (grayscale)** to:
- emphasize shape and contrast features  
- reduce computational cost  

## Methodology

- Training split: 70% train / 15% val / 15% test  
- Training conducted in Docker environment (multi-GPU setup)  
- Models trained from scratch for fair comparison  

## Results

| Model        | AP50 |
|-------------|------|
| YOLOv10sLight | 0.831 |
| YOLOv11m     | **0.851** |
| YOLOv12sLight | 0.585 |

- YOLOv11 family outperformed newer YOLOv12 in this setup  
- YOLOv12 requires further tuning or larger datasets  
- Lightweight models achieved strong efficiency gains  

## Insights

- Simpler architectures can outperform more complex ones in limited-data regimes  
- Grayscale input is sufficient for ship detection in SAR/optical imagery  
- Pruning is effective for edge-oriented models  

##  Tech Stack

- Python, PyTorch  
- Ultralytics YOLO  
- WandB (tracking)  
- Docker (reproducibility)  
- CodeCarbon (CO₂ tracking)

##  Future Work

- Hyperparameter tuning for YOLOv12  
- Pretraining on larger datasets (COCO/ImageNet)  
- Multichannel input (RGB vs grayscale comparison)  
- Error analysis (false positives/negatives)

## References & Resources

- [Galdelli et al. (2025)](https://www.sciencedirect.com/science/article/pii/S0167865525000649)


