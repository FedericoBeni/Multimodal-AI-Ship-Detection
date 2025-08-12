# Multimodal AI for Ship Detection in Satellite Imagery

This repository contains the source code for a project focused on the comparative analysis of YOLO architectures for ship detection in satellite imagery. 

## Project Objectives

The primary goals of this project were to:

-   **Replicate the Baseline:** validate the results of the **YOLOv10sLight** model as presented in the reference study on our target dataset("Multimodal AI-enhanced ship detection for mapping fishing vessels and informing on suspicious activities", [Galdelli et al.(2025)](https://www.sciencedirect.com/science/article/pii/S0167865525000649).
-   **Explore Advanced Architectures:** implement and evaluate newer models, specifically [YOLOv11](https://docs.ultralytics.com/it/models/yolo11/) and [YOLOv12](https://docs.ultralytics.com/it/models/yolo12/), for the ship detection task.
-   **Develop a Custom Model:** create a lightweight version of YOLOv12 (**YOLOv12sLight**) by pruning its detection head, optimizing it for efficiency on smaller targets.
-   **Establish a Comparative Framework:** set up a consistent training and evaluation pipeline to fairly compare the different architectures.

## Methodology and Dataset

This project follows the core pipeline proposed by Galdelli et al. The primary distinction in our approach is the exclusive use of **single-channel (grayscale) images**. This was done to test the models' ability to detect ships based purely on shape, contrast, and form, without color information.

### Composite Dataset
The training and evaluation were performed on a large, composite dataset created by merging the following six open-source collections of satellite imagery. This dataset is the same as the one used in the reference study, adapted to a single-channel format.

-   [SDDCB](https://github.com/CAESAR-Radi/SAR-Ship-Dataset)
-   [SSDD](https://drive.google.com/file/d/1grDw3zbGjQKYPjOxv9-h4WSUctoUvu1O/view)
-   [HRSID](https://github.com/chaozhong2010/HRSID)
-   [S2\_Detection](https://universe.roboflow.com/sentinel2/sentinel-2-ship_detection)
-   [S2\_FC](https://huggingface.co/mayrajeo/marine-vessel-detection-yolov8)
-   [SDAI](https://www.kaggle.com/datasets/andrewmvd/ship-detection)

