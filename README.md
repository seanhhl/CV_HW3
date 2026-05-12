# NYCU Computer Vision 2026 HW2

- Student ID：112261014
- Name：李瀚翔

## Introduction
This repository contains the PyTorch implementation for NYCU Computer Vision 2026 HW3: Medical Cell Instance Segmentation. The goal is to detect cells in medical images and predict their bounding boxes, classes, and pixel-level masks.
To effectively improve the model's performance and overcome the challenge of an extremely small dataset (only 209 images), we applied several advanced techniques:

* Self-Supervised Vision Transformer Backbone (DINOv2)

* Custom Feature Pyramid Network (FPN) Wrapper

* Robust Data Augmentation (Geometric & Photometric via albumentations)

* AdamW Optimizer & Cosine Annealing Learning Rate Scheduler

* Automatic Mixed Precision (AMP) & Gradient Clipping for stable training

---

## Environment Setup
This project is implemented and fully tested on **Google Colab**.

### Basic Requirements:
* **Python:** 3.8+
* **Libraries:** `torch`, `torchvision`, `transformers`, `albumentations`, `Pillow`, `tqdm`
* **Hardware:** CUDA-enabled GPU (e.g., Colab T4/L4 GPU)

---

## Usage

To train the DETR model from scratch and generate the predictions, simply run the main script (or execute the cells sequentially in Google Colab):
```bash
python main.py
```

## Snapshot
<img width="930" height="41" alt="image" src="https://github.com/user-attachments/assets/faac336b-bd2c-49bd-974a-5ac502c3ed61" />

585052df9ee0" />

