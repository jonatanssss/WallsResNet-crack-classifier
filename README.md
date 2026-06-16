# Physics-Informed Deep Learning for Concrete Crack Detection

## Overview

This project implements a physics-informed deep learning pipeline for structural crack detection on concrete wall images using a custom ResNet architecture (WallsResNet).

The model is designed to classify images into two categories:
- Cracked
- NonCracked

In addition to deep learning classification, the project includes physics-based feature analysis to provide interpretable evidence of crack-related visual patterns such as intensity discontinuity, edge sharpness, high-frequency energy, and surface energy concentration.

## Dataset

The project uses the full Walls dataset:
- Total images: 18,138
- Cracked: 3,851
- NonCracked: 14,287

The data is split using an 80/20 train-test ratio:
- Training set: 14,510 images
- Test set: 3,628 images

To address class imbalance, the Cracked class in the training set is augmented four times, increasing the total training samples to 26,854.

## Methodology

### 1. Data Preprocessing
- Resize images to 128 × 128 pixels
- Convert images to grayscale
- Normalize pixel values
- Use ImageFolder-based loading

### 2. Data Augmentation
Augmentation is applied only to the Cracked class in the training set using:
- Horizontal / vertical flip
- Random rotation
- Gaussian blur
- Color jitter

### 3. Physics-Informed Feature Extraction
The pipeline extracts 38 numerical features from seven analysis groups:
- Intensity statistics
- Sobel gradient and Laplacian energy
- Canny edge features
- Otsu thresholding and skeletonization
- GLCM texture and Shannon entropy
- FFT power spectrum
- Surface energy density

### 4. Model Architecture
The classification model is a custom ResNet variant called WallsResNet:
- Input: grayscale 128 × 128
- Trainable parameters: 11,236,162
- Residual blocks with skip connections
- Batch Normalization
- Dropout2d
- Adaptive average pooling
- Fully connected classifier

### 5. Training Setup
- Epochs: 40
- Batch size: 32
- Initial learning rate: 0.001
- Weight decay: 1e-4
- Loss function: Weighted CrossEntropyLoss
- Optimizer: Adam
- Scheduler: CosineAnnealingLR
- Gradient clipping: max norm 1.0

## Results

The best model was achieved at epoch 20:
- Validation accuracy: 87.8445%
- AUC-ROC: 0.8978

Final test performance:
- Cracked: Precision 0.70, Recall 0.74, F1-score 0.72
- NonCracked: Precision 0.93, Recall 0.91, F1-score 0.92

The physics-informed analysis shows that the strongest discriminative features for cracked surfaces are:
- Laplacian energy (+29.6%)
- High-frequency FFT energy (+26.9%)
- Surface energy density (+20.0%)

These findings support the interpretation that cracks generate stronger discontinuities and higher spatial-frequency content in the image.

## Why This Project Matters

This project is relevant for structural health monitoring because manual crack inspection is often subjective and time-consuming. The proposed approach combines:
- Deep learning for classification
- Physics-based feature interpretation
- Imbalanced learning strategies
- Visual explanation of crack-related patterns

## Repository Structure

```text
.
├── README.md
├── requirements.txt
└── notebooks/
    └── walls_crack_physics_analysis.ipynb
