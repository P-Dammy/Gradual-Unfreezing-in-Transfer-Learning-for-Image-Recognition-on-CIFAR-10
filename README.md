# Gradual Unfreezing in Transfer Learning for Image Recognition on CIFAR-10

## Introduction

This project implements a **Convolutional Neural Network (CNN)** using **Transfer Learning** for image classification on the **CIFAR-10 dataset**. The model leverages a pre-trained **VGG16** network, initially freezing its layers and later gradually unfreezing them for fine-tuning. The goal is to improve the model's accuracy through transfer learning, while also exploring its performance after unfreezing and retraining.

The **CIFAR-10 dataset** is a collection of 60,000 32x32 color images across 10 different classes. The project demonstrates data preparation, transfer learning, model training, evaluation, and visualization, including confusion matrix analysis.

---

## Data Description

The **CIFAR-10 dataset** consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images. Each image is categorized into one of the following classes:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The dataset is available directly from the `tensorflow.keras.datasets` module.

---

## Model Architecture

The model used in this project is based on the VGG16 architecture, which was pre-trained on the ImageNet dataset. The key components of the model include:

Pre-trained VGG16 Base: Used as the feature extractor.

Custom Layers on Top:

Flatten Layer: Converts the 3D feature maps to 1D.
Dense Layer: Fully connected layer with 256 units and ReLU activation.
Dropout Layer: Dropout with a rate of 0.5 to prevent overfitting.
Output Layer: 10 units (one per class) with Softmax activation for classification.

---

## Transfer Learning Approach
Freezing the Base Model
Initially, all layers of the pre-trained VGG16 base model are frozen, meaning their weights remain unchanged during training. This allows the model to leverage the learned features from ImageNet without any modifications.

---

## Results

**Model Accuracy**

The model was evaluated on both the training and validation datasets during training. The final performance on the test set showed the following results:

**Training Accuracy: 0.94**
**Validation Accuracy: 0.85**
**Test Accuracy: 0.83**

The fine-tuning of the pre-trained model after gradual unfreezing helped improve the model's performance on the test set.
