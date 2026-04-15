# CNN Architecture Study: Representation and Robustness

## Overview
This project presents a comparative study of three classical convolutional neural network (CNN) architectures: LeNet, AlexNet, and ResNet. The goal is to analyze how network depth and architectural design influence representation learning, training dynamics, and robustness.

## Research Question
How do different CNN architectures (LeNet, AlexNet, ResNet) differ in:
- Representation quality?
- Convergence behavior during training?
- Robustness to input perturbations?

## Dataset
- CIFAR-10 dataset
- 10 classes of natural images
- Standard train/test split

## Models
- **LeNet**: shallow CNN baseline
- **AlexNet**: deeper CNN with larger capacity
- **ResNet**: deep residual network with skip connections

## Experimental Setup
- Same dataset and preprocessing for all models
- Same training pipeline for fair comparison
- Evaluation metrics:
  - Accuracy
  - Loss
  - Convergence speed

## Analysis Plan
This project focuses not only on model performance but also on understanding model behavior:

1. **Training Dynamics**
   - Compare loss and accuracy curves
   - Analyze convergence stability

2. **Representation Learning**
   - Extract features from trained models
   - Visualize using t-SNE / PCA
   - Compare separability of learned representations

3. **Robustness Analysis**
   - Evaluate models under input perturbations:
     - Gaussian noise
     - Blur
   - Compare performance degradation

## Expected Outcomes
- Deeper architectures (ResNet) will learn more discriminative representations
- Residual connections improve training stability
- Model depth influences robustness to noise

## Reproducibility
The codebase is organized for reproducible experiments:
- Unified training pipeline
- Configurable experiment settings
- Clear separation of models, training, and analysis

## Future Work
- Extend to Transformer-based models (e.g., Vision Transformer)
- Explore multimodal extensions (vision-language models)
