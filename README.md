# Turbulence Modeling Using Physics-Informed Neural Networks (PINNs)

This repository contains the implementation of a **Physics-Informed Neural Network (PINN)** for modeling turbulence in fluid dynamics, using Computational Fluid Dynamics (CFD) datasets. The project explores how PINNs can be used to approximate turbulence closures and simulate fluid flow phenomena governed by the Navier-Stokes equations.

## Overview

### Objectives
- Leverage **PINNs** to integrate physical laws (e.g., Navier-Stokes equations) into the neural network training process.
- Predict turbulence behavior using a supervised and physics-based learning approach.
- Explore the use of machine learning to enhance traditional turbulence modeling techniques.

### Dataset
The dataset used in this project is derived from **[Turbulence CFD Dataset on Kaggle](https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset)**, specifically the `komegasst.csv` file, which provides detailed turbulence parameters and flow features.

### Notebook Highlights
The notebook demonstrates the following:
1. **Data Loading and Preprocessing**:
   - Reading the `komegasst.csv` dataset and cleaning irrelevant or missing data.
   - Normalizing features and labels for efficient model training.
2. **Model Architecture**:
   - A **PINN model** is constructed with:
     - 4 hidden layers, each with 64 neurons.
     - `tanh` activation function.
   - Physics-based loss functions enforce fluid dynamics laws (e.g., Navier-Stokes equations).
3. **Training**:
   - Supervised learning on CFD data using mean squared error loss.
   - Physics-informed loss terms added to ensure consistency with physical laws.
4. **Evaluation**:
   - Visualizing predictions and gradients to analyze the model’s learning behavior.
   - Troubleshooting gradient dependency issues with TensorFlow's autograd utilities.

---

## Requirements

### Software and Libraries
- **Python**: 3.7+
- **TensorFlow**: 2.12+
- **NumPy**: For numerical operations.
- **Matplotlib**: For visualizing results.
- **Pandas**: For data manipulation and preprocessing.

Install the dependencies via pip:
```bash
pip install tensorflow numpy matplotlib pandas
