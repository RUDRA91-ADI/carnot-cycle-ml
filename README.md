# Carnot Cycle Simulation Using Neural Networks

My first Machine Learning project, built using Python and PyTorch.

This project explores how a neural network can learn the relationships between the thermodynamic states of a Carnot cycle using data generated from mathematical equations.

## Project Overview

A Carnot cycle consists of four thermodynamic processes:

1. Isothermal expansion
2. Adiabatic expansion
3. Isothermal compression
4. Adiabatic compression

The objective is to create a neural network that takes initial thermodynamic parameters as inputs and predicts the pressure and volume at the different stages of the cycle.

## Project Versions

### Version 1 — Original Hardcoded Neural Network

The first implementation manually defines:

- Input and process neurons
- Individual weights and biases
- Four hidden neurons
- ReLU activation
- Output neurons
- Loss calculation and SGD optimization

This version represents an initial attempt to understand how neural networks are constructed internally.

**Development note:** This is an early architectural experiment. The original code has incomplete training connections: some hidden-layer parameters are referenced before definition, and the hidden and output layers are not included in the training process.

### Version 2 — Optimized PyTorch Neural Network

The second implementation uses PyTorch's neural-network modules to build a more structured model.

#### Dataset

- 6,000 generated samples
- 6 input features
- 8 output values
- 5,000 training samples
- 1,000 testing samples

#### Input features

- Initial pressure (P1)
- Initial volume (V1)
- Hot-reservoir temperature (TH)
- Cold-reservoir temperature (TC)
- Volume ratio (r)
- Heat-capacity ratio (gamma)

#### Predicted outputs

- Pressure and volume after isothermal expansion
- Pressure and volume after adiabatic expansion
- Pressure and volume after isothermal compression
- Final pressure and volume after adiabatic compression

#### Neural network architecture

- Input layer: 6 neurons
- Hidden layer 1: 64 neurons with ReLU
- Hidden layer 2: 64 neurons with ReLU
- Output layer: 8 neurons

#### Training configuration

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Mean Squared Error
- Training epochs: 2,001

#### Additional techniques

- Input standardization
- Log transformation of positive output values
- Output standardization
- Training-loss visualization
- Evaluation on unseen test data
- Average pressure and volume percentage-error calculations

## Technologies Used

- Python
- PyTorch
- Matplotlib

## Repository Structure

```text
carnot-cycle-ml/
├── README.md
├── requirements.txt
├── version_1_hardcoded/
│   └── carnot_hardcoded.py
└── version_2_optimized/
    └── carnot_optimized.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/RUDRA91-ADI/carnot-cycle-ml.git
cd carnot-cycle-ml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run Version 1:

```bash
python version_1_hardcoded/carnot_hardcoded.py
```

Run Version 2:

```bash
python version_2_optimized/carnot_optimized.py
```

## What I Learned

Through this project, I explored:

- How neurons use weights and biases
- How multiple neurons form hidden layers
- How activation functions work
- How backpropagation updates model parameters
- How gradient-based optimizers train neural networks
- Why input and output normalization matter
- How to evaluate a model on unseen data
- How to visualize the training process

## Learning Journey

This project represents my first practical step into Machine Learning.

I started by manually constructing a neural-network architecture and gradually moved toward a more structured implementation using PyTorch. The two versions are preserved to show that progression.

**Note:** The training data is generated using mathematical Carnot-cycle relationships. The model learns to approximate these generated relationships; its predictions are not a replacement for thermodynamic calculations or experimental validation.
