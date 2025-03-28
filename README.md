# Variational Continual Learning (VCL)

This is a reimplementation of the paper "Variational Continual Learning" (Nguyen et al., ICLR 2018) using JAX and Flax.

## Overview

The project implements a variational approach to continual learning, where a neural network learns multiple tasks sequentially without forgetting previous tasks. The key idea is to maintain a variational posterior over the network weights that is updated with each new task while respecting the knowledge from previous tasks through a KL divergence term.

## Project Structure

- `src/model.py`: Implementation of the variational neural network
- `src/loss.py`: Implementation of the variational loss function with KL divergence
- `src/mnist_perm.py`: Dataset handling for permuted MNIST
- `src/train.py`: Training loop and continual learning implementation

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To train the model on permuted MNIST tasks:

```bash
python src/train.py
```

The script will:
1. Create 5 different permuted MNIST tasks
2. Train the model sequentially on each task
3. Print training and test metrics every 5 epochs
4. Update the prior parameters after each task

## Model Architecture

The model consists of:
- Three variational dense layers (100, 100, 10 units)
- ReLU activation between layers
- Softmax output layer

## Loss Function

The loss function combines:
- Negative log likelihood (cross-entropy) for the current task
- KL divergence between current posterior and previous posterior (prior)

## Results

The model should maintain good performance on previous tasks while learning new ones, demonstrating the effectiveness of the variational continual learning approach. 