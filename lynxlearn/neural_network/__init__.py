"""
Neural Network module for LynxLearn.

This module provides a beginner-friendly neural network API inspired by Keras,
built from scratch using NumPy. It includes layers, optimizers, loss functions,
and utilities for building and training neural networks.

Available Components
--------------------
- Sequential: Linear stack of layers
- Dense: Fully connected layer
- SGD: Stochastic Gradient Descent optimizer
- MeanSquaredError: MSE loss function
- MeanAbsoluteError: MAE loss function
- HuberLoss: Huber loss function
- Weight initializers: HeNormal, XavierNormal, etc.

Quick Start
-----------
>>> from lynxlearn.neural_network import Sequential, Dense
>>> from lynxlearn.neural_network.optimizers import SGD
>>> from lynxlearn.neural_network.losses import MeanSquaredError
>>>
>>> # Create a simple model
>>> model = Sequential([
...     Dense(64, activation='relu', input_shape=(10,)),
...     Dense(32, activation='relu'),
...     Dense(1)
... ])
>>>
>>> # Compile the model
>>> model.compile(optimizer='sgd', loss='mse')
>>>
>>> # Train the model
>>> history = model.train(X_train, y_train, epochs=100, batch_size=32)
>>>
>>> # Make predictions
>>> predictions = model.predict(X_test)

Examples
--------
See the `examples/` directory for comprehensive usage examples.
"""

# Import submodules
from . import initializers, layers, losses, optimizers
from ._base import BaseNeuralNetwork
from ._sequential import Sequential
from .initializers import (
    GlorotNormal,
    GlorotUniform,
    HeNormal,
    HeUniform,
    XavierNormal,
    XavierUniform,
    get_initializer,
)

# Import commonly used classes for convenience
from .layers import BaseLayer, Dense
from .losses import BaseLoss, HuberLoss, MeanAbsoluteError, MeanSquaredError
from .optimizers import SGD, BaseOptimizer

__all__ = [
    # Main classes
    "Sequential",
    "BaseNeuralNetwork",
    # Layers
    "Dense",
    "BaseLayer",
    "layers",
    # Optimizers
    "SGD",
    "BaseOptimizer",
    "optimizers",
    # Losses
    "MeanSquaredError",
    "MeanAbsoluteError",
    "HuberLoss",
    "BaseLoss",
    "losses",
    # Initializers
    "HeNormal",
    "HeUniform",
    "XavierNormal",
    "XavierUniform",
    "GlorotNormal",
    "GlorotUniform",
    "get_initializer",
    "initializers",
]
