"""
Neural Network module for LynxLearn.

A beginner-friendly neural network API built from scratch using NumPy,
with full support for multiple data types, custom components, and
extensive configuration options.

Quick Start
-----------
>>> from lynxlearn.neural_network import Sequential, Dense
>>>
>>> # Create a simple model
>>> model = Sequential([
...     Dense(64, activation='relu', input_shape=(10,)),
...     Dense(32, activation='relu'),
...     Dense(1)
... ])
>>>
>>> # Compile and train
>>> model.compile(optimizer='sgd', loss='mse')
>>> history = model.train(X_train, y_train, epochs=100)

Custom Precision
----------------
>>> from lynxlearn.neural_network.layers import DenseBF16, DenseFloat16
>>>
>>> # BF16 precision (requires ml_dtypes)
>>> model = Sequential([
...     DenseBF16(128, activation='relu'),
...     DenseBF16(1)
... ])
>>>
>>> # Mixed precision
>>> from lynxlearn.neural_network.layers import DenseMixedPrecision
>>> layer = DenseMixedPrecision(128, storage_dtype='float16', compute_dtype='float32')

With Regularization
-------------------
>>> from lynxlearn.neural_network.layers import Dense, L2Regularizer, MaxNorm
>>>
>>> model = Sequential([
...     Dense(128, activation='relu',
...           kernel_regularizer=L2Regularizer(l2=0.01),
...           kernel_constraint=MaxNorm(3.0)),
...     Dense(1)
... ])

Available Components
--------------------
Layers:
  - Dense: Universal dense layer with full customization
  - DenseFloat16, DenseFloat32, DenseFloat64, DenseBF16: Dtype-specific layers
  - DenseMixedPrecision: Mixed precision training layer

Optimizers:
  - SGD: Stochastic Gradient Descent with momentum, Nesterov, clipping

Loss Functions:
  - MeanSquaredError, MSE: Mean squared error for regression
  - MeanAbsoluteError, MAE: Mean absolute error for regression
  - HuberLoss: Robust loss combining MSE and MAE

Regularizers:
  - L1Regularizer, L2Regularizer, L1L2Regularizer

Constraints:
  - MaxNorm, NonNeg, UnitNorm, MinMaxNorm

Initializers:
  - HeNormal, HeUniform: For ReLU-family activations
  - XavierNormal, XavierUniform: For tanh/sigmoid
  - LeCunNormal, LeCunUniform: For SELU
  - RandomNormal, RandomUniform, Zeros, Ones, Constant

Models:
  - Sequential: Linear stack of layers

For more information, visit: https://github.com/notlousybook/LynxLearn
"""

__version__ = "0.3.0"

# Import submodules
from . import initializers, layers, losses, optimizers

# Base classes
from ._base import BaseNeuralNetwork
from ._sequential import Sequential

# Initializer classes (commonly used)
from .initializers import (
    Constant,
    GlorotNormal,
    GlorotUniform,
    HeNormal,
    HeUniform,
    KaimingNormal,
    KaimingUniform,
    LeCunNormal,
    LeCunUniform,
    Ones,
    Orthogonal,
    RandomNormal,
    RandomUniform,
    TruncatedNormal,
    XavierNormal,
    XavierUniform,
    Zeros,
    get_initializer,
)

# Layer classes
from .layers import (
    # Activation registry
    ActivationRegistry,
    BaseLayer,
    # Constraints
    Constraint,
    Dense,
    DenseBF16,
    DenseFloat16,
    DenseFloat32,
    DenseFloat64,
    DenseMixedPrecision,
    # Regularizers
    L1L2Regularizer,
    L1Regularizer,
    L2Regularizer,
    MaxNorm,
    MinMaxNorm,
    NonNeg,
    Regularizer,
    UnitNorm,
)

# Loss function classes
from .losses import (
    MAE,
    MSE,
    BaseLoss,
    HuberLoss,
    MeanAbsoluteError,
    MeanSquaredError,
)

# Optimizer classes
from .optimizers import LBFGS, SGD, BaseOptimizer, LBFGSLinearRegression

# Public API
__all__ = [
    # Version
    "__version__",
    # Submodules
    "layers",
    "optimizers",
    "losses",
    "initializers",
    # Models
    "Sequential",
    "BaseNeuralNetwork",
    # Layers
    "Dense",
    "DenseFloat16",
    "DenseFloat32",
    "DenseFloat64",
    "DenseBF16",
    "DenseMixedPrecision",
    "BaseLayer",
    # Regularizers
    "Regularizer",
    "L1Regularizer",
    "L2Regularizer",
    "L1L2Regularizer",
    # Constraints
    "Constraint",
    "MaxNorm",
    "NonNeg",
    "UnitNorm",
    "MinMaxNorm",
    # Optimizers
    "SGD",
    "BaseOptimizer",
    "LBFGS",
    "LBFGSLinearRegression",
    # Losses
    "MeanSquaredError",
    "MeanAbsoluteError",
    "HuberLoss",
    "MSE",
    "MAE",
    "BaseLoss",
    # Initializers
    "HeNormal",
    "HeUniform",
    "XavierNormal",
    "XavierUniform",
    "GlorotNormal",
    "GlorotUniform",
    "LeCunNormal",
    "LeCunUniform",
    "KaimingNormal",
    "KaimingUniform",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
    "Zeros",
    "Ones",
    "Constant",
    "Orthogonal",
    "get_initializer",
    # Activation registry
    "ActivationRegistry",
]
