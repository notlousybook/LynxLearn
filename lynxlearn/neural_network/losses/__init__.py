"""
Loss functions for neural network training - HYPER-OPTIMIZED.

This module provides various loss functions for training neural networks.
All loss functions are optimized using vectorized operations from _core.py.

Available Loss Functions
------------------------
- MeanSquaredError (MSE): Mean squared error for regression
- MeanAbsoluteError (MAE): Mean absolute error for regression
- HuberLoss: Robust loss combining MSE and MAE
- BinaryCrossEntropy (BCE): Binary classification loss
- CategoricalCrossEntropy (CCE): Multi-class classification loss

Performance
-----------
All loss functions use:
- Contiguous arrays for cache efficiency
- In-place operations to reduce memory allocation
- Optimized vectorized operations from _core.py
- Fast paths for common reduction modes

Quick Start
-----------
>>> from lynxlearn.neural_network.losses import MeanSquaredError, BinaryCrossEntropy
>>>
>>> # For regression
>>> loss_fn = MeanSquaredError()
>>> loss = loss_fn.compute(y_true, y_pred)
>>> gradient = loss_fn.gradient(y_true, y_pred)
>>>
>>> # For binary classification
>>> loss_fn = BinaryCrossEntropy()
>>> loss = loss_fn.compute(y_true, y_pred)
>>>
>>> # Use string identifiers in model.compile()
>>> model.compile(optimizer='sgd', loss='mse')
>>> model.compile(optimizer='adam', loss='binary_crossentropy')
"""

from ._base import BaseLoss
from ._mse import (
    BCE,
    CCE,
    MAE,
    MSE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    HuberLoss,
    MeanAbsoluteError,
    MeanSquaredError,
)

__all__ = [
    # Base class
    "BaseLoss",
    # Loss functions
    "MeanSquaredError",
    "MeanAbsoluteError",
    "HuberLoss",
    "BinaryCrossEntropy",
    "CategoricalCrossEntropy",
    # Aliases
    "MSE",
    "MAE",
    "BCE",
    "CCE",
]
