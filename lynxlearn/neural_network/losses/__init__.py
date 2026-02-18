"""
Loss functions for neural network training.

This module provides various loss functions for training neural networks.

Available Loss Functions
------------------------
- MeanSquaredError (MSE): Mean squared error for regression
- MeanAbsoluteError (MAE): Mean absolute error for regression
- HuberLoss: Robust loss combining MSE and MAE

Quick Start
-----------
>>> from lynxlearn.neural_network.losses import MeanSquaredError, MeanAbsoluteError
>>>
>>> # For regression
>>> loss_fn = MeanSquaredError()
>>> loss = loss_fn.compute(y_true, y_pred)
>>> gradient = loss_fn.gradient(y_true, y_pred)
>>>
>>> # Use string identifiers in model.compile()
>>> model.compile(optimizer='sgd', loss='mse')
"""

from ._base import BaseLoss
from ._mse import (
    MAE,
    MSE,
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
    # Aliases
    "MSE",
    "MAE",
]
