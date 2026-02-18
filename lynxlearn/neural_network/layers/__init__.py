"""
Neural network layers for LynxLearn.

This module provides various layer types for building neural networks,
with full support for different data types, custom initializers,
activations, regularizers, and constraints.

Available Layers
----------------
- Dense: Universal fully connected layer with full customization
- DenseFloat16: Dense layer with float16 precision
- DenseFloat32: Dense layer with float32 precision
- DenseFloat64: Dense layer with float64 precision
- DenseBF16: Dense layer with bfloat16 precision (requires ml_dtypes)
- DenseMixedPrecision: Dense layer with mixed precision training
- BaseLayer: Abstract base class for custom layers

Regularizers
------------
- Regularizer: Base regularizer with L1 and L2 support
- L1Regularizer: L1 (Lasso) regularization
- L2Regularizer: L2 (Ridge) regularization
- L1L2Regularizer: Combined L1 and L2 (Elastic Net) regularization

Constraints
-----------
- Constraint: Base constraint class
- MaxNorm: Maximum norm constraint
- NonNeg: Non-negative constraint
- UnitNorm: Unit norm constraint
- MinMaxNorm: Min-max norm constraint

Activations
-----------
- ActivationRegistry: Registry for activation functions

Built-in activations: relu, leaky_relu, sigmoid, tanh, softmax, elu,
selu, swish, silu, gelu, softplus, softsign, mish, linear

Quick Start
-----------
>>> from lynxlearn.neural_network.layers import Dense, L2Regularizer, MaxNorm
>>>
>>> # Basic usage
>>> layer = Dense(128, activation='relu')
>>>
>>> # With all customizations
>>> layer = Dense(
...     units=256,
...     activation='gelu',
...     kernel_initializer='xavier_uniform',
...     kernel_regularizer=L2Regularizer(l2=0.01),
...     kernel_constraint=MaxNorm(3.0),
...     dtype='float32'
... )
>>>
>>> # Mixed precision training
>>> from lynxlearn.neural_network.layers import DenseMixedPrecision
>>> layer = DenseMixedPrecision(128, storage_dtype='float16', compute_dtype='float32')
>>>
>>> # BF16 precision (requires ml_dtypes)
>>> from lynxlearn.neural_network.layers import DenseBF16
>>> layer = DenseBF16(128, activation='relu')
"""

from ._base import BaseLayer
from ._dense import (
    ActivationRegistry,
    Constraint,
    Dense,
    DenseBF16,
    DenseFloat16,
    DenseFloat32,
    DenseFloat64,
    DenseMixedPrecision,
    L1L2Regularizer,
    L1Regularizer,
    L2Regularizer,
    MaxNorm,
    MinMaxNorm,
    NonNeg,
    Regularizer,
    UnitNorm,
)

# List of public exports
__all__ = [
    # Base class
    "BaseLayer",
    # Main layer
    "Dense",
    # Dtype-specific layers
    "DenseFloat16",
    "DenseFloat32",
    "DenseFloat64",
    "DenseBF16",
    "DenseMixedPrecision",
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
    # Activation registry
    "ActivationRegistry",
]

# Version info for the layers module
__version__ = "0.3.0"
