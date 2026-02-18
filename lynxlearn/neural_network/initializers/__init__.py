"""
Weight initializers for neural network layers.

This module provides various weight initialization strategies
to help with training stability and convergence.

Available Initializers
----------------------
- RandomNormal: Initialize from normal distribution
- RandomUniform: Initialize from uniform distribution
- Zeros: Initialize with zeros
- Ones: Initialize with ones
- Constant: Initialize with constant value
- XavierNormal/GlorotNormal: Xavier normal initialization (for tanh/sigmoid)
- XavierUniform/GlorotUniform: Xavier uniform initialization
- HeNormal/KaimingNormal: He normal initialization (for ReLU)
- HeUniform/KaimingUniform: He uniform initialization
- LeCunNormal: LeCun normal initialization (for SELU)
- LeCunUniform: LeCun uniform initialization
- Orthogonal: Orthogonal initialization
- TruncatedNormal: Truncated normal distribution

Quick Start
-----------
>>> from lynxlearn.neural_network.initializers import HeNormal, XavierNormal
>>>
>>> # For ReLU activation
>>> he_init = HeNormal(seed=42)
>>> weights = he_init.initialize((784, 128))
>>>
>>> # For tanh/sigmoid activation
>>> xavier_init = XavierNormal(seed=42)
>>> weights = xavier_init.initialize((784, 128))
>>>
>>> # Use string identifiers
>>> from lynxlearn.neural_network.initializers import get_initializer
>>> init = get_initializer('he_normal', seed=42)
"""

from ._initializers import (
    INITIALIZERS,
    BaseInitializer,
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

__all__ = [
    # Base class
    "BaseInitializer",
    # Initializers
    "RandomNormal",
    "RandomUniform",
    "Zeros",
    "Ones",
    "Constant",
    "XavierNormal",
    "XavierUniform",
    "HeNormal",
    "HeUniform",
    "LeCunNormal",
    "LeCunUniform",
    "Orthogonal",
    "TruncatedNormal",
    # Aliases
    "GlorotNormal",
    "GlorotUniform",
    "KaimingNormal",
    "KaimingUniform",
    # Registry
    "INITIALIZERS",
    # Utilities
    "get_initializer",
]
