"""
Neural network layers for LynxLearn.

This module provides various layer types for building neural networks.

Available Layers
----------------
- Dense: Fully connected (dense) layer
- BaseLayer: Abstract base class for custom layers

Quick Start
-----------
>>> from lynxlearn.neural_network.layers import Dense
>>>
>>> # Create a dense layer
>>> layer = Dense(128, activation='relu', input_shape=(784,))
>>> output = layer.forward(X)
"""

from ._base import BaseLayer
from ._dense import Dense

__all__ = [
    "BaseLayer",
    "Dense",
]
