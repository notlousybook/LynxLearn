"""
Neural network optimizers for LynxLearn.

This module provides various optimization algorithms for training
neural networks.

Available Optimizers
--------------------
- SGD: Stochastic Gradient Descent with momentum support

Quick Start
-----------
>>> from lynxlearn.neural_network.optimizers import SGD
>>>
>>> # Vanilla SGD
>>> optimizer = SGD(learning_rate=0.01)
>>>
>>> # SGD with momentum
>>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
>>>
>>> # SGD with Nesterov momentum
>>> optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
>>>
>>> # With gradient clipping
>>> optimizer = SGD(learning_rate=0.01, clipnorm=1.0)
"""

from ._base import BaseOptimizer
from ._sgd import SGD

__all__ = [
    # Base class
    "BaseOptimizer",
    # Optimizers
    "SGD",
]
