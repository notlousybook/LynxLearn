"""
Neural network optimizers for LynxLearn.

This module provides various optimization algorithms for training
neural networks.

Available Optimizers
--------------------
- SGD: Stochastic Gradient Descent with momentum support
- LBFGS: Limited-memory BFGS (quasi-Newton) - THE SECRET SAUCE!
- LBFGSLinearRegression: Fast linear regression using L-BFGS

Quick Start
-----------
>>> from lynxlearn.neural_network.optimizers import SGD, LBFGS
>>>
>>> # Vanilla SGD
>>> optimizer = SGD(learning_rate=0.01)
>>>
>>> # SGD with momentum
>>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
>>>
>>> # L-BFGS for fast convex optimization
>>> optimizer = LBFGS(memory_size=10, tol=1e-6)

L-BFGS: The Secret Sauce
------------------------
L-BFGS is the algorithm that makes scikit-learn's linear models FAST!
It uses quasi-Newton optimization with limited memory, achieving
superlinear convergence (faster than SGD's linear convergence).

Best for:
- Linear regression (can beat Normal Equation for medium problems)
- Logistic regression
- Small-to-medium convex optimization problems

>>> from lynxlearn.neural_network.optimizers import LBFGSLinearRegression
>>> model = LBFGSLinearRegression()
>>> model.fit(X, y)
>>> predictions = model.predict(X_test)
"""

from ._base import BaseOptimizer
from ._lbfgs import LBFGS, LBFGSLinearRegression
from ._sgd import SGD

__all__ = [
    # Base class
    "BaseOptimizer",
    # Optimizers
    "SGD",
    "LBFGS",
    "LBFGSLinearRegression",
]
