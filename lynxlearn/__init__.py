"""
LynxLearn - A simple machine learning library for educational purposes.

This library provides implementations of linear regression algorithms
and neural networks with a beginner-friendly API. Built from scratch
using NumPy for learning and understanding ML fundamentals.

Available Modules
-----------------
- linear_model: Regression models (OLS, Gradient Descent, Ridge, Lasso, etc.)
- neural_network: Neural network components (Sequential, Dense, optimizers, losses)
- model_selection: Data splitting utilities
- metrics: Evaluation metrics (MSE, RMSE, MAE, R²)
- visualizations: Plotting and analysis tools

Quick Start - Linear Regression
-------------------------------
>>> from lynxlearn.linear_model import LinearRegression
>>> from lynxlearn.model_selection import train_test_split
>>> from lynxlearn import metrics
>>>
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
>>> model = LinearRegression()
>>> model.fit(X_train, y_train)
>>> predictions = model.predict(X_test)
>>> print(f"R²: {metrics.r2_score(y_test, predictions):.4f}")

Quick Start - Neural Networks
-----------------------------
>>> from lynxlearn.neural_network import Sequential, Dense
>>> from lynxlearn.neural_network.optimizers import SGD
>>> from lynxlearn.neural_network.losses import MeanSquaredError
>>>
>>> model = Sequential([
...     Dense(64, activation='relu', input_shape=(10,)),
...     Dense(1)
... ])
>>> model.compile(optimizer='sgd', loss='mse')
>>> history = model.train(X_train, y_train, epochs=100)
>>> predictions = model.predict(X_test)

Examples
--------
See the `examples/demo.py` script for comprehensive usage examples.

For more information, visit: https://github.com/notlousybook/LynxLearn
"""

__version__ = "0.3.0"
__author__ = "lousybook01"

# Import all regression models for easy access
from . import metrics, neural_network, visualizations
from .linear_model import (
    BayesianRidge,
    ElasticNet,
    GradientDescentRegressor,
    HuberRegressor,
    Lasso,
    LinearRegression,
    PolynomialFeatures,
    PolynomialRegression,
    QuantileRegressor,
    Ridge,
)
from .model_selection import train_test_split

# Import commonly used neural network components for convenience
from .neural_network import (
    SGD,
    Dense,
    MeanAbsoluteError,
    MeanSquaredError,
    Sequential,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Regression Models
    "LinearRegression",
    "GradientDescentRegressor",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "PolynomialRegression",
    "PolynomialFeatures",
    "HuberRegressor",
    "QuantileRegressor",
    "BayesianRidge",
    # Neural Networks
    "Sequential",
    "Dense",
    "SGD",
    "MeanSquaredError",
    "MeanAbsoluteError",
    "neural_network",
    # Utilities
    "train_test_split",
    "metrics",
    "visualizations",
]
