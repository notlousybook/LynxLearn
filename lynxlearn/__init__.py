"""
LynxLearn - A simple machine learning library for educational purposes.

This library provides implementations of linear regression algorithms
with a beginner-friendly API. Built from scratch using NumPy for
learning and understanding ML fundamentals.

Available Modules
-----------------
- linear_model: Regression models (OLS, Gradient Descent, Ridge, Lasso, etc.)
- model_selection: Data splitting utilities
- metrics: Evaluation metrics (MSE, RMSE, MAE, R²)
- visualizations: Plotting and analysis tools

Quick Start
-----------
>>> from lynxlearn.linear_model import LinearRegression
>>> from lynxlearn.model_selection import train_test_split
>>> from lynxlearn import metrics
>>>
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
>>> model = LinearRegression()
>>> model.fit(X_train, y_train)
>>> predictions = model.predict(X_test)
>>> print(f"R²: {metrics.r2_score(y_test, predictions):.4f}")

Examples
--------
See the `examples/demo.py` script for comprehensive usage examples.

For more information, visit: https://github.com/notlousybook/LynxLearn
"""

__version__ = "0.2.0"
__author__ = "lousybook01"

# Import all regression models for easy access
from .linear_model import (
    LinearRegression,
    GradientDescentRegressor,
    Ridge,
    Lasso,
    ElasticNet,
    PolynomialRegression,
    PolynomialFeatures,
    HuberRegressor,
    QuantileRegressor,
    BayesianRidge,
)
from .model_selection import train_test_split
from . import metrics
from . import visualizations

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
    # Utilities
    "train_test_split",
    "metrics",
    "visualizations",
]
