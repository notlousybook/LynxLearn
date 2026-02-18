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
>>>
>>> # Create a simple model
>>> model = Sequential([
...     Dense(64, activation='relu', input_shape=(10,)),
...     Dense(32, activation='relu'),
...     Dense(1)
... ])
>>> model.compile(optimizer='sgd', loss='mse')
>>> history = model.train(X_train, y_train, epochs=100)

Custom Precision
----------------
>>> from lynxlearn.neural_network import DenseBF16, DenseFloat16
>>>
>>> # BF16 precision (requires ml_dtypes)
>>> model = Sequential([
...     DenseBF16(128, activation='relu'),
...     DenseBF16(1)
... ])

With Regularization
-------------------
>>> from lynxlearn.neural_network import Dense, L2Regularizer, MaxNorm
>>>
>>> layer = Dense(128, activation='relu',
...               kernel_regularizer=L2Regularizer(l2=0.01),
...               kernel_constraint=MaxNorm(3.0))

Examples
--------
See the `examples/` directory for comprehensive usage examples.

For more information, visit: https://github.com/notlousybook/LynxLearn
"""

__version__ = "0.3.0"
__author__ = "lousybook01"

# Import submodules
from . import metrics, neural_network, visualizations

# Import linear models
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

# Import model selection
from .model_selection import train_test_split

# Import neural network components - Models
# Import neural network components - Layers
# Import neural network components - Regularizers
# Import neural network components - Constraints
# Import neural network components - Optimizers
# Import neural network components - Losses
# Import neural network components - Initializers
# Import neural network components - Other
from .neural_network import (
    LBFGS,
    MAE,
    MSE,
    SGD,
    ActivationRegistry,
    BaseLayer,
    BaseLoss,
    BaseNeuralNetwork,
    BaseOptimizer,
    Constant,
    Constraint,
    Dense,
    DenseBF16,
    DenseFloat16,
    DenseFloat32,
    DenseFloat64,
    DenseMixedPrecision,
    GlorotNormal,
    GlorotUniform,
    HeNormal,
    HeUniform,
    HuberLoss,
    KaimingNormal,
    KaimingUniform,
    L1L2Regularizer,
    L1Regularizer,
    L2Regularizer,
    LBFGSLinearRegression,
    LeCunNormal,
    LeCunUniform,
    MaxNorm,
    MeanAbsoluteError,
    MeanSquaredError,
    MinMaxNorm,
    NonNeg,
    Ones,
    Orthogonal,
    RandomNormal,
    RandomUniform,
    Regularizer,
    Sequential,
    TruncatedNormal,
    UnitNorm,
    XavierNormal,
    XavierUniform,
    Zeros,
    get_initializer,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Submodules
    "metrics",
    "visualizations",
    "neural_network",
    # Linear Models
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
    # Neural Networks - Models
    "Sequential",
    "BaseNeuralNetwork",
    # Neural Networks - Layers
    "Dense",
    "DenseFloat16",
    "DenseFloat32",
    "DenseFloat64",
    "DenseBF16",
    "DenseMixedPrecision",
    "BaseLayer",
    # Neural Networks - Regularizers
    "Regularizer",
    "L1Regularizer",
    "L2Regularizer",
    "L1L2Regularizer",
    # Neural Networks - Constraints
    "Constraint",
    "MaxNorm",
    "NonNeg",
    "UnitNorm",
    "MinMaxNorm",
    # Neural Networks - Optimizers
    "SGD",
    "LBFGS",
    "LBFGSLinearRegression",
    "BaseOptimizer",
    # Neural Networks - Losses
    "MeanSquaredError",
    "MeanAbsoluteError",
    "HuberLoss",
    "MSE",
    "MAE",
    "BaseLoss",
    # Neural Networks - Initializers
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
    # Neural Networks - Other
    "ActivationRegistry",
]
