"""
Metrics for evaluating regression models.

This module provides common regression metrics for evaluating model performance:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score (coefficient of determination)

Examples
--------
>>> from lousybookml import metrics
>>>
>>> # Calculate metrics
>>> mse = metrics.mean_squared_error(y_true, y_pred)
>>> rmse = metrics.root_mean_squared_error(y_true, y_pred)
>>> mae = metrics.mean_absolute_error(y_true, y_pred)
>>> r2 = metrics.r2_score(y_true, y_pred)
>>>
>>> # Or use convenient aliases
>>> mse = metrics.mse(y_true, y_pred)
>>> rmse = metrics.rmse(y_true, y_pred)
>>> mae = metrics.mae(y_true, y_pred)
"""

from ._regression import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

__all__ = [
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "mse",
    "rmse",
    "mae",
]

# Aliases for convenience
mse = mean_squared_error
rmse = root_mean_squared_error
mae = mean_absolute_error
