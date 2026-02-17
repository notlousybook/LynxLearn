"""
Visualization utilities for LynxLearn.

This module provides comprehensive plotting and visualization capabilities
for regression models, including:

- Basic plotting: regression lines, residuals, cost history
- Comprehensive analysis: 1D regression visualizations, metrics dashboards
- Gradient descent analysis: convergence plots, learning curves
- Regularization comparison: Ridge/Lasso coefficient paths
- Model comparison: side-by-side model performance reports

Examples
--------
>>> from lynxlearn import visualizations as viz
>>> from lynxlearn.linear_model import LinearRegression
>>>
>>> model = LinearRegression()
>>> model.fit(X_train, y_train)
>>>
>>> # Basic plots
>>> fig = viz.plot_regression(X_test, y_test, model)
>>> fig = viz.plot_residuals(y_test, y_pred)
>>>
>>> # Comprehensive analysis
>>> fig = viz.visualize_1d_regression(X_test, y_test, model)
>>> fig = viz.visualize_all_metrics(y_test, y_pred)
>>>
>>> # Model comparison
>>> report = viz.create_comprehensive_report(X_train, y_train, X_test, y_test, models)
"""

# Basic plots (from _plots.py)
from ._plots import (
    plot_regression,
    plot_cost_history,
    plot_residuals,
    plot_coefficients,
    compare_models,
)

# Comprehensive regression analysis (from _regression_analysis.py)
from ._regression_analysis import (
    visualize_1d_regression,
    visualize_all_metrics,
)

# Gradient descent visualizations (from _gradient_descent.py)
from ._gradient_descent import (
    visualize_gradient_descent,
)

# Regularization comparisons (from _regularization.py)
from ._regularization import (
    visualize_ridge_comparison,
)

# Model comparison reports (from _model_comparison.py)
from ._model_comparison import (
    create_comprehensive_report,
)

# Aliases for easier imports (backward compatibility)
viz_1d = visualize_1d_regression
viz_gd = visualize_gradient_descent
viz_ridge = visualize_ridge_comparison
viz_metrics = visualize_all_metrics
viz_report = create_comprehensive_report

__all__ = [
    # Basic plots
    "plot_regression",
    "plot_cost_history",
    "plot_residuals",
    "plot_coefficients",
    "compare_models",
    # Comprehensive analysis
    "visualize_1d_regression",
    "visualize_all_metrics",
    # Gradient descent
    "visualize_gradient_descent",
    # Regularization
    "visualize_ridge_comparison",
    # Model comparison
    "create_comprehensive_report",
    # Aliases
    "viz_1d",
    "viz_gd",
    "viz_ridge",
    "viz_metrics",
    "viz_report",
]
