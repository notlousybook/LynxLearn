"""
Plotting functions for regression visualization.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_regression(X, y, model, title="Linear Regression", figsize=(10, 6), save_path=None):
    """
    Plot data points and regression line.

    For 1D features: scatter plot with regression line.
    For 2D features: 3D scatter with regression plane.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target values.
    model : fitted regression model
        Model with predict method.
    title : str, default="Linear Regression"
        Plot title.
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str, optional
        Path to save the figure.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    fig = plt.figure(figsize=figsize)

    if X.ndim == 1 or X.shape[1] == 1:
        # 1D regression
        ax = fig.add_subplot(111)
        x_flat = X.flatten() if X.ndim > 1 else X

        # Scatter plot
        ax.scatter(x_flat, y, alpha=0.6, label="Data points", color="blue")

        # Regression line
        x_line = np.linspace(x_flat.min(), x_flat.max(), 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, color="red", linewidth=2, label="Regression line")

        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif X.shape[1] == 2:
        # 2D regression - 3D plot
        ax = fig.add_subplot(111, projection="3d")

        # Scatter plot
        ax.scatter(X[:, 0], X[:, 1], y, alpha=0.6, label="Data points", color="blue")

        # Regression plane
        x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
        x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        # Predict on grid
        grid_points = np.c_[X1.ravel(), X2.ravel()]
        Y = model.predict(grid_points).reshape(X1.shape)

        ax.plot_surface(X1, X2, Y, alpha=0.3, color="red", label="Regression plane")

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Target")
        ax.set_title(title)

    else:
        raise ValueError("plot_regression only supports 1D or 2D features")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cost_history(model, title="Cost Function History", figsize=(10, 6), save_path=None):
    """
    Plot the cost function history from gradient descent.

    Parameters
    ----------
    model : GradientDescentRegressor
        Fitted gradient descent model with cost_history attribute.
    title : str, default="Cost Function History"
        Plot title.
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    if not hasattr(model, "cost_history") or not model.cost_history:
        raise ValueError("Model must have cost_history attribute. Use GradientDescentRegressor.")

    fig, ax = plt.subplots(figsize=figsize)

    iterations = range(1, len(model.cost_history) + 1)
    ax.plot(iterations, model.cost_history, linewidth=2, color="blue")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost (MSE)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_residuals(y_true, y_pred, title="Residuals Plot", figsize=(10, 6), save_path=None):
    """
    Plot residuals (errors) vs predicted values.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    title : str, default="Residuals Plot"
        Plot title.
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_pred, residuals, alpha=0.6, color="blue")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)

    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_coefficients(models, feature_names=None, title="Model Coefficients", figsize=(12, 6), save_path=None):
    """
    Compare coefficients across multiple models.

    Parameters
    ----------
    models : dict
        Dictionary of {model_name: fitted_model}.
    feature_names : list, optional
        Names of features.
    title : str, default="Model Coefficients"
        Plot title.
    figsize : tuple, default=(12, 6)
        Figure size.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    model_names = list(models.keys())
    n_models = len(model_names)
    n_features = len(list(models.values())[0].weights)

    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]

    x = np.arange(n_features)
    width = 0.8 / n_models

    for i, (name, model) in enumerate(models.items()):
        offset = width * (i - n_models/2 + 0.5)
        ax.bar(x + offset, model.weights, width, label=name)

    ax.set_xlabel("Features")
    ax.set_ylabel("Coefficient Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def compare_models(X_test, y_test, models, title="Model Comparison", figsize=(12, 6), save_path=None):
    """
    Compare predictions from multiple models.

    Parameters
    ----------
    X_test : array-like
        Test features.
    y_test : array-like
        True test targets.
    models : dict
        Dictionary of {model_name: fitted_model}.
    title : str, default="Model Comparison"
        Plot title.
    figsize : tuple, default=(12, 6)
        Figure size.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    fig, axes = plt.subplots(1, len(models), figsize=figsize, sharey=True)

    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)

        # Scatter of predicted vs actual
        ax.scatter(y_test, y_pred, alpha=0.6, color="blue")

        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")

        # R² score
        from ..metrics import r2_score
        r2 = r2_score(y_test, y_pred)

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{name}\nR² = {r2:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig