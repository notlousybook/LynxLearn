"""
Comprehensive regression analysis visualizations.

This module provides extensive visualization capabilities for regression models,
including residual analysis, Q-Q plots, and model diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_1d_regression(X, y, model, title="Linear Regression Visualization", save_path=None):
    """
    Create a comprehensive 1D regression visualization with multiple subplots.

    Parameters
    ----------
    X : array-like of shape (n_samples, 1) or (n_samples,)
        Feature data (must be 1D).
    y : array-like of shape (n_samples,)
        Target values.
    model : fitted model
        Any fitted regression model with predict() method.
    title : str, default="Linear Regression Visualization"
        Main title for the figure.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim > 1:
        X = X.flatten()

    # Create predictions
    y_pred = model.predict(X.reshape(-1, 1))
    residuals = y - y_pred

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Main scatter plot with regression line
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.scatter(X, y, alpha=0.6, color='blue', s=50, label='Data points', edgecolors='white', linewidth=0.5)

    # Regression line
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax1.plot(x_line, y_line, color='red', linewidth=3, label='Regression line')

    # Confidence interval (simple approximation)
    std_residual = np.std(residuals)
    ax1.fill_between(x_line, y_line - 2*std_residual, y_line + 2*std_residual,
                     alpha=0.2, color='red', label='±2σ confidence')

    ax1.set_xlabel('Feature (X)', fontsize=12)
    ax1.set_ylabel('Target (y)', fontsize=12)
    ax1.set_title('Regression Line with Data', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. Residuals plot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(y_pred, residuals, alpha=0.6, color='green', s=50, edgecolors='white', linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Residuals histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(residuals, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residual Value', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Q-Q plot for residuals normality
    ax4 = fig.add_subplot(gs[1, 1])
    try:
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    except ImportError:
        ax4.text(0.5, 0.5, 'scipy not available\nfor Q-Q plot', ha='center', va='center')

    # 5. Actual vs Predicted
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(y, y_pred, alpha=0.6, color='orange', s=50, edgecolors='white', linewidth=0.5)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
    ax5.set_xlabel('Actual Values', fontsize=11)
    ax5.set_ylabel('Predicted Values', fontsize=11)
    ax5.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Model parameters info box
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    # Calculate metrics
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Get model info
    model_type = type(model).__name__
    weights_str = f"{model.weights}" if hasattr(model, 'weights') and model.weights is not None else "N/A"
    bias_str = f"{model.bias:.6f}" if hasattr(model, 'bias') and model.bias is not None else "N/A"

    info_text = f"""
    Model Information:
    ─────────────────
    Model Type: {model_type}
    Weights (coefficients): {weights_str}
    Bias (intercept): {bias_str}

    Performance Metrics:
    ────────────────────
    Mean Squared Error (MSE): {mse:.6f}
    Root Mean Squared Error (RMSE): {rmse:.6f}
    Mean Absolute Error (MAE): {mae:.6f}
    R² Score: {r2:.6f}
    """

    ax6.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_all_metrics(y_true, y_pred, title="Complete Metrics Visualization", save_path=None):
    """
    Visualize all regression metrics in one comprehensive figure.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    title : str, default="Complete Metrics Visualization"
        Main title.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    # Calculate metrics
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Actual vs Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.6, color='blue', s=50)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax1.set_xlabel('Actual', fontsize=11)
    ax1.set_ylabel('Predicted', fontsize=11)
    ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Residuals vs Predicted
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_pred, residuals, alpha=0.6, color='green', s=50)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Residuals histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residual', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Q-Q plot
    ax4 = fig.add_subplot(gs[1, 0])
    try:
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    except ImportError:
        ax4.text(0.5, 0.5, 'scipy not available\nfor Q-Q plot', ha='center', va='center')

    # 5. Residuals vs Actual
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(y_true, residuals, alpha=0.6, color='orange', s=50)
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Actual', fontsize=11)
    ax5.set_ylabel('Residuals', fontsize=11)
    ax5.set_title('Residuals vs Actual', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Squared residuals
    ax6 = fig.add_subplot(gs[1, 2])
    squared_res = residuals**2
    ax6.scatter(y_pred, squared_res, alpha=0.6, color='red', s=50)
    ax6.set_xlabel('Predicted', fontsize=11)
    ax6.set_ylabel('Squared Residuals', fontsize=11)
    ax6.set_title('Squared Residuals vs Predicted', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 7. Metrics bar chart
    ax7 = fig.add_subplot(gs[2, :2])
    metrics_names = ['MSE', 'RMSE', 'MAE', 'R²']
    metrics_values = [mse, rmse, mae, r2]
    colors = ['red', 'orange', 'yellow', 'green']
    bars = ax7.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
    ax7.set_ylabel('Value', fontsize=11)
    ax7.set_title('Summary Metrics', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # 8. Metrics text summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    summary_text = f"""
    METRICS SUMMARY
    ═══════════════

    Mean Squared Error:
    {mse:.6f}

    Root Mean Squared Error:
    {rmse:.6f}

    Mean Absolute Error:
    {mae:.6f}

    R² Score:
    {r2:.6f}

    Residual Std Dev:
    {np.std(residuals):.6f}
    """

    ax8.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
