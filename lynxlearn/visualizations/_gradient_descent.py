"""
Gradient descent visualization and analysis.

This module provides specialized visualizations for gradient descent
optimization, including convergence analysis and learning curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_gradient_descent(X, y, gd_model, title="Gradient Descent Analysis", save_path=None):
    """
    Comprehensive visualization for Gradient Descent Regressor.

    Parameters
    ----------
    X : array-like
        Feature data.
    y : array-like
        Target values.
    gd_model : GradientDescentRegressor
        Fitted gradient descent model.
    title : str, default="Gradient Descent Analysis"
        Main title.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim > 1 and X.shape[1] == 1:
        X = X.flatten()

    y_pred = gd_model.predict(X.reshape(-1, 1) if X.ndim == 1 else X)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Cost history
    ax1 = fig.add_subplot(gs[0, 0])
    iterations = range(1, len(gd_model.cost_history) + 1)
    ax1.plot(iterations, gd_model.cost_history, linewidth=2, color='blue')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Cost (MSE)', fontsize=11)
    ax1.set_title('Cost Function Convergence', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add convergence point annotation
    final_cost = gd_model.cost_history[-1]
    ax1.axhline(y=final_cost, color='red', linestyle='--', alpha=0.5)
    ax1.text(len(gd_model.cost_history) * 0.7, final_cost * 1.1,
             f'Final Cost: {final_cost:.6f}', fontsize=9, color='red')

    # 2. Cost history (log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(iterations, gd_model.cost_history, linewidth=2, color='green')
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Cost (MSE) - Log Scale', fontsize=11)
    ax2.set_title('Cost Convergence (Log Scale)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Learning curve (cost reduction rate)
    ax3 = fig.add_subplot(gs[0, 2])
    if len(gd_model.cost_history) > 1:
        cost_diff = np.diff(gd_model.cost_history)
        ax3.plot(range(2, len(gd_model.cost_history) + 1), cost_diff, linewidth=2, color='purple')
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Cost Change', fontsize=11)
        ax3.set_title('Cost Reduction per Iteration', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 4. Regression plot (if 1D)
    ax4 = fig.add_subplot(gs[1, 0])
    if X.ndim == 1:
        ax4.scatter(X, y, alpha=0.6, color='blue', s=50, label='Data')
        x_line = np.linspace(X.min(), X.max(), 100)
        y_line = gd_model.predict(x_line.reshape(-1, 1))
        ax4.plot(x_line, y_line, color='red', linewidth=2, label='GD Regression')
        ax4.set_xlabel('Feature', fontsize=11)
        ax4.set_ylabel('Target', fontsize=11)
        ax4.set_title('Gradient Descent Fit', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. Residuals
    ax5 = fig.add_subplot(gs[1, 1])
    residuals = y - y_pred
    ax5.scatter(y_pred, residuals, alpha=0.6, color='orange', s=50)
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Predicted', fontsize=11)
    ax5.set_ylabel('Residuals', fontsize=11)
    ax5.set_title('Residuals Plot', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Model info
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    info_text = f"""
    Gradient Descent Configuration:
    ───────────────────────────────
    Learning Rate: {gd_model.learning_rate}
    Max Iterations: {gd_model.n_iterations}
    Actual Iterations: {gd_model.n_iter_}
    Tolerance: {gd_model.tolerance}
    Fit Intercept: {gd_model.fit_intercept}

    Final Parameters:
    ─────────────────
    Weights: {gd_model.weights}
    Bias: {gd_model.bias:.6f}

    Convergence:
    ────────────
    Initial Cost: {gd_model.cost_history[0]:.6f}
    Final Cost: {gd_model.cost_history[-1]:.6f}
    Cost Reduction: {gd_model.cost_history[0] - gd_model.cost_history[-1]:.6f}
    """

    ax6.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
