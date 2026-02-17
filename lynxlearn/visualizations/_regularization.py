"""
Regularization visualization and comparison.

This module provides visualizations for comparing different regularization
strengths and their effects on model coefficients and performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_ridge_comparison(X_train, y_train, X_test, y_test, alphas=None, 
                                title="Ridge Regularization Comparison", save_path=None):
    """
    Compare Ridge regression with different alpha values.

    Parameters
    ----------
    X_train, y_train : array-like
        Training data.
    X_test, y_test : array-like
        Test data.
    alphas : list, optional
        List of alpha values to compare. Default: [0.001, 0.01, 0.1, 1, 10, 100]
    title : str, default="Ridge Regularization Comparison"
        Main title.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    from ..linear_model import Ridge
    from ..metrics import mean_squared_error, r2_score

    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1, 10, 100]

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    results = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({
            'alpha': alpha,
            'mse': mse,
            'r2': r2,
            'weights': model.weights.copy(),
            'bias': model.bias
        })

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. MSE vs Alpha
    ax1 = fig.add_subplot(gs[0, 0])
    mses = [r['mse'] for r in results]
    ax1.semilogx(alphas, mses, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Alpha (Regularization Strength)', fontsize=11)
    ax1.set_ylabel('Mean Squared Error', fontsize=11)
    ax1.set_title('MSE vs Regularization Strength', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Highlight best alpha
    best_idx = np.argmin(mses)
    ax1.plot(alphas[best_idx], mses[best_idx], 'ro', markersize=12, label=f'Best: α={alphas[best_idx]}')
    ax1.legend()

    # 2. R² vs Alpha
    ax2 = fig.add_subplot(gs[0, 1])
    r2s = [r['r2'] for r in results]
    ax2.semilogx(alphas, r2s, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Alpha (Regularization Strength)', fontsize=11)
    ax2.set_ylabel('R² Score', fontsize=11)
    ax2.set_title('R² vs Regularization Strength', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Coefficient paths
    ax3 = fig.add_subplot(gs[1, 0])
    n_features = len(results[0]['weights'])
    for i in range(n_features):
        coef_path = [r['weights'][i] for r in results]
        ax3.semilogx(alphas, coef_path, 'o-', linewidth=2, label=f'Feature {i+1}')
    ax3.set_xlabel('Alpha (Regularization Strength)', fontsize=11)
    ax3.set_ylabel('Coefficient Value', fontsize=11)
    ax3.set_title('Coefficient Paths vs Regularization', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # 4. Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    table_text = "Alpha      | MSE       | R²        | Weights\n"
    table_text += "─" * 70 + "\n"
    for r in results:
        weights_str = str([f"{w:.3f}" for w in r['weights']])
        table_text += f"{r['alpha']:<10.3f} | {r['mse']:<9.4f} | {r['r2']:<9.4f} | {weights_str}\n"

    ax4.text(0.1, 0.5, table_text, fontsize=9, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
