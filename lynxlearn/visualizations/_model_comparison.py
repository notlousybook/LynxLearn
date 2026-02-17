"""
Model comparison and reporting visualizations.

This module provides comprehensive model comparison reports and
side-by-side visualizations for multiple regression models.
"""

import numpy as np
import matplotlib.pyplot as plt


def create_comprehensive_report(X_train, y_train, X_test, y_test, models_dict, save_dir=None):
    """
    Create a comprehensive report comparing multiple models.

    Parameters
    ----------
    X_train, y_train : array-like
        Training data.
    X_test, y_test : array-like
        Test data.
    models_dict : dict
        Dictionary of {model_name: fitted_model}.
    save_dir : str, optional
        Directory to save all figures.

    Returns
    -------
    results : dict
        Dictionary with all results and figures.
    """
    from ..metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
    from ._regression_analysis import visualize_1d_regression

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    results = {}
    figures = {}

    # Create comparison figure
    fig, axes = plt.subplots(2, len(models_dict), figsize=(5*len(models_dict), 10))
    if len(models_dict) == 1:
        axes = axes.reshape(-1, 1)

    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = root_mean_squared_error(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        results[name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'weights': model.weights.copy() if hasattr(model, 'weights') else None,
            'bias': model.bias if hasattr(model, 'bias') else None
        }

        # Plot 1: Actual vs Predicted
        ax1 = axes[0, idx]
        ax1.scatter(y_test, y_pred_test, alpha=0.6, s=50)
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'{name}\nR² = {test_r2:.4f}')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residuals
        ax2 = axes[1, idx]
        residuals = y_test - y_pred_test
        ax2.scatter(y_pred_test, residuals, alpha=0.6, s=50, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{name}\nRMSE = {test_rmse:.4f}')
        ax2.grid(True, alpha=0.3)

    plt.suptitle('Model Comparison Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    figures['comparison'] = fig

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f'{save_dir}/model_comparison.png', dpi=150, bbox_inches='tight')

        # Save individual model visualizations
        for name, model in models_dict.items():
            if X_test.ndim == 1 or (X_test.ndim == 2 and X_test.shape[1] == 1):
                fig_individual = visualize_1d_regression(X_test, y_test, model, title=f'{name} Visualization')
                fig_individual.savefig(f'{save_dir}/{name.lower().replace(" ", "_")}_viz.png', dpi=150, bbox_inches='tight')
                plt.close(fig_individual)

    return {'results': results, 'figures': figures}
