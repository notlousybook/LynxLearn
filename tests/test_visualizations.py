"""
Tests for visualizations module.
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lynxlearn import visualizations
from lynxlearn.linear_model import LinearRegression, GradientDescentRegressor, Ridge


@pytest.fixture(autouse=True)
def close_plots():
    """Close all plots after each test."""
    yield
    plt.close("all")


class TestBasicPlots:
    """Tests for basic plotting functions."""

    def test_plot_regression_1d(self):
        """Test 1D regression plot."""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 2 * X.flatten() + 1 + np.random.randn(50) * 0.1

        model = LinearRegression()
        model.train(X, y)

        fig = visualizations.plot_regression(X, y, model)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_cost_history(self):
        """Test cost history plot."""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 2 * X.flatten() + 1 + np.random.randn(50) * 0.1

        model = GradientDescentRegressor(n_iterations=100)
        model.train(X, y)

        fig = visualizations.plot_cost_history(model)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_cost_history_raises_without_history(self):
        """Test that plot_cost_history raises for model without cost_history."""
        model = LinearRegression()

        with pytest.raises(ValueError, match="cost_history"):
            visualizations.plot_cost_history(model)

    def test_plot_residuals(self):
        """Test residuals plot."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        fig = visualizations.plot_residuals(y_true, y_pred)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_coefficients(self):
        """Test coefficients comparison plot."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ [1, 2, 3] + np.random.randn(50) * 0.1

        model1 = LinearRegression()
        model1.train(X, y)

        model2 = Ridge(alpha=1.0)
        model2.train(X, y)

        models = {"OLS": model1, "Ridge": model2}

        fig = visualizations.plot_coefficients(models, feature_names=["A", "B", "C"])

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_compare_models(self):
        """Test model comparison plot."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ [1, 2] + np.random.randn(50) * 0.1

        model1 = LinearRegression()
        model1.train(X, y)

        model2 = Ridge(alpha=1.0)
        model2.train(X, y)

        models = {"OLS": model1, "Ridge": model2}

        fig = visualizations.compare_models(X, y, models)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)


class TestComprehensiveVisualizations:
    """Tests for comprehensive visualization functions."""

    def test_visualize_1d_regression(self):
        """Test comprehensive 1D regression visualization."""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 2 * X.flatten() + 1 + np.random.randn(50) * 0.1

        model = LinearRegression()
        model.train(X, y)

        fig = visualizations.visualize_1d_regression(X, y, model)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_visualize_all_metrics(self):
        """Test comprehensive metrics visualization."""
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_pred = y_true + np.random.randn(10) * 0.1

        fig = visualizations.visualize_all_metrics(y_true, y_pred)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_visualize_gradient_descent(self):
        """Test gradient descent visualization."""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 2 * X.flatten() + 1 + np.random.randn(50) * 0.1

        model = GradientDescentRegressor(n_iterations=100)
        model.train(X, y)

        fig = visualizations.visualize_gradient_descent(X, y, model)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_visualize_ridge_comparison(self):
        """Test ridge comparison visualization."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ [1, 2, 3] + np.random.randn(100) * 0.1

        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        fig = visualizations.visualize_ridge_comparison(
            X_train, y_train, X_test, y_test, alphas=[0.01, 0.1, 1.0, 10.0]
        )

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_create_comprehensive_report(self):
        """Test comprehensive report generation."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X @ [1, 2] + np.random.randn(100) * 0.1

        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        model1 = LinearRegression()
        model1.train(X_train, y_train)

        model2 = Ridge(alpha=1.0)
        model2.train(X_train, y_train)

        models = {"OLS": model1, "Ridge": model2}

        report = visualizations.create_comprehensive_report(
            X_train, y_train, X_test, y_test, models
        )

        assert "results" in report
        assert "figures" in report
        assert len(report["results"]) == 2


class TestVisualizationAliases:
    """Tests for backward-compatible aliases."""

    def test_viz_1d_alias(self):
        """Test viz_1d alias."""
        assert visualizations.viz_1d is visualizations.visualize_1d_regression

    def test_viz_gd_alias(self):
        """Test viz_gd alias."""
        assert visualizations.viz_gd is visualizations.visualize_gradient_descent

    def test_viz_ridge_alias(self):
        """Test viz_ridge alias."""
        assert visualizations.viz_ridge is visualizations.visualize_ridge_comparison

    def test_viz_metrics_alias(self):
        """Test viz_metrics alias."""
        assert visualizations.viz_metrics is visualizations.visualize_all_metrics

    def test_viz_report_alias(self):
        """Test viz_report alias."""
        assert visualizations.viz_report is visualizations.create_comprehensive_report
