"""
Tests for linear regression models.
"""

import numpy as np
import pytest

from lynxlearn.linear_model import (
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
    SGDRegressor,
    Lars,
    LassoLars,
    OrthogonalMatchingPursuit,
    RANSACRegressor,
    TheilSenRegressor,
    ARDRegression,
    WeightedLeastSquares,
    GeneralizedLeastSquares,
    GeneralizedLinearModel,
    PoissonRegressor,
    GammaRegressor,
    TweedieRegressor,
    LinearSVR,
    IsotonicRegression,
    MultiTaskElasticNet,
    MultiTaskLasso,
)


class TestLinearRegression:
    """Tests for LinearRegression (OLS)."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting on simple data."""
        X, y, true_weights, true_bias = simple_regression_data

        model = LinearRegression()
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained
        assert np.abs(model.weights[0] - true_weights) < 0.5
        assert np.abs(model.bias - true_bias) < 0.5

    def test_predict_before_fit_raises(self):
        """Test that predict raises error before fitting."""
        model = LinearRegression()
        X = np.array([[1], [2], [3]])

        with pytest.raises(RuntimeError, match="Model must be trained first"):
            model.predict(X)

    def test_multi_feature_fit(self, multi_feature_data):
        """Test fitting on multi-feature data."""
        X, y, true_weights, true_bias = multi_feature_data

        model = LinearRegression()
        model.train(X, y)

        assert len(model.weights) == 5
        assert np.allclose(model.weights, true_weights, atol=0.5)
        assert np.abs(model.bias - true_bias) < 0.5

    def test_no_intercept(self, simple_regression_data):
        """Test fitting without intercept."""
        X, y, true_weights, true_bias = simple_regression_data

        model = LinearRegression(fit_intercept=False)
        model.train(X, y)

        assert model.bias == 0.0

    def test_score(self, train_test_data):
        """Test R² score calculation."""
        X_train, X_test, y_train, y_test, _, _ = train_test_data

        model = LinearRegression()
        model.train(X_train, y_train)
        score = model.score(X_test, y_test)

        assert 0 < score <= 1.0

    def test_evaluate_alias(self, train_test_data):
        """Test that evaluate is alias for score."""
        X_train, X_test, y_train, y_test, _, _ = train_test_data

        model = LinearRegression()
        model.train(X_train, y_train)

        score1 = model.score(X_test, y_test)
        score2 = model.evaluate(X_test, y_test)

        assert score1 == score2

    def test_get_params(self, simple_regression_data):
        """Test getting model parameters."""
        X, y, _, _ = simple_regression_data

        model = LinearRegression()
        model.train(X, y)
        params = model.get_params()

        assert "weights" in params
        assert "bias" in params
        assert params["weights"] is not None

    def test_repr(self):
        """Test string representation."""
        model = LinearRegression(fit_intercept=True)
        repr_str = repr(model)

        assert "LinearRegression" in repr_str
        assert "fit_intercept=True" in repr_str

    def test_sklearn_attributes(self, simple_regression_data):
        """Test sklearn-style attributes."""
        X, y, _, _ = simple_regression_data

        model = LinearRegression()
        model.train(X, y)

        assert hasattr(model, "coef_")
        assert hasattr(model, "intercept_")
        assert hasattr(model, "n_features_in_")
        assert hasattr(model, "n_samples_")
        assert np.allclose(model.coef_, model.weights)
        assert np.isclose(model.intercept_, model.bias)
        assert model.n_features_in_ == X.shape[1]
        assert model.n_samples_ == X.shape[0]

    def test_statistical_features(self, simple_regression_data):
        """Test statistical features."""
        X, y, _, _ = simple_regression_data

        model = LinearRegression()
        model.train(X, y)

        assert hasattr(model, "std_errors_")
        assert hasattr(model, "t_values_")
        assert hasattr(model, "p_values_")
        assert len(model.std_errors_) == X.shape[1]
        assert len(model.t_values_) == X.shape[1]
        assert len(model.p_values_) == X.shape[1]
        assert np.all(model.p_values_ >= 0)
        assert np.all(model.p_values_ <= 1)

    def test_confidence_intervals(self, simple_regression_data):
        """Test confidence intervals."""
        X, y, _, _ = simple_regression_data

        model = LinearRegression()
        model.train(X, y)

        ci = model.confidence_intervals()
        assert ci.shape == (X.shape[1], 2)
        assert np.all(ci[:, 0] <= ci[:, 1])

    def test_residuals(self, simple_regression_data):
        """Test residuals method."""
        X, y, _, _ = simple_regression_data

        model = LinearRegression()
        model.train(X, y)

        residuals = model.residuals()
        assert len(residuals) == len(y)
        assert np.allclose(residuals, y - model.predict(X))

    def test_leverage_and_cooks_distance(self, simple_regression_data):
        """Test leverage and Cook's distance."""
        X, y, _, _ = simple_regression_data

        model = LinearRegression()
        model.train(X, y)

        assert hasattr(model, "leverage_")
        assert hasattr(model, "cooks_distance_")
        assert len(model.leverage_) == len(y)
        assert len(model.cooks_distance_) == len(y)
        assert np.all(model.leverage_ >= 0)
        assert np.all(model.leverage_ <= 1)

    def test_vif(self, multi_feature_data):
        """Test Variance Inflation Factor."""
        X, y, _, _ = multi_feature_data

        model = LinearRegression()
        model.train(X, y)

        vif = model.vif()
        assert len(vif) == X.shape[1]
        assert np.all(vif >= 0.9)

    def test_model_quality_metrics(self, simple_regression_data):
        """Test model quality metrics."""
        X, y, _, _ = simple_regression_data

        model = LinearRegression()
        model.train(X, y)

        assert hasattr(model, "r2_")
        assert hasattr(model, "adj_r2_")
        assert hasattr(model, "mse_")
        assert hasattr(model, "rmse_")
        assert hasattr(model, "aic_")
        assert hasattr(model, "bic_")
        assert 0 <= model.r2_ <= 1
        assert model.adj_r2_ <= model.r2_
        assert model.rmse_ == np.sqrt(model.mse_)

    def test_predict_interval(self, simple_regression_data):
        """Test prediction intervals."""
        X, y, _, _ = simple_regression_data

        model = LinearRegression()
        model.train(X, y)

        pred, lower, upper = model.predict_interval(X)
        assert len(pred) == len(y)
        assert len(lower) == len(y)
        assert len(upper) == len(y)
        assert np.all(lower <= pred)
        assert np.all(pred <= upper)

    def test_copy_X(self, simple_regression_data):
        """Test copy_X parameter."""
        X, y, _, _ = simple_regression_data
        X_original = X.copy()

        model = LinearRegression(copy_X=True)
        model.train(X, y)
        assert np.allclose(X, X_original)

    def test_positive_constraint(self):
        """Test positive constraint forces non-negative coefficients."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.abs(X @ np.array([2, 1, 0.5])) + np.random.randn(100) * 0.1

        model = LinearRegression(positive=True)
        model.train(X, y)

        assert np.all(model.coef_ >= -1e-10)

    def test_summary_stats(self, simple_regression_data):
        """Test summary_stats method runs without error."""
        X, y, _, _ = simple_regression_data

        model = LinearRegression()
        model.train(X, y)

        model.summary_stats()


class TestGradientDescentRegressor:
    """Tests for GradientDescentRegressor."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, true_weights, true_bias = simple_regression_data

        model = GradientDescentRegressor(
            learning_rate=0.01, n_iterations=1000, tolerance=1e-6
        )
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained
        assert len(model.cost_history) > 0
        assert model.n_iter_ > 0

    def test_cost_history_decreases(self, simple_regression_data):
        """Test that cost decreases during training."""
        X, y, _, _ = simple_regression_data

        model = GradientDescentRegressor(learning_rate=0.01, n_iterations=500)
        model.train(X, y)

        initial_cost = model.cost_history[0]
        final_cost = model.cost_history[-1]

        assert final_cost < initial_cost

    def test_convergence(self, simple_regression_data):
        """Test that model converges."""
        X, y, true_weights, true_bias = simple_regression_data

        model = GradientDescentRegressor(
            learning_rate=0.01, n_iterations=2000, tolerance=1e-8
        )
        model.train(X, y)

        # Should converge before max iterations
        assert model.n_iter_ < 2000

        # Weights should be close to true values
        assert np.abs(model.weights[0] - true_weights) < 0.5
        assert np.abs(model.bias - true_bias) < 0.5

    def test_learning_rate_too_high(self):
        """Test that very high learning rate causes divergence."""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 2 * X.flatten() + 1 + np.random.randn(50) * 0.1

        # Use extremely high learning rate to force divergence
        model = GradientDescentRegressor(learning_rate=10.0, n_iterations=50)
        model.train(X, y)

        # Cost should increase or become NaN with very high learning rate
        final_cost = model.cost_history[-1]
        initial_cost = model.cost_history[0]

        # Either diverged (NaN) or cost increased significantly
        assert np.isnan(final_cost) or final_cost > initial_cost * 10


class TestRidge:
    """Tests for Ridge regression."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, true_weights, true_bias = simple_regression_data

        model = Ridge(alpha=1.0)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained

    def test_regularization_effect(self, multi_feature_data):
        """Test that higher alpha shrinks weights more."""
        X, y, _, _ = multi_feature_data

        model_low = Ridge(alpha=0.01)
        model_low.train(X, y)

        model_high = Ridge(alpha=100.0)
        model_high.train(X, y)

        # Higher alpha should result in smaller weights
        assert np.linalg.norm(model_high.weights) < np.linalg.norm(model_low.weights)

    def test_alpha_zero_equals_ols(self, simple_regression_data):
        """Test that alpha=0 is equivalent to OLS."""
        X, y, _, _ = simple_regression_data

        ridge = Ridge(alpha=0.0)
        ridge.train(X, y)

        ols = LinearRegression()
        ols.train(X, y)

        # Should be very close (numerical differences may exist)
        assert np.allclose(ridge.weights, ols.weights, atol=1e-10)
        assert np.allclose(ridge.bias, ols.bias, atol=1e-10)


class TestLasso:
    """Tests for Lasso regression."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = Lasso(alpha=0.1)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained

    def test_feature_selection(self):
        """Test that Lasso performs feature selection."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        # Only 3 features are actually relevant
        X = np.random.randn(n_samples, n_features)
        true_weights = np.array([2.0, -1.5, 0.5] + [0.0] * 7)
        y = X @ true_weights + np.random.randn(n_samples) * 0.1

        model = Lasso(alpha=0.5)
        model.train(X, y)

        # Many weights should be zero
        n_zero = np.sum(np.abs(model.weights) < 1e-5)
        assert n_zero >= 5  # At least 5 features should be zeroed out

    def test_sparsity_increases_with_alpha(self):
        """Test that higher alpha creates more sparsity."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.1

        model_low = Lasso(alpha=0.01)
        model_low.train(X, y)

        model_high = Lasso(alpha=10.0)
        model_high.train(X, y)

        n_nonzero_low = np.sum(np.abs(model_low.weights) > 1e-5)
        n_nonzero_high = np.sum(np.abs(model_high.weights) > 1e-5)

        assert n_nonzero_high <= n_nonzero_low


class TestElasticNet:
    """Tests for ElasticNet regression."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained

    def test_l1_ratio_zero_is_l2_only(self, multi_feature_data):
        """Test that l1_ratio=0 uses only L2 regularization (no L1 sparsity)."""
        X, y, _, _ = multi_feature_data

        # ElasticNet with l1_ratio=0 should be L2 only (no feature selection)
        elastic = ElasticNet(alpha=0.1, l1_ratio=0.0, max_iter=2000, tol=1e-6)
        elastic.train(X, y)

        # With pure L2, all weights should be non-zero (no sparsity)
        n_nonzero = np.sum(np.abs(elastic.weights) > 1e-10)
        assert n_nonzero == len(elastic.weights), (
            "L2-only should not produce sparse weights"
        )

        # Should shrink weights compared to OLS
        ols = LinearRegression()
        ols.train(X, y)
        assert np.linalg.norm(elastic.weights) < np.linalg.norm(ols.weights)

    def test_l1_ratio_one_equals_lasso(self, multi_feature_data):
        """Test that l1_ratio=1 is equivalent to Lasso."""
        X, y, _, _ = multi_feature_data

        elastic = ElasticNet(alpha=0.1, l1_ratio=1.0, max_iter=2000)
        elastic.train(X, y)

        lasso = Lasso(alpha=0.1, max_iter=2000)
        lasso.train(X, y)

        # Should be close
        assert np.allclose(elastic.weights, lasso.weights, atol=0.1)


class TestPolynomialRegression:
    """Tests for PolynomialRegression."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = PolynomialRegression(degree=2)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained

    def test_higher_degree_better_fit(self):
        """Test that higher degree can fit complex patterns."""
        np.random.seed(42)
        X = np.linspace(-3, 3, 100).reshape(-1, 1)
        y = X.flatten() ** 3 + np.random.randn(100) * 0.5

        model_linear = PolynomialRegression(degree=1)
        model_linear.train(X, y)
        y_pred_linear = model_linear.predict(X)
        mse_linear = np.mean((y - y_pred_linear) ** 2)

        model_cubic = PolynomialRegression(degree=3)
        model_cubic.train(X, y)
        y_pred_cubic = model_cubic.predict(X)
        mse_cubic = np.mean((y - y_pred_cubic) ** 2)

        assert mse_cubic < mse_linear


class TestPolynomialFeatures:
    """Tests for PolynomialFeatures transformer."""

    def test_degree_2(self):
        """Test degree 2 polynomial features."""
        X = np.array([[2, 3], [3, 4]])

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        # Should have: 1 (bias), x1, x2, x1^2, x1*x2, x2^2 = 6 features
        assert X_poly.shape == (2, 6)

        # Check specific values
        assert X_poly[0, 0] == 1.0  # bias
        assert X_poly[0, 1] == 2.0  # x1
        assert X_poly[0, 2] == 3.0  # x2
        assert X_poly[0, 3] == 4.0  # x1^2
        assert X_poly[0, 4] == 6.0  # x1*x2
        assert X_poly[0, 5] == 9.0  # x2^2

    def test_no_bias(self):
        """Test without bias term."""
        X = np.array([[2, 3]])

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Should have: x1, x2, x1^2, x1*x2, x2^2 = 5 features
        assert X_poly.shape == (1, 5)


class TestHuberRegressor:
    """Tests for HuberRegressor."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = HuberRegressor(epsilon=1.35)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained

    def test_robust_to_outliers(self):
        """Test that Huber is more robust to outliers than OLS."""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 2 * X.flatten() + 1 + np.random.randn(50) * 0.1

        # Add outliers
        y[0] += 50
        y[1] -= 50

        ols = LinearRegression()
        ols.train(X, y)

        huber = HuberRegressor(epsilon=1.35)
        huber.train(X, y)

        # Huber should be closer to true weights (2.0) than OLS
        assert np.abs(huber.weights[0] - 2.0) < np.abs(ols.weights[0] - 2.0)


class TestQuantileRegressor:
    """Tests for QuantileRegressor."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = QuantileRegressor(quantile=0.5)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained

    def test_different_quantiles(self):
        """Test fitting at different quantiles."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = 2 * X.flatten() + 1 + np.random.randn(100) * 2

        model_low = QuantileRegressor(quantile=0.1)
        model_low.train(X, y)

        model_high = QuantileRegressor(quantile=0.9)
        model_high.train(X, y)

        # Higher quantile should have higher predictions on average
        y_pred_low = model_low.predict(X)
        y_pred_high = model_high.predict(X)

        assert np.mean(y_pred_high) > np.mean(y_pred_low)


class TestBayesianRidge:
    """Tests for BayesianRidge."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = BayesianRidge()
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained

    def test_predict_with_std(self, simple_regression_data):
        """Test prediction with uncertainty."""
        X, y, _, _ = simple_regression_data

        model = BayesianRidge()
        model.train(X, y)

        y_pred, y_std = model.predict(X, return_std=True)

        assert len(y_pred) == len(X)
        assert len(y_std) == len(X)
        assert np.all(y_std >= 0)  # Standard deviation should be non-negative


class TestSGDRegressor:
    """Tests for SGDRegressor."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = SGDRegressor(max_iter=100, learning_rate=0.01)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained

    def test_l1_penalty(self, simple_regression_data):
        """Test L1 penalty creates sparse weights."""
        X, y, _, _ = simple_regression_data

        model = SGDRegressor(penalty="l1", alpha=0.1, max_iter=100)
        model.train(X, y)

        # With L1, some weights might be zero
        assert model.weights is not None


class TestLars:
    """Tests for Lars."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = Lars(n_nonzero_coefs=1)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained
        assert np.sum(model.weights != 0) <= 1  # At most 1 non-zero weight


class TestLassoLars:
    """Tests for LassoLars."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = LassoLars(alpha=0.1)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained


class TestOrthogonalMatchingPursuit:
    """Tests for OrthogonalMatchingPursuit."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = OrthogonalMatchingPursuit(n_nonzero_coefs=1)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained
        assert np.sum(model.weights != 0) <= 1


class TestRANSACRegressor:
    """Tests for RANSACRegressor."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = RANSACRegressor(max_trials=50)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained
        assert model.inlier_mask_ is not None

    def test_robust_to_outliers(self):
        """Test robustness to outliers."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = 2 * X.flatten() + 1 + np.random.randn(100) * 0.1

        # Add outliers
        y[0] += 50
        y[1] -= 50

        ransac = RANSACRegressor(max_trials=50)
        ransac.train(X, y)

        ols = LinearRegression()
        ols.train(X, y)

        # RANSAC should be closer to true weights (2.0)
        assert np.abs(ransac.weights[0] - 2.0) < np.abs(ols.weights[0] - 2.0)


class TestTheilSenRegressor:
    """Tests for TheilSenRegressor."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = TheilSenRegressor(n_subsamples=100)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained


class TestARDRegression:
    """Tests for ARDRegression."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = ARDRegression(n_iter=50)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained
        assert model.alpha_ is not None


class TestWeightedLeastSquares:
    """Tests for WeightedLeastSquares."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = WeightedLeastSquares()
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained

    def test_with_weights(self, simple_regression_data):
        """Test fitting with custom weights."""
        X, y, _, _ = simple_regression_data

        weights = np.ones(len(y))
        weights[:10] = 2.0  # Higher weight for first 10 samples

        model = WeightedLeastSquares(weights=weights)
        model.train(X, y)

        assert model.weights is not None
        assert model._is_trained


class TestGeneralizedLeastSquares:
    """Tests for GeneralizedLeastSquares."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = GeneralizedLeastSquares()
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained


class TestGeneralizedLinearModel:
    """Tests for GeneralizedLinearModel."""

    def test_gaussian_family(self, simple_regression_data):
        """Test Gaussian family (similar to OLS)."""
        X, y, _, _ = simple_regression_data

        model = GeneralizedLinearModel(family="gaussian", max_iter=50)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained


class TestPoissonRegressor:
    """Tests for PoissonRegressor."""

    def test_basic_fit(self):
        """Test basic fitting on count data."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = np.exp(2 * X.flatten() + 1)  # Poisson-like positive data

        model = PoissonRegressor(max_iter=50)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained


class TestGammaRegressor:
    """Tests for GammaRegressor."""

    def test_basic_fit(self):
        """Test basic fitting on positive data."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = np.exp(2 * X.flatten() + 1)  # Positive data

        model = GammaRegressor(max_iter=50)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained


class TestTweedieRegressor:
    """Tests for TweedieRegressor."""

    def test_basic_fit(self):
        """Test basic fitting."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = np.exp(2 * X.flatten() + 1)  # Positive data

        model = TweedieRegressor(power=1.5, max_iter=50)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained


class TestLinearSVR:
    """Tests for LinearSVR."""

    def test_basic_fit(self, simple_regression_data):
        """Test basic fitting."""
        X, y, _, _ = simple_regression_data

        model = LinearSVR(epsilon=0.1, C=1.0, max_iter=100)
        model.train(X, y)

        assert model.weights is not None
        assert model.bias is not None
        assert model._is_trained


class TestIsotonicRegression:
    """Tests for IsotonicRegression."""

    def test_basic_fit(self):
        """Test basic fitting."""
        np.random.seed(42)
        X = np.random.randn(100)
        y = X + np.random.randn(100) * 0.1

        model = IsotonicRegression()
        model.train(X, y)

        assert model.X_thresholds_ is not None
        assert model.y_thresholds_ is not None
        assert model._is_trained

    def test_monotonic_predictions(self):
        """Test that predictions are monotonic."""
        X = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 3, 2, 4, 5])

        model = IsotonicRegression(increasing=True)
        model.train(X, y)

        predictions = model.predict(X)
        # Predictions should be non-decreasing
        assert np.all(np.diff(predictions) >= -1e-10)


class TestMultiTaskElasticNet:
    """Tests for MultiTaskElasticNet."""

    def test_basic_fit(self):
        """Test basic fitting on multi-task data."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 3)  # 3 tasks

        model = MultiTaskElasticNet(alpha=0.1, max_iter=100)
        model.train(X, y)

        assert model.weights.shape == (5, 3)
        assert model.bias.shape == (3,)
        assert model._is_trained

    def test_predict_shape(self):
        """Test prediction shape."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100, 3)
        X_test = np.random.randn(20, 5)

        model = MultiTaskElasticNet(alpha=0.1, max_iter=100)
        model.train(X_train, y_train)

        predictions = model.predict(X_test)
        assert predictions.shape == (20, 3)


class TestMultiTaskLasso:
    """Tests for MultiTaskLasso."""

    def test_basic_fit(self):
        """Test basic fitting on multi-task data."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 3)  # 3 tasks

        model = MultiTaskLasso(alpha=0.1, max_iter=100)
        model.train(X, y)

        assert model.weights.shape == (5, 3)
        assert model.bias.shape == (3,)
        assert model._is_trained

    def test_predict_shape(self):
        """Test prediction shape."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100, 3)
        X_test = np.random.randn(20, 5)

        model = MultiTaskLasso(alpha=0.1, max_iter=100)
        model.train(X_train, y_train)

        predictions = model.predict(X_test)
        assert predictions.shape == (20, 3)
