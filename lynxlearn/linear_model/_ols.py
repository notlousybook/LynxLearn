"""
Ordinary Least Squares Linear Regression.
"""

import numpy as np
from scipy import stats

from ._base import BaseRegressor


class LinearRegression(BaseRegressor):
    """
    Ordinary Least Squares Linear Regression - the simplest and fastest method.

    This finds the best-fit line by solving the Normal Equation directly.
    Perfect for beginners and when you need quick, accurate results.

    Parameters
    ----------
    learn_bias : bool, default=True
        Whether to learn the bias/intercept term. Usually keep this True.
        (Also accepts `fit_intercept` for backward compatibility)
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    positive : bool, default=False
        Force coefficients to be positive.

    Attributes
    ----------
    coef_ : ndarray
        The learned coefficients (slopes). Alias for weights.
    intercept_ : float
        The learned intercept. Alias for bias.
    n_features_in_ : int
        Number of features seen during training.
    n_samples_ : int
        Number of training samples.
    std_errors_ : ndarray
        Standard errors for each coefficient.
    t_values_ : ndarray
        t-statistics for each coefficient.
    p_values_ : ndarray
        Two-tailed p-values for coefficient significance.
    leverage_ : ndarray
        Leverage values (hat matrix diagonal).
    cooks_distance_ : ndarray
        Cook's distance for influence detection.
    r2_ : float
        R-squared on training data.
    adj_r2_ : float
        Adjusted R-squared.
    mse_ : float
        Mean squared error on training data.
    rmse_ : float
        Root mean squared error on training data.
    aic_ : float
        Akaike Information Criterion.
    bic_ : float
        Bayesian Information Criterion.

    Examples
    --------
    >>> from lousybookml import LinearRegression
    >>> model = LinearRegression()
    >>> model.train(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> score = model.evaluate(X_test, y_test)
    """

    def __init__(
        self, learn_bias=True, copy_X=True, positive=False, fit_intercept=None
    ):
        super().__init__()
        # Backward compatibility: fit_intercept overrides learn_bias if provided
        if fit_intercept is not None:
            learn_bias = fit_intercept
        self.learn_bias = learn_bias
        self.fit_intercept = learn_bias  # Alias for backward compatibility
        self.copy_X = copy_X
        self.positive = positive

    def train(self, X, y):
        """
        Train the model on your data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Your training data (features).
        y : array-like of shape (n_samples,)
            Your training targets (what you want to predict).

        Returns
        -------
        self : LinearRegression
            The trained model (you can chain methods).

        Examples
        --------
        >>> model = LinearRegression()
        >>> model.train(X_train, y_train)
        >>> # Or chain: model.train(X, y).predict(X_new)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if self.copy_X:
            X = X.copy()

        self._X_train = X
        self._y_train = y
        self.n_samples_, self.n_features_in_ = X.shape

        if self.fit_intercept:
            X_b = np.c_[np.ones((self.n_samples_, 1)), X]
        else:
            X_b = X

        if self.positive:
            theta = self._solve_positive(X_b, y)
        else:
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta

        self.weights = self.coef_
        self.bias = self.intercept_

        self._is_trained = True
        self._compute_statistics(X_b, y)
        return self

    def _solve_positive(self, X_b, y):
        """Solve OLS with non-negative constraint using NNLS."""
        from scipy.optimize import nnls

        theta, _ = nnls(X_b, y)
        return theta

    def _compute_statistics(self, X_b, y):
        """Compute all statistical measures after fitting."""
        y_pred = self.predict(self._X_train)
        residuals = y - y_pred
        self._residuals = residuals

        n = self.n_samples_
        p = self.n_features_in_
        if self.fit_intercept:
            p += 1

        residual_var = np.sum(residuals**2) / (n - p) if n > p else 0

        try:
            XtX_inv = np.linalg.pinv(X_b.T @ X_b)
            var_theta = residual_var * XtX_inv
            self.std_errors_ = np.sqrt(np.diag(var_theta))[: self.n_features_in_]
        except:
            self.std_errors_ = np.full(self.n_features_in_, np.nan)

        # Use np.divide to handle division by zero gracefully
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = self.coef_ / self.std_errors_
        self.t_values_ = np.nan_to_num(t_values, nan=0.0, posinf=0.0, neginf=0.0)

        df = n - p
        self.p_values_ = (
            2 * (1 - stats.t.cdf(np.abs(self.t_values_), df))
            if df > 0
            else np.full(self.n_features_in_, np.nan)
        )

        H = X_b @ np.linalg.pinv(X_b.T @ X_b) @ X_b.T
        self.leverage_ = np.diag(H)

        mse_res = np.sum(residuals**2) / (n - p) if n > p else 0
        # Use np.errstate to handle division by zero gracefully
        with np.errstate(divide="ignore", invalid="ignore"):
            self.cooks_distance_ = (residuals**2 / (p * mse_res)) * (
                self.leverage_ / (1 - self.leverage_) ** 2
            )
        self.cooks_distance_ = np.nan_to_num(
            self.cooks_distance_, nan=0.0, posinf=0.0, neginf=0.0
        )

        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum(residuals**2)
        self.r2_ = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        self.adj_r2_ = 1 - (1 - self.r2_) * (n - 1) / (n - p) if n > p else self.r2_

        self.mse_ = ss_res / n
        self.rmse_ = np.sqrt(self.mse_)

        self.aic_ = n * np.log(self.mse_) + 2 * p if self.mse_ > 0 else np.nan
        self.bic_ = n * np.log(self.mse_) + p * np.log(n) if self.mse_ > 0 else np.nan

    def residuals(self):
        """Return training residuals."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained first!")
        return self._residuals.copy()

    def confidence_intervals(self, alpha=0.05):
        """
        Compute confidence intervals for coefficients.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level (0.05 = 95% CI).

        Returns
        -------
        ci : ndarray of shape (n_features, 2)
            Lower and upper bounds for each coefficient.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained first!")

        n = self.n_samples_
        p = self.n_features_in_ + (1 if self.fit_intercept else 0)
        df = n - p

        t_crit = stats.t.ppf(1 - alpha / 2, df) if df > 0 else 0

        lower = self.coef_ - t_crit * self.std_errors_
        upper = self.coef_ + t_crit * self.std_errors_

        return np.column_stack([lower, upper])

    def predict_interval(self, X, alpha=0.05):
        """
        Compute prediction intervals for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.
        alpha : float, default=0.05
            Significance level (0.05 = 95% prediction interval).

        Returns
        -------
        predictions : ndarray
            Predicted values.
        lower : ndarray
            Lower bound of prediction interval.
        upper : ndarray
            Upper bound of prediction interval.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained first!")

        X = np.asarray(X, dtype=np.float64)
        predictions = self.predict(X)

        n = self.n_samples_
        p = self.n_features_in_ + (1 if self.fit_intercept else 0)
        df = n - p

        if self.fit_intercept:
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
            X_train_b = np.c_[np.ones((n, 1)), self._X_train]
        else:
            X_b = X
            X_train_b = self._X_train

        mse = np.sum(self._residuals**2) / df if df > 0 else 0

        try:
            XtX_inv = np.linalg.pinv(X_train_b.T @ X_train_b)
            se_pred = np.sqrt(mse * (1 + np.sum((X_b @ XtX_inv) * X_b, axis=1)))
        except:
            se_pred = np.sqrt(mse) * np.ones(X.shape[0])

        t_crit = stats.t.ppf(1 - alpha / 2, df) if df > 0 else 1.96

        lower = predictions - t_crit * se_pred
        upper = predictions + t_crit * se_pred

        return predictions, lower, upper

    def vif(self):
        """
        Compute Variance Inflation Factor for multicollinearity detection.

        Returns
        -------
        vif : ndarray
            VIF for each feature. Values > 10 indicate high multicollinearity.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained first!")

        X = self._X_train
        n_features = self.n_features_in_

        if n_features < 2:
            return np.array([1.0])

        vif_values = []
        for i in range(n_features):
            X_other = np.delete(X, i, axis=1)
            y_i = X[:, i]

            theta = np.linalg.pinv(X_other.T @ X_other) @ X_other.T @ y_i
            y_pred = X_other @ theta

            ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
            ss_res = np.sum((y_i - y_pred) ** 2)
            r2_i = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            vif = 1 / (1 - r2_i) if (1 - r2_i) > 1e-10 else np.inf
            vif_values.append(vif)

        return np.array(vif_values)

    def summary_stats(self):
        """
        Print a statistical summary of the model.

        Shows coefficients, standard errors, t-values, p-values, and CIs.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained first!")

        print("=" * 60)
        print(f"Linear Regression Results")
        print("=" * 60)
        print(f"Observations: {self.n_samples_}")
        print(f"Features: {self.n_features_in_}")
        print(f"R²: {self.r2_:.4f}  |  Adj R²: {self.adj_r2_:.4f}")
        print(f"MSE: {self.mse_:.4f}  |  RMSE: {self.rmse_:.4f}")
        print(f"AIC: {self.aic_:.2f}  |  BIC: {self.bic_:.2f}")
        print("-" * 60)
        print(f"{'Feature':<12} {'Coef':>10} {'Std Err':>10} {'t':>8} {'p-value':>10}")
        print("-" * 60)

        ci = self.confidence_intervals()
        for i in range(self.n_features_in_):
            p_val = self.p_values_[i]
            sig = (
                "***"
                if p_val < 0.001
                else "**"
                if p_val < 0.01
                else "*"
                if p_val < 0.05
                else ""
            )
            print(
                f"x{i + 1:<10} {self.coef_[i]:>10.4f} {self.std_errors_[i]:>10.4f} {self.t_values_[i]:>8.2f} {p_val:>10.4f} {sig}"
            )

        if self.fit_intercept:
            print(f"{'Intercept':<12} {self.intercept_:>10.4f}")
        print("-" * 60)
        print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
        print("=" * 60)

    def __repr__(self):
        return f"LinearRegression(fit_intercept={self.fit_intercept})"
