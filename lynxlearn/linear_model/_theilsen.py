"""
Theil-Sen Regressor - Median-based robust regression.
"""

import numpy as np

from ._base import BaseRegressor
from ._ols import LinearRegression


class TheilSenRegressor(BaseRegressor):
    """
    Theil-Sen Estimator: a robust regression method.

    Computes the slope as the median of slopes between all pairs of points.
    More robust to outliers than OLS, but computationally expensive for
    large datasets.

    Parameters
    ----------
    n_subsamples : int or None, default=None
        Number of subsamples to compute. If None, uses all combinations.
    max_iter : int, default=300
        Maximum number of iterations for intercept calculation.
    tol : float, default=1e-3
        Tolerance for convergence.
    learn_bias : bool, default=True
        Whether to learn the bias term.
        (Also accepts `fit_intercept` for backward compatibility)
    random_state : int, default=None
        Random seed for reproducibility.
    n_jobs : int, default=1
        Number of jobs to run in parallel (not implemented, kept for API compatibility).

    Attributes
    ----------
    weights : ndarray
        Coefficients.
    bias : float
        Intercept.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(
        self,
        n_subsamples=None,
        max_iter=300,
        tol=1e-3,
        learn_bias=True,
        fit_intercept=None,
        random_state=None,
        n_jobs=1,
    ):
        super().__init__()
        # Backward compatibility: fit_intercept overrides learn_bias if provided
        if fit_intercept is not None:
            learn_bias = fit_intercept
        self.n_subsamples = n_subsamples
        self.max_iter = max_iter
        self.tol = tol
        self.learn_bias = learn_bias
        self.fit_intercept = learn_bias  # Alias for backward compatibility
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_iter_ = 0

    def train(self, X, y):
        """
        Train the Theil-Sen regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : TheilSenRegressor
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Center data if fitting intercept
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y
            X_mean = np.zeros(n_features)
            y_mean = 0.0

        # Determine number of subsamples
        if self.n_subsamples is None:
            # Use all pairs for small datasets
            max_pairs = n_samples * (n_samples - 1) // 2
            if max_pairs <= 10000:
                n_pairs = max_pairs
            else:
                n_pairs = 10000
        else:
            n_pairs = min(self.n_subsamples, n_samples * (n_samples - 1) // 2)

        # Compute slopes for each feature
        self.weights = np.zeros(n_features)

        for j in range(n_features):
            slopes = []

            # Generate random pairs
            indices = np.random.choice(n_samples, size=(n_pairs, 2), replace=True)

            for i in range(n_pairs):
                idx1, idx2 = indices[i]

                # Skip if x values are too close
                if abs(X_centered[idx1, j] - X_centered[idx2, j]) < 1e-10:
                    continue

                # Compute slope
                slope = (y_centered[idx1] - y_centered[idx2]) / (
                    X_centered[idx1, j] - X_centered[idx2, j]
                )
                slopes.append(slope)

            # Use median as the slope
            if slopes:
                self.weights[j] = np.median(slopes)
            else:
                self.weights[j] = 0.0

        # Compute intercept using median of residuals
        if self.fit_intercept:
            residuals = y_centered - X_centered @ self.weights
            self.bias = np.median(residuals) + y_mean
        else:
            self.bias = 0.0

        self.n_iter_ = 1
        self._is_trained = True
        return self

    def __repr__(self):
        return f"TheilSenRegressor(n_subsamples={self.n_subsamples})"
