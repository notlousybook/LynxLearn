"""
Weighted Least Squares and Generalized Least Squares.
"""

import numpy as np
from ._base import BaseRegressor


class WeightedLeastSquares(BaseRegressor):
    """
    Weighted Least Squares (WLS) regression.

    Handles heteroscedasticity by assigning different weights to observations.
    Observations with higher variance get lower weights.

    Parameters
    ----------
    weights : array-like or None, default=None
        Sample weights. If None, uses uniform weights.
    fit_intercept : bool, default=True
        Whether to fit the intercept.

    Attributes
    ----------
    weights_ : ndarray
        The sample weights used for fitting.
    """

    def __init__(self, weights=None, fit_intercept=True):
        super().__init__()
        self.weights = weights
        self.fit_intercept = fit_intercept
        self.weights_ = None

    def train(self, X, y):
        """
        Train the WLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : WeightedLeastSquares
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Set weights
        if self.weights is None:
            self.weights_ = np.ones(n_samples)
        else:
            self.weights_ = np.asarray(self.weights)

        # Normalize weights
        self.weights_ = self.weights_ / np.mean(self.weights_)

        # Compute weighted X and y
        sqrt_w = np.sqrt(self.weights_)
        X_weighted = X * sqrt_w[:, np.newaxis]
        y_weighted = y * sqrt_w

        # Fit weighted least squares
        if self.fit_intercept:
            # Add column of ones for intercept
            X_b = np.c_[np.ones(n_samples), X_weighted]
        else:
            X_b = X_weighted

        # Normal equation: theta = (X'X)^(-1)X'y
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_weighted

        if self.fit_intercept:
            self.bias = theta[0]
            self.weights = theta[1:]
        else:
            self.bias = 0.0
            self.weights = theta

        self._is_trained = True
        return self

    def __repr__(self):
        return f"WeightedLeastSquares(fit_intercept={self.fit_intercept})"


class GeneralizedLeastSquares(BaseRegressor):
    """
    Generalized Least Squares (GLS) regression.

    Handles correlated errors and heteroscedasticity by using a
    covariance matrix of the errors.

    Parameters
    ----------
    sigma : array-like or None, default=None
        Covariance matrix of the errors. If None, uses identity matrix.
    fit_intercept : bool, default=True
        Whether to fit the intercept.

    Attributes
    ----------
    sigma_ : ndarray
        The covariance matrix used for fitting.
    """

    def __init__(self, sigma=None, fit_intercept=True):
        super().__init__()
        self.sigma = sigma
        self.fit_intercept = fit_intercept
        self.sigma_ = None

    def train(self, X, y):
        """
        Train the GLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GeneralizedLeastSquares
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Set covariance matrix
        if self.sigma is None:
            self.sigma_ = np.eye(n_samples)
        else:
            self.sigma_ = np.asarray(self.sigma)

        # Compute Cholesky decomposition of sigma inverse
        # sigma_inv = L.T @ L where L is lower triangular
        try:
            L = np.linalg.cholesky(np.linalg.inv(self.sigma_))
        except np.linalg.LinAlgError:
            # Add small regularization if not positive definite
            sigma_reg = self.sigma_ + 1e-6 * np.eye(n_samples)
            L = np.linalg.cholesky(np.linalg.inv(sigma_reg))

        # Transform data
        X_transformed = L @ X
        y_transformed = L @ y

        # Fit OLS on transformed data
        if self.fit_intercept:
            X_b = np.c_[np.ones(n_samples), X_transformed]
        else:
            X_b = X_transformed

        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_transformed

        if self.fit_intercept:
            self.bias = theta[0]
            self.weights = theta[1:]
        else:
            self.bias = 0.0
            self.weights = theta

        self._is_trained = True
        return self

    def __repr__(self):
        return f"GeneralizedLeastSquares(fit_intercept={self.fit_intercept})"
