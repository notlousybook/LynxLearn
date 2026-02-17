"""
Bayesian Linear Regression with automatic relevance determination.
"""

import numpy as np
from ._base import BaseRegressor


class BayesianRidge(BaseRegressor):
    """
    Bayesian Ridge Regression with automatic hyperparameter tuning.

    Estimates a probabilistic linear model with Gaussian priors on weights.
    Automatically tunes the regularization strength (alpha) and noise level.

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations for hyperparameter estimation.
    tol : float, default=1e-3
        Convergence tolerance.
    alpha_1 : float, default=1e-6
        Shape parameter for Gamma prior on alpha (precision of weights).
    alpha_2 : float, default=1e-6
        Rate parameter for Gamma prior on alpha.
    lambda_1 : float, default=1e-6
        Shape parameter for Gamma prior on lambda (precision of noise).
    lambda_2 : float, default=1e-6
        Rate parameter for Gamma prior on lambda.
    fit_intercept : bool, default=True
        Whether to fit the intercept.

    Attributes
    ----------
    weights : ndarray
        Mean of the posterior distribution over weights.
    bias : float
        Intercept.
    alpha_ : float
        Estimated precision of weights.
    lambda_ : float
        Estimated precision of noise.
    sigma_ : ndarray
        Covariance matrix of the posterior.
    """

    def __init__(self, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6,
                 lambda_1=1e-6, lambda_2=1e-6, fit_intercept=True):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.fit_intercept = fit_intercept

    def train(self, X, y):
        """
        Train the Bayesian Ridge regression using evidence maximization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : BayesianRidge
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Center data
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y

        # Initialize hyperparameters
        alpha = 1.0  # Precision of weights
        lambda_prec = 1.0  # Precision of noise

        # Precompute X.T @ X
        XTX = X_centered.T @ X_centered

        for iteration in range(self.max_iter):
            # Compute posterior covariance and mean
            sigma_inv = lambda_prec * XTX + alpha * np.eye(n_features)
            sigma = np.linalg.inv(sigma_inv)
            self.weights = lambda_prec * sigma @ X_centered.T @ y_centered

            # Update hyperparameters using evidence maximization
            # Update alpha (weight precision)
            gamma = np.sum(lambda_prec * np.diag(sigma) / alpha)
            alpha = (gamma + 2 * self.alpha_1) / (np.sum(self.weights ** 2) + 2 * self.alpha_2)

            # Update lambda (noise precision)
            residuals = y_centered - X_centered @ self.weights
            lambda_prec = (n_samples + 2 * self.lambda_1) / (np.sum(residuals ** 2) + 2 * self.lambda_2)

            # Check convergence
            if iteration > 0:
                if abs(alpha - alpha_old) < self.tol and abs(lambda_prec - lambda_old) < self.tol:
                    break

            alpha_old = alpha
            lambda_old = lambda_prec

        # Store final values
        self.alpha_ = alpha
        self.lambda_ = lambda_prec
        self.sigma_ = sigma

        # Compute intercept
        if self.fit_intercept:
            self.bias = y_mean - X_mean @ self.weights
        else:
            self.bias = 0.0

        self._is_trained = True
        return self

    def predict(self, X, return_std=False):
        """
        Predict with optional uncertainty estimation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        return_std : bool, default=False
            If True, return standard deviation of predictive distribution.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        y_std : ndarray of shape (n_samples,)
            Standard deviation (if return_std=True).
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained first! Call model.train(X, y)")

        X = np.asarray(X)
        y_pred = X @ self.weights + self.bias

        if return_std:
            # Compute predictive variance
            X_centered = X - np.mean(X, axis=0) if self.fit_intercept else X
            var = 1.0 / self.lambda_ + np.sum(X_centered @ self.sigma_ * X_centered, axis=1)
            return y_pred, np.sqrt(var)

        return y_pred

    def __repr__(self):
        return f"BayesianRidge(max_iter={self.max_iter}, tol={self.tol})"
