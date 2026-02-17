"""
Automatic Relevance Determination (ARD) Regression.
"""

import numpy as np
from ._base import BaseRegressor


class ARDRegression(BaseRegressor):
    """
    Bayesian ARD Regression with Automatic Relevance Determination.

    Similar to Bayesian Ridge but with separate precision parameters
    for each weight, allowing for automatic feature selection.
    Weights with low precision are effectively removed.

    Parameters
    ----------
    n_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-3
        Tolerance for convergence.
    alpha_1 : float, default=1e-6
        Shape parameter for Gamma prior on alpha.
    alpha_2 : float, default=1e-6
        Rate parameter for Gamma prior on alpha.
    lambda_1 : float, default=1e-6
        Shape parameter for Gamma prior on lambda.
    lambda_2 : float, default=1e-6
        Rate parameter for Gamma prior on lambda.
    compute_score : bool, default=False
        Whether to compute the log marginal likelihood.
    fit_intercept : bool, default=True
        Whether to fit the intercept.

    Attributes
    ----------
    weights : ndarray
        Mean of the posterior distribution over weights.
    bias : float
        Intercept.
    alpha_ : ndarray
        Estimated precision of weights (one per feature).
    lambda_ : float
        Estimated precision of noise.
    sigma_ : ndarray
        Covariance matrix of the posterior.
    scores_ : ndarray
        Log marginal likelihood at each iteration.
    """

    def __init__(
        self,
        n_iter=300,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        compute_score=False,
        fit_intercept=True,
    ):
        super().__init__()
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept

        # Runtime attributes
        self.alpha_ = None
        self.lambda_ = None
        self.sigma_ = None
        self.scores_ = None

    def train(self, X, y):
        """
        Train the ARD regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : ARDRegression
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

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

        # Initialize hyperparameters
        alpha = np.ones(n_features)  # Precision of weights (one per feature)
        lambda_prec = 1.0  # Precision of noise

        # Precompute X.T @ X
        XTX = X_centered.T @ X_centered
        XTy = X_centered.T @ y_centered

        # Iterative estimation
        if self.compute_score:
            self.scores_ = []

        for iteration in range(self.n_iter):
            # Previous alpha for convergence check
            alpha_old = alpha.copy()

            # Compute posterior covariance and mean
            sigma_inv = lambda_prec * XTX + np.diag(alpha)
            sigma = np.linalg.inv(sigma_inv)
            self.weights = lambda_prec * sigma @ XTy

            # Update alpha (precision of weights) - ARD specific
            # Each weight gets its own precision
            gamma = 1 - alpha * np.diag(sigma)
            alpha = (gamma + 2 * self.alpha_1) / (
                self.weights ** 2 + 2 * self.alpha_2
            )

            # Update lambda (precision of noise)
            lambda_prec = (n_samples - np.sum(gamma) + 2 * self.lambda_1) / (
                np.sum((y_centered - X_centered @ self.weights) ** 2) +
                2 * self.lambda_2
            )

            # Compute log marginal likelihood if requested
            if self.compute_score:
                score = self._log_marginal_likelihood(X_centered, y_centered, alpha, lambda_prec, sigma)
                self.scores_.append(score)

            # Check convergence
            if np.max(np.abs(alpha - alpha_old)) < self.tol:
                break

        self.sigma_ = sigma
        self.alpha_ = alpha
        self.lambda_ = lambda_prec

        # Compute intercept
        if self.fit_intercept:
            self.bias = y_mean - X_mean @ self.weights
        else:
            self.bias = 0.0

        self._is_trained = True
        return self

    def _log_marginal_likelihood(self, X, y, alpha, lambda_prec, sigma):
        """Compute log marginal likelihood."""
        n_samples, n_features = X.shape

        # Log determinant of sigma
        sign, logdet = np.linalg.slogdet(sigma)

        # Compute the log marginal likelihood
        # This is a simplified version
        residuals = y - X @ self.weights
        rss = np.sum(residuals ** 2)

        # Log likelihood
        log_likelihood = (
            -0.5 * n_samples * np.log(2 * np.pi)
            - 0.5 * lambda_prec * rss
            - 0.5 * logdet
        )

        # Log prior
        log_prior = (
            np.sum(np.log(alpha) - alpha * self.weights ** 2)
            + np.log(lambda_prec) - lambda_prec * rss / n_samples
        )

        return log_likelihood + log_prior

    def __repr__(self):
        return f"ARDRegression(n_iter={self.n_iter})"
