"""
Multi-task ElasticNet and Lasso regression.
"""

import numpy as np

from ._base import BaseRegressor


class MultiTaskElasticNet(BaseRegressor):
    """
    Multi-task ElasticNet with L1 and L2 regularization.

    Solves multiple regression problems jointly, encouraging
    similar sparsity patterns across tasks.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    l1_ratio : float, default=0.5
        Mixing parameter (0 = Ridge, 1 = Lasso).
    max_iter : int, default=1000
        Maximum iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    learn_bias : bool, default=True
        Whether to learn the bias term.
        (Also accepts `fit_intercept` for backward compatibility)

    Attributes
    ----------
    weights : ndarray of shape (n_features, n_tasks)
        Coefficients for each task.
    bias : ndarray of shape (n_tasks,)
        Intercepts for each task.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
        max_iter=1000,
        tol=1e-4,
        learn_bias=True,
        fit_intercept=None,
    ):
        super().__init__()
        # Backward compatibility: fit_intercept overrides learn_bias if provided
        if fit_intercept is not None:
            learn_bias = fit_intercept
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.learn_bias = learn_bias
        self.fit_intercept = learn_bias  # Alias for backward compatibility
        self.n_iter_ = 0

    def _soft_threshold(self, x, gamma):
        """Soft thresholding operator."""
        return np.sign(x) * np.maximum(np.abs(x) - gamma, 0.0)

    def train(self, X, y):
        """
        Train the multi-task ElasticNet model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_tasks)
            Target values for multiple tasks.

        Returns
        -------
        self : MultiTaskElasticNet
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_tasks = y.shape[1]

        # Center data if fitting intercept
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y, axis=0)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y
            X_mean = np.zeros(n_features)
            y_mean = np.zeros(n_tasks)

        # Initialize weights
        self.weights = np.zeros((n_features, n_tasks))
        self.bias = np.zeros(n_tasks)

        # Precompute X.T @ X
        XTX = X_centered.T @ X_centered
        XTy = X_centered.T @ y_centered

        # Coordinate descent
        alpha_l1 = self.alpha * self.l1_ratio
        alpha_l2 = self.alpha * (1 - self.l1_ratio)

        for iteration in range(self.max_iter):
            weights_old = self.weights.copy()

            for j in range(n_features):
                # Compute residual without feature j
                residual = (
                    y_centered
                    - X_centered @ self.weights
                    + X_centered[:, j : j + 1] * self.weights[j : j + 1, :]
                )

                # Compute update
                rho_j = (
                    XTy[j, :]
                    - np.sum(XTX[j, :][:, np.newaxis] * self.weights, axis=0)
                    + XTX[j, j] * self.weights[j, :]
                )

                # Soft threshold for L1
                norm_rho_j = np.linalg.norm(rho_j)
                if norm_rho_j > alpha_l1:
                    shrinkage = (norm_rho_j - alpha_l1) / (XTX[j, j] + alpha_l2)
                    self.weights[j, :] = shrinkage * rho_j / norm_rho_j
                else:
                    self.weights[j, :] = 0.0

            # Check convergence
            if np.max(np.abs(self.weights - weights_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break

        self.n_iter_ = iteration + 1

        # Compute intercept
        if self.fit_intercept:
            self.bias = y_mean - X_mean @ self.weights
        else:
            self.bias = np.zeros(n_tasks)

        self._is_trained = True
        return self

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.

        Returns
        -------
        predictions : ndarray of shape (n_samples, n_tasks)
            Predicted values for each task.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained first! Call model.train(X, y)")

        X = np.asarray(X)
        return X @ self.weights + self.bias

    def __repr__(self):
        return f"MultiTaskElasticNet(alpha={self.alpha}, l1_ratio={self.l1_ratio})"


class MultiTaskLasso(BaseRegressor):
    """
    Multi-task Lasso with L1 regularization.

    Solves multiple regression problems jointly, encouraging
    similar sparsity patterns across tasks.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    max_iter : int, default=1000
        Maximum iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    learn_bias : bool, default=True
        Whether to learn the bias term.
        (Also accepts `fit_intercept` for backward compatibility)

    Attributes
    ----------
    weights : ndarray of shape (n_features, n_tasks)
        Coefficients for each task.
    bias : ndarray of shape (n_tasks,)
        Intercepts for each task.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(
        self,
        alpha=1.0,
        max_iter=1000,
        tol=1e-4,
        learn_bias=True,
        fit_intercept=None,
    ):
        super().__init__()
        # Backward compatibility: fit_intercept overrides learn_bias if provided
        if fit_intercept is not None:
            learn_bias = fit_intercept
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learn_bias = learn_bias
        self.fit_intercept = learn_bias  # Alias for backward compatibility
        self.n_iter_ = 0

    def _soft_threshold(self, x, gamma):
        """Soft thresholding operator."""
        return np.sign(x) * np.maximum(np.abs(x) - gamma, 0.0)

    def train(self, X, y):
        """
        Train the multi-task Lasso model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_tasks)
            Target values for multiple tasks.

        Returns
        -------
        self : MultiTaskLasso
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_tasks = y.shape[1]

        # Center data if fitting intercept
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y, axis=0)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y
            X_mean = np.zeros(n_features)
            y_mean = np.zeros(n_tasks)

        # Initialize weights
        self.weights = np.zeros((n_features, n_tasks))
        self.bias = np.zeros(n_tasks)

        # Precompute X.T @ X
        XTX = X_centered.T @ X_centered
        XTy = X_centered.T @ y_centered

        # Coordinate descent
        for iteration in range(self.max_iter):
            weights_old = self.weights.copy()

            for j in range(n_features):
                # Compute residual without feature j
                residual = (
                    y_centered
                    - X_centered @ self.weights
                    + X_centered[:, j : j + 1] * self.weights[j : j + 1, :]
                )

                # Compute update
                rho_j = (
                    XTy[j, :]
                    - np.sum(XTX[j, :][:, np.newaxis] * self.weights, axis=0)
                    + XTX[j, j] * self.weights[j, :]
                )

                # Soft threshold for L1 (group lasso style)
                norm_rho_j = np.linalg.norm(rho_j)
                if norm_rho_j > self.alpha:
                    shrinkage = (norm_rho_j - self.alpha) / XTX[j, j]
                    self.weights[j, :] = shrinkage * rho_j / norm_rho_j
                else:
                    self.weights[j, :] = 0.0

            # Check convergence
            if np.max(np.abs(self.weights - weights_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break

        self.n_iter_ = iteration + 1

        # Compute intercept
        if self.fit_intercept:
            self.bias = y_mean - X_mean @ self.weights
        else:
            self.bias = np.zeros(n_tasks)

        self._is_trained = True
        return self

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.

        Returns
        -------
        predictions : ndarray of shape (n_samples, n_tasks)
            Predicted values for each task.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained first! Call model.train(X, y)")

        X = np.asarray(X)
        return X @ self.weights + self.bias

    def __repr__(self):
        return f"MultiTaskLasso(alpha={self.alpha})"
