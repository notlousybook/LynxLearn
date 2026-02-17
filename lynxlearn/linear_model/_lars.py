"""
Least Angle Regression (LARS) and LassoLARS.
"""

import numpy as np
from ._base import BaseRegressor


class Lars(BaseRegressor):
    """
    Least Angle Regression (LARS).

    A model selection algorithm similar to forward stepwise regression.
    Less greedy than forward selection, adding variables along their
    equiangular directions.

    Parameters
    ----------
    n_nonzero_coefs : int or None, default=None
        Target number of non-zero coefficients. If None, uses min(n_samples, n_features).
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    verbose : bool, default=False
        Whether to print progress.

    Attributes
    ----------
    weights : ndarray
        Coefficients.
    bias : float
        Intercept.
    alphas_ : ndarray
        Path of alpha values (regularization parameter).
    active_ : list
        Indices of active variables at each step.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(self, n_nonzero_coefs=None, fit_intercept=True, verbose=False):
        super().__init__()
        self.n_nonzero_coefs = n_nonzero_coefs
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.alphas_ = None
        self.active_ = None
        self.n_iter_ = 0

    def train(self, X, y):
        """
        Train the LARS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : Lars
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

        # Normalize columns
        X_norm = X_centered / (np.linalg.norm(X_centered, axis=0) + 1e-10)

        # Determine maximum number of non-zero coefficients
        max_iter = min(n_samples, n_features)
        if self.n_nonzero_coefs is not None:
            max_iter = min(self.n_nonzero_coefs, max_iter)

        # Initialize
        self.weights = np.zeros(n_features)
        self.alphas_ = []
        self.active_ = []
        residual = y_centered.copy()

        # LARS algorithm
        for k in range(max_iter):
            # Find most correlated feature
            correlations = X_norm.T @ residual
            abs_corr = np.abs(correlations)
            max_corr = np.max(abs_corr)

            # Stop if correlation is negligible
            if max_corr < 1e-10:
                break

            # Find active set (features with max correlation)
            active = np.where(np.abs(abs_corr - max_corr) < 1e-10)[0]
            self.active_.append(active.copy())

            # Get active features
            X_active = X_norm[:, active]

            # Compute equiangular direction
            # Solve for direction u in span of active features
            G_active = X_active.T @ X_active
            ones = np.ones(len(active))

            # Solve G_active * a = ones
            try:
                a = np.linalg.solve(G_active, ones)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                a = np.linalg.pinv(G_active) @ ones

            # Compute direction
            u = X_active @ a
            u_norm = np.linalg.norm(u)
            u = u / u_norm

            # Compute step size
            # Find how far we can go before a new feature joins
            c = X_norm.T @ u
            gamma = np.inf

            for j in range(n_features):
                if j not in active:
                    # Distance to join
                    c_j = correlations[j]
                    c_max = max_corr
                    if (c_max - c_j) / (c_max - c[j]) > 0:
                        gamma_j = (c_max - c_j) / (c_max - c[j])
                        if 0 < gamma_j < gamma:
                            gamma = gamma_j

            # Also check if we reach the solution
            # Find distance to zero out a coefficient
            # This is more complex, simplified here
            if gamma == np.inf:
                gamma = 1.0

            # Update residual and coefficients
            residual -= gamma * max_corr * u

            # Update weights for active features
            for idx, j in enumerate(active):
                self.weights[j] += gamma * max_corr * a[idx]

            # Store alpha (correlation magnitude)
            self.alphas_.append(max_corr)

            if self.verbose:
                print(f"Iteration {k+1}: Active features = {active}, alpha = {max_corr:.4f}")

        self.n_iter_ = k + 1
        self.alphas_ = np.array(self.alphas_)

        # Compute intercept
        if self.fit_intercept:
            self.bias = y_mean - X_mean @ self.weights
        else:
            self.bias = 0.0

        self._is_trained = True
        return self

    def __repr__(self):
        return f"Lars(n_nonzero_coefs={self.n_nonzero_coefs})"


class LassoLars(BaseRegressor):
    """
    Lasso model fit with Least Angle Regression (LARS).

    Uses the LARS algorithm to compute the Lasso solution path.
    More efficient than coordinate descent for high-dimensional problems.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 penalty.
    max_iter : int, default=500
        Maximum number of iterations.
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    verbose : bool, default=False
        Whether to print progress.

    Attributes
    ----------
    weights : ndarray
        Coefficients.
    bias : float
        Intercept.
    alphas_ : ndarray
        Path of alpha values.
    active_ : list
        Indices of active variables at each step.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(self, alpha=1.0, max_iter=500, fit_intercept=True, verbose=False):
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.alphas_ = None
        self.active_ = None
        self.n_iter_ = 0

    def train(self, X, y):
        """
        Train the LassoLARS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LassoLars
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

        # Normalize columns
        X_norm = X_centered / (np.linalg.norm(X_centered, axis=0) + 1e-10)

        # Initialize
        self.weights = np.zeros(n_features)
        self.alphas_ = []
        self.active_ = []
        residual = y_centered.copy()

        # LARS-Lasso algorithm
        for k in range(self.max_iter):
            # Find most correlated feature
            correlations = X_norm.T @ residual
            abs_corr = np.abs(correlations)
            max_corr = np.max(abs_corr)

            # Stop if correlation is negligible or below alpha
            if max_corr < 1e-10 or max_corr < self.alpha:
                break

            # Find active set
            active = np.where(np.abs(abs_corr - max_corr) < 1e-10)[0]
            self.active_.append(active.copy())

            # Get active features
            X_active = X_norm[:, active]

            # Compute equiangular direction
            G_active = X_active.T @ X_active
            ones = np.ones(len(active))

            try:
                a = np.linalg.solve(G_active, ones)
            except np.linalg.LinAlgError:
                a = np.linalg.pinv(G_active) @ ones

            u = X_active @ a
            u_norm = np.linalg.norm(u)
            u = u / u_norm

            # Compute step size
            c = X_norm.T @ u
            gamma = np.inf

            # Distance to join
            for j in range(n_features):
                if j not in active:
                    c_j = correlations[j]
                    c_max = max_corr
                    denominator = c_max - c[j]
                    if abs(denominator) > 1e-10:
                        gamma_j = (c_max - c_j) / denominator
                        if 0 < gamma_j < gamma:
                            gamma = gamma_j

            # Distance to drop (Lasso modification)
            # Check if any coefficient would cross zero
            for idx, j in enumerate(active):
                if a[idx] != 0:
                    gamma_j = -self.weights[j] / (max_corr * a[idx])
                    if 0 < gamma_j < gamma:
                        gamma = gamma_j

            # Clamp gamma
            if gamma == np.inf:
                gamma = 1.0

            # Update residual
            residual -= gamma * max_corr * u

            # Update weights
            for idx, j in enumerate(active):
                self.weights[j] += gamma * max_corr * a[idx]

            # Apply soft thresholding (Lasso)
            # This is a simplified version
            if self.alpha > 0:
                for j in active:
                    if abs(self.weights[j]) < 1e-10:
                        self.weights[j] = 0.0

            self.alphas_.append(max_corr)

            if self.verbose:
                print(f"Iteration {k+1}: Active features = {active}, alpha = {max_corr:.4f}")

        self.n_iter_ = k + 1
        self.alphas_ = np.array(self.alphas_)

        # Compute intercept
        if self.fit_intercept:
            self.bias = y_mean - X_mean @ self.weights
        else:
            self.bias = 0.0

        self._is_trained = True
        return self

    def __repr__(self):
        return f"LassoLars(alpha={self.alpha})"
