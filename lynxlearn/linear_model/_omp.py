"""
Orthogonal Matching Pursuit (OMP).
"""

import numpy as np
from ._base import BaseRegressor


class OrthogonalMatchingPursuit(BaseRegressor):
    """
    Orthogonal Matching Pursuit (OMP).

    A greedy algorithm that approximates the solution of a linear system
    with sparse coefficients. At each step, it selects the feature most
    correlated with the current residual.

    Parameters
    ----------
    n_nonzero_coefs : int or None, default=None
        Desired number of non-zero coefficients. If None, uses min(n_samples, n_features).
    tol : float or None, default=None
        Maximum norm of the residual. If not None, overrides n_nonzero_coefs.
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
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(self, n_nonzero_coefs=None, tol=None, fit_intercept=True, verbose=False):
        super().__init__()
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.n_iter_ = 0

    def train(self, X, y):
        """
        Train the OMP model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : OrthogonalMatchingPursuit
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

        # Determine stopping criteria
        if self.tol is not None:
            # Use tolerance-based stopping
            max_iter = min(n_samples, n_features)
        else:
            # Use number of non-zero coefficients
            max_iter = min(n_samples, n_features)
            if self.n_nonzero_coefs is not None:
                max_iter = min(self.n_nonzero_coefs, max_iter)

        # Initialize
        self.weights = np.zeros(n_features)
        residual = y_centered.copy()
        active_indices = []

        # OMP algorithm
        for k in range(max_iter):
            # Find most correlated feature
            correlations = np.abs(X_norm.T @ residual)
            best_idx = np.argmax(correlations)

            # Add to active set
            if best_idx not in active_indices:
                active_indices.append(best_idx)

            # Get active submatrix
            X_active = X_norm[:, active_indices]

            # Solve least squares on active set
            try:
                w_active = np.linalg.lstsq(X_active, y_centered, rcond=None)[0]
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                w_active = np.linalg.pinv(X_active) @ y_centered

            # Update residual
            residual = y_centered - X_active @ w_active

            # Update weights
            self.weights[:] = 0.0
            for idx, j in enumerate(active_indices):
                self.weights[j] = w_active[idx]

            # Check tolerance
            if self.tol is not None:
                residual_norm = np.linalg.norm(residual)
                if residual_norm < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {k+1}, residual norm: {residual_norm:.6f}")
                    break

            if self.verbose:
                print(f"Iteration {k+1}: Active features = {active_indices}, "
                      f"residual norm = {np.linalg.norm(residual):.6f}")

        self.n_iter_ = k + 1

        # Compute intercept
        if self.fit_intercept:
            self.bias = y_mean - X_mean @ self.weights
        else:
            self.bias = 0.0

        self._is_trained = True
        return self

    def __repr__(self):
        return f"OrthogonalMatchingPursuit(n_nonzero_coefs={self.n_nonzero_coefs}, tol={self.tol})"
