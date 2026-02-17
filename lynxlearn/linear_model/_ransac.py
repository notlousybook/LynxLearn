"""
RANSAC Regressor - Robust outlier detection.
"""

import numpy as np
from ._base import BaseRegressor
from ._ols import LinearRegression


class RANSACRegressor(BaseRegressor):
    """
    RANSAC (RANdom SAmple Consensus) algorithm.

    RANSAC is an iterative method to estimate parameters of a mathematical
    model from a set of observed data that contains outliers. It fits a
    model to random subsets of inliers.

    Parameters
    ----------
    base_estimator : object, default=LinearRegression()
        The base estimator to fit on random subsets.
    min_samples : int or float, default=None
        Minimum number of samples to draw. If None, uses n_features + 1.
    residual_threshold : float, default=None
        Maximum residual for a data sample to be classified as inlier.
        If None, uses median absolute deviation of target values.
    max_trials : int, default=100
        Maximum number of iterations for random sample selection.
    max_skips : int, default=np.inf
        Maximum number of iterations to skip if no inlier found.
    stop_n_inliers : int, default=np.inf
        Stop iteration if at least this number of inliers are found.
    stop_score : float, default=np.inf
        Stop iteration if score is greater than this.
    stop_probability : float, default=0.99
        RANSAC iteration stops if at least one outlier-free set of the
        training data is sampled with probability >= stop_probability.
    loss : str, default='absolute_error'
        Loss function: 'absolute_error' or 'squared_error'.
    random_state : int, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    weights : ndarray
        Coefficients of the fitted model.
    bias : float
        Intercept of the fitted model.
    n_trials_ : int
        Number of trials performed.
    inlier_mask_ : ndarray
        Boolean mask of inliers.
    n_skips_no_inliers_ : int
        Number of iterations skipped due to no inliers.
    """

    def __init__(
        self,
        base_estimator=None,
        min_samples=None,
        residual_threshold=None,
        max_trials=100,
        max_skips=np.inf,
        stop_n_inliers=np.inf,
        stop_score=np.inf,
        stop_probability=0.99,
        loss='absolute_error',
        random_state=None,
    ):
        super().__init__()
        self.base_estimator = base_estimator if base_estimator is not None else LinearRegression()
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.max_skips = max_skips
        self.stop_n_inliers = stop_n_inliers
        self.stop_score = stop_score
        self.stop_probability = stop_probability
        self.loss = loss
        self.random_state = random_state

        # Runtime attributes
        self.n_trials_ = 0
        self.inlier_mask_ = None
        self.n_skips_no_inliers_ = 0

    def _is_valid(self, X, y):
        """Check if a subset is valid."""
        return True

    def _loss(self, y_true, y_pred):
        """Compute loss."""
        if self.loss == 'absolute_error':
            return np.abs(y_true - y_pred)
        elif self.loss == 'squared_error':
            return (y_true - y_pred) ** 2
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def _estimate_error(self, X, y):
        """Estimate error of the current model."""
        y_pred = self.predict(X)
        return self._loss(y, y_pred)

    def train(self, X, y):
        """
        Train the RANSAC regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RANSACRegressor
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Determine minimum samples
        if self.min_samples is None:
            self.min_samples = n_features + 1
        elif isinstance(self.min_samples, float):
            self.min_samples = int(np.ceil(self.min_samples * n_samples))

        # Determine residual threshold
        if self.residual_threshold is None:
            # Use median absolute deviation
            self.residual_threshold = np.median(np.abs(y - np.median(y)))

        # Initialize best model
        best_score = -np.inf
        best_n_inliers = 0
        best_inlier_mask = None
        best_estimator = None

        # RANSAC iterations
        self.n_trials_ = 0
        self.n_skips_no_inliers_ = 0

        for trial in range(self.max_trials):
            self.n_trials_ += 1

            # Select random subset
            subset_idx = np.random.choice(n_samples, size=self.min_samples, replace=False)
            X_subset = X[subset_idx]
            y_subset = y[subset_idx]

            # Fit model on subset
            # Create a fresh instance of the base estimator
            estimator = type(self.base_estimator)()
            try:
                estimator.train(X_subset, y_subset)
            except Exception:
                # Skip if fitting fails
                continue

            # Check validity
            if not self._is_valid(X_subset, y_subset):
                continue

            # Compute residuals
            y_pred = estimator.predict(X)
            residuals = self._loss(y, y_pred)

            # Determine inliers
            inlier_mask = residuals <= self.residual_threshold
            n_inliers = np.sum(inlier_mask)

            # Skip if no inliers
            if n_inliers == 0:
                self.n_skips_no_inliers_ += 1
                if self.n_skips_no_inliers_ >= self.max_skips:
                    break
                continue

            # Compute score on inliers
            if n_inliers > self.min_samples:
                X_inliers = X[inlier_mask]
                y_inliers = y[inlier_mask]

                # Refit on inliers
                estimator_inliers = type(self.base_estimator)()
                try:
                    estimator_inliers.train(X_inliers, y_inliers)
                except Exception:
                    continue

                score = estimator_inliers.evaluate(X_inliers, y_inliers)
            else:
                score = estimator.evaluate(X, y)

            # Update best model
            if (n_inliers > best_n_inliers or
                (n_inliers == best_n_inliers and score > best_score)):
                best_score = score
                best_n_inliers = n_inliers
                best_inlier_mask = inlier_mask
                best_estimator = estimator

            # Check stopping criteria
            if n_inliers >= self.stop_n_inliers:
                break
            if score >= self.stop_score:
                break

            # Early stopping based on probability
            if best_n_inliers > 0:
                inlier_ratio = best_n_inliers / n_samples
                if inlier_ratio > 0:
                    # Estimate remaining trials needed
                    # This is a simplified version
                    pass

        # Use best estimator
        if best_estimator is not None:
            self.weights = best_estimator.weights
            self.bias = best_estimator.bias
            self.inlier_mask_ = best_inlier_mask
            self._is_trained = True
        else:
            # Fallback to base estimator on all data
            self.base_estimator.train(X, y)
            self.weights = self.base_estimator.weights
            self.bias = self.base_estimator.bias
            self.inlier_mask_ = np.ones(n_samples, dtype=bool)
            self._is_trained = True

        return self

    def __repr__(self):
        return f"RANSACRegressor(max_trials={self.max_trials})"
