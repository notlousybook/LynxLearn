"""
Isotonic Regression - Monotonic regression.
"""

import numpy as np
from ._base import BaseRegressor


class IsotonicRegression(BaseRegressor):
    """
    Isotonic Regression.

    Fits a non-decreasing function to 1D data. The solution is
    obtained by finding the monotonic function that minimizes
    the sum of squared errors.

    Parameters
    ----------
    y_min : float or None, default=None
        Lower bound on the predicted values.
    y_max : float or None, default=None
        Upper bound on the predicted values.
    increasing : bool, default=True
        Whether to fit an increasing (True) or decreasing (False) function.
    out_of_bounds : str, default='nan'
        How to handle out-of-bounds X values: 'nan', 'clip', or 'raise'.

    Attributes
    ----------
    X_thresholds_ : ndarray
        Unique sorted X values used for fitting.
    y_thresholds_ : ndarray
        Fitted values at the X thresholds.
    """

    def __init__(self, y_min=None, y_max=None, increasing=True, out_of_bounds='nan'):
        super().__init__()
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

        # Runtime attributes
        self.X_thresholds_ = None
        self.y_thresholds_ = None
        # Override weights/bias since isotonic doesn't use them
        self.weights = None
        self.bias = None

    def _isotonic_regression(self, X, y):
        """
        Compute isotonic regression using the pool adjacent violators algorithm (PAVA).

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Input values.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        X_unique : ndarray
            Unique sorted X values.
        y_fitted : ndarray
            Fitted values at unique X points.
        """
        # Sort by X
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]

        # Group by unique X values
        X_unique, unique_idx = np.unique(X_sorted, return_inverse=True)
        y_unique = np.zeros(len(X_unique))

        # Average y for each unique X
        for i in range(len(X_unique)):
            mask = unique_idx == i
            y_unique[i] = np.mean(y_sorted[mask])

        # Pool adjacent violators algorithm
        # Initialize blocks
        blocks = [{'start': i, 'end': i, 'mean': y_unique[i], 'size': 1}
                  for i in range(len(X_unique))]

        # Merge violating blocks
        i = 0
        while i < len(blocks) - 1:
            if (self.increasing and blocks[i]['mean'] > blocks[i + 1]['mean']) or \
               (not self.increasing and blocks[i]['mean'] < blocks[i + 1]['mean']):
                # Merge blocks
                total_size = blocks[i]['size'] + blocks[i + 1]['size']
                total_mean = (blocks[i]['mean'] * blocks[i]['size'] +
                             blocks[i + 1]['mean'] * blocks[i + 1]['size']) / total_size

                blocks[i] = {
                    'start': blocks[i]['start'],
                    'end': blocks[i + 1]['end'],
                    'mean': total_mean,
                    'size': total_size
                }
                blocks.pop(i + 1)

                # Check if merge caused violation with previous block
                if i > 0:
                    i -= 1
            else:
                i += 1

        # Extract fitted values
        y_fitted = np.zeros(len(X_unique))
        for block in blocks:
            y_fitted[block['start']:block['end'] + 1] = block['mean']

        # Apply bounds
        if self.y_min is not None:
            y_fitted = np.maximum(y_fitted, self.y_min)
        if self.y_max is not None:
            y_fitted = np.minimum(y_fitted, self.y_max)

        return X_unique, y_fitted

    def train(self, X, y):
        """
        Train the isotonic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training data (1D only).
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : IsotonicRegression
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Ensure 1D
        if X.ndim != 1:
            raise ValueError("IsotonicRegression only supports 1D input X")

        n_samples = len(X)

        # Fit isotonic regression
        self.X_thresholds_, self.y_thresholds_ = self._isotonic_regression(X, y)

        self._is_trained = True
        return self

    def predict(self, X):
        """
        Make predictions using the fitted isotonic function.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Data to predict on (1D only).

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained first! Call model.train(X, y)")

        X = np.asarray(X)

        # Ensure 1D
        if X.ndim != 1:
            raise ValueError("IsotonicRegression only supports 1D input X")

        # Handle out of bounds
        X_min = self.X_thresholds_[0]
        X_max = self.X_thresholds_[-1]

        if self.out_of_bounds == 'nan':
            # Return NaN for out of bounds
            predictions = np.full_like(X, np.nan, dtype=float)
            in_bounds = (X >= X_min) & (X <= X_max)
            X_in = X[in_bounds]
        elif self.out_of_bounds == 'clip':
            # Clip to bounds
            X_in = np.clip(X, X_min, X_max)
            in_bounds = np.ones(len(X), dtype=bool)
        elif self.out_of_bounds == 'raise':
            # Raise error for out of bounds
            if np.any(X < X_min) or np.any(X > X_max):
                raise ValueError("X contains values outside the training range")
            X_in = X
            in_bounds = np.ones(len(X), dtype=bool)
        else:
            raise ValueError(f"Unknown out_of_bounds: {self.out_of_bounds}")

        # Interpolate
        predictions_in = np.interp(X_in, self.X_thresholds_, self.y_thresholds_)

        # Fill predictions
        predictions = np.zeros_like(X, dtype=float)
        predictions[in_bounds] = predictions_in

        return predictions

    def evaluate(self, X, y):
        """
        Evaluate model performance.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Test data.
        y : array-like of shape (n_samples,)
            True values.

        Returns
        -------
        score : float
            R² score (1.0 is perfect, 0.0 is bad).
        """
        from ..metrics import r2_score

        y_pred = self.predict(X)
        # Only evaluate on valid predictions
        valid_mask = ~np.isnan(y_pred)
        if np.sum(valid_mask) == 0:
            return 0.0
        return r2_score(y[valid_mask], y_pred[valid_mask])

    def __repr__(self):
        return f"IsotonicRegression(increasing={self.increasing})"
