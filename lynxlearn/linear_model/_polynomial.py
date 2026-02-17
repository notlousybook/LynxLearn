"""
Polynomial Regression using basis expansion.
"""

import numpy as np
from itertools import combinations_with_replacement
from ._base import BaseRegressor
from ._ols import LinearRegression


class PolynomialFeatures:
    """
    Generate polynomial and interaction features.

    Parameters
    ----------
    degree : int, default=2
        Degree of polynomial features.
    include_bias : bool, default=True
        If True, include a bias column (all ones).

    Example
    -------
    >>> X = np.array([[2, 3], [3, 4]])
    >>> poly = PolynomialFeatures(degree=2)
    >>> poly.fit_transform(X)
    array([[ 1.,  2.,  3.,  4.,  6.,  9.],
           [ 1.,  3.,  4.,  9., 12., 16.]])
    """

    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        self.n_input_features_ = None
        self.n_output_features_ = None

    def fit(self, X):
        """Compute number of output features."""
        X = np.asarray(X)
        self.n_input_features_ = X.shape[1]
        # Calculate number of combinations: C(n+d, d) where n=n_features, d=degree
        self.n_output_features_ = int(
            np.prod([self.n_input_features_ + i for i in range(1, self.degree + 1)]) /
            np.prod(range(1, self.degree + 1))
        )
        if self.include_bias:
            self.n_output_features_ += 1
        return self

    def transform(self, X):
        """Transform data to polynomial features."""
        X = np.asarray(X)
        n_samples, n_features = X.shape

        if self.n_input_features_ is None:
            self.fit(X)

        # Start with bias term if requested
        if self.include_bias:
            features = [np.ones(n_samples)]
        else:
            features = []

        # Add original features (degree 1)
        for i in range(n_features):
            features.append(X[:, i])

        # Add higher degree features
        for degree in range(2, self.degree + 1):
            for indices in combinations_with_replacement(range(n_features), degree):
                feature = np.prod([X[:, i] for i in indices], axis=0)
                features.append(feature)

        return np.column_stack(features)

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class PolynomialRegression(BaseRegressor):
    """
    Polynomial Regression using basis expansion + Linear Regression.

    Parameters
    ----------
    degree : int, default=2
        Degree of polynomial features.
    fit_intercept : bool, default=True
        Whether to fit intercept.

    Attributes
    ----------
    poly_transformer : PolynomialFeatures
        The polynomial feature transformer.
    linear_model : LinearRegression
        The underlying linear regression model.
    """

    def __init__(self, degree=2, fit_intercept=True):
        super().__init__()
        self.degree = degree
        self.fit_intercept = fit_intercept
        self.poly_transformer = PolynomialFeatures(degree=degree, include_bias=not fit_intercept)
        self.linear_model = LinearRegression(fit_intercept=fit_intercept)

    def train(self, X, y):
        """
        Train the polynomial regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : PolynomialRegression
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Transform to polynomial features
        X_poly = self.poly_transformer.fit_transform(X)

        # Fit linear regression on transformed features
        self.linear_model.fit(X_poly, y)

        # Store parameters
        self.weights = self.linear_model.weights
        self.bias = self.linear_model.bias
        self._is_trained = True

        return self

    def predict(self, X):
        """
        Predict using polynomial regression.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained first! Call model.train(X, y)")

        X = np.asarray(X)
        X_poly = self.poly_transformer.transform(X)
        return self.linear_model.predict(X_poly)

    def __repr__(self):
        return f"PolynomialRegression(degree={self.degree}, fit_intercept={self.fit_intercept})"
