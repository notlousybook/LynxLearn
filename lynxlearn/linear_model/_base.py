"""
Base class for linear regression models.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseRegressor(ABC):
    """Base class for all regression models in LynxLearn."""

    def __init__(self):
        self.weights = None
        self.bias = None
        self._is_trained = False

    @abstractmethod
    def train(self, X, y):
        """
        Train the model on data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (features).
        y : array-like of shape (n_samples,)
            Training data (target values).

        Examples
        --------
        >>> model = LinearRegression()
        >>> model.train(X_train, y_train)
        """
        pass

    # Alias for scikit-learn compatibility
    def fit(self, X, y):
        """Alias for train(). Same as model.train(X, y)."""
        return self.train(X, y)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted values.

        Examples
        --------
        >>> predictions = model.predict(X_test)
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained first! Call model.train(X, y)")

        X = np.asarray(X)
        return X @ self.weights + self.bias

    def evaluate(self, X, y):
        """
        Evaluate model performance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True values.

        Returns
        -------
        score : float
            R² score (1.0 is perfect, 0.0 is bad).

        Examples
        --------
        >>> score = model.evaluate(X_test, y_test)
        >>> print(f"Model accuracy: {score:.2%}")
        """
        from ..metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    # Alias for scikit-learn compatibility
    def score(self, X, y):
        """Alias for evaluate(). Same as model.evaluate(X, y)."""
        return self.evaluate(X, y)

    def get_params(self):
        """
        Get the learned parameters.

        Returns
        -------
        params : dict
            Dictionary with 'weights' and 'bias'.

        Examples
        --------
        >>> params = model.get_params()
        >>> print(f"Weight: {params['weights']}, Bias: {params['bias']}")
        """
        return {"weights": self.weights, "bias": self.bias}

    def summary(self):
        """
        Print a summary of the trained model.

        Examples
        --------
        >>> model.summary()
        Model: LinearRegression
        Weights: [2.5, -1.3]
        Bias: 5.2
        """
        model_name = self.__class__.__name__
        status = "Trained" if self._is_trained else "Not trained"
        print(f"Model: {model_name}")
        print(f"Status: {status}")
        if self._is_trained:
            print(f"Weights: {self.weights}")
            print(f"Bias: {self.bias:.6f}")
