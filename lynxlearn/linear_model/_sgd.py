"""
Stochastic Gradient Descent Regressor.
"""

import numpy as np
from ._base import BaseRegressor


class SGDRegressor(BaseRegressor):
    """
    Linear regression using Stochastic Gradient Descent.

    Updates weights using one sample at a time, making it efficient
    for large datasets. Supports various loss functions and regularization.

    Parameters
    ----------
    learning_rate : str or float, default='invscaling'
        Learning rate schedule. 'constant', 'invscaling', or 'adaptive'.
        If float, uses constant learning rate.
    eta0 : float, default=0.01
        Initial learning rate.
    power_t : float, default=0.25
        Exponent for inverse scaling learning rate.
    max_iter : int, default=1000
        Maximum number of passes over the training data.
    tol : float, default=1e-3
        Stopping criterion tolerance.
    alpha : float, default=0.0001
        L2 regularization strength.
    penalty : str, default='l2'
        Penalty to use: 'l2', 'l1', or 'elasticnet'.
    l1_ratio : float, default=0.15
        Elastic Net mixing parameter (only used when penalty='elasticnet').
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    shuffle : bool, default=True
        Whether to shuffle training data before each epoch.
    random_state : int, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    weights : ndarray
        Coefficients.
    bias : float
        Intercept.
    n_iter_ : int
        Actual number of iterations.
    """

    def __init__(
        self,
        learning_rate='invscaling',
        eta0=0.01,
        power_t=0.25,
        max_iter=1000,
        tol=1e-3,
        alpha=0.0001,
        penalty='l2',
        l1_ratio=0.15,
        fit_intercept=True,
        shuffle=True,
        random_state=None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_iter_ = 0

    def _get_learning_rate(self, t):
        """Compute learning rate at iteration t."""
        if isinstance(self.learning_rate, (int, float)):
            return float(self.learning_rate)
        elif self.learning_rate == 'constant':
            return self.eta0
        elif self.learning_rate == 'invscaling':
            return self.eta0 / (t ** self.power_t)
        elif self.learning_rate == 'adaptive':
            # Simplified adaptive learning rate
            return self.eta0
        else:
            raise ValueError(f"Unknown learning_rate: {self.learning_rate}")

    def _soft_threshold(self, x, gamma):
        """Soft thresholding operator for L1 regularization."""
        if x > gamma:
            return x - gamma
        elif x < -gamma:
            return x + gamma
        else:
            return 0.0

    def train(self, X, y):
        """
        Train the SGD regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SGDRegressor
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0 if self.fit_intercept else 0.0

        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Precompute regularization parameters
        alpha_l1 = self.alpha * self.l1_ratio
        alpha_l2 = self.alpha * (1 - self.l1_ratio)

        # Training loop
        prev_loss = float('inf')
        no_improvement_count = 0

        for epoch in range(self.max_iter):
            # Shuffle data if requested
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y

            epoch_loss = 0.0

            # Iterate over samples
            for i in range(n_samples):
                t = epoch * n_samples + i + 1
                lr = self._get_learning_rate(t)

                x_i = X_shuffled[i]
                y_i = y_shuffled[i]

                # Prediction
                y_pred = np.dot(x_i, self.weights) + self.bias
                error = y_pred - y_i

                # Compute gradients
                grad_w = error * x_i
                grad_b = error

                # Add L2 regularization
                if self.penalty in ['l2', 'elasticnet']:
                    grad_w += alpha_l2 * self.weights

                # Update weights
                if self.penalty == 'l1':
                    # Proximal gradient for L1
                    self.weights = self._soft_threshold(
                        self.weights - lr * grad_w, lr * alpha_l1
                    )
                elif self.penalty == 'elasticnet':
                    # Elastic Net
                    self.weights = self._soft_threshold(
                        self.weights - lr * grad_w, lr * alpha_l1
                    )
                else:  # l2 or none
                    self.weights -= lr * grad_w

                # Update bias (no regularization)
                if self.fit_intercept:
                    self.bias -= lr * grad_b

                # Accumulate loss
                epoch_loss += 0.5 * error ** 2

            # Add regularization to loss
            if self.penalty == 'l2':
                epoch_loss += 0.5 * alpha_l2 * np.sum(self.weights ** 2)
            elif self.penalty == 'l1':
                epoch_loss += alpha_l1 * np.sum(np.abs(self.weights))
            elif self.penalty == 'elasticnet':
                epoch_loss += (0.5 * alpha_l2 * np.sum(self.weights ** 2) +
                              alpha_l1 * np.sum(np.abs(self.weights)))

            # Check convergence
            if np.abs(prev_loss - epoch_loss) < self.tol:
                no_improvement_count += 1
                if no_improvement_count >= 5:
                    self.n_iter_ = epoch + 1
                    break
            else:
                no_improvement_count = 0

            prev_loss = epoch_loss

        self.n_iter_ = epoch + 1
        self._is_trained = True
        return self

    def __repr__(self):
        return (f"SGDRegressor(learning_rate={self.learning_rate}, "
                f"alpha={self.alpha}, penalty={self.penalty})")
