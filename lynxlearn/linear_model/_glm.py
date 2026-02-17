"""
Generalized Linear Models (GLM) - Poisson, Gamma, Tweedie.
"""

import numpy as np

from ._base import BaseRegressor


class GeneralizedLinearModel(BaseRegressor):
    """
    Generalized Linear Model.

    A flexible framework for regression that extends linear regression
    to allow for non-normal error distributions through link functions.

    Parameters
    ----------
    family : str, default='gaussian'
        The distribution family: 'gaussian', 'poisson', 'gamma', 'tweedie'.
    link : str or None, default=None
        The link function. If None, uses the canonical link for the family.
    alpha : float, default=0.0
        L2 regularization strength.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    learn_bias : bool, default=True
        Whether to learn the bias/intercept term.
        (Also accepts `fit_intercept` for backward compatibility)
    power : float, default=1.5
        Power parameter for Tweedie distribution (1 < power < 2).

    Attributes
    ----------
    weights : ndarray
        Coefficients.
    bias : float
        Intercept.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(
        self,
        family="gaussian",
        link=None,
        alpha=0.0,
        max_iter=100,
        tol=1e-4,
        learn_bias=True,
        fit_intercept=None,
        power=1.5,
    ):
        super().__init__()
        # Backward compatibility: fit_intercept overrides learn_bias if provided
        if fit_intercept is not None:
            learn_bias = fit_intercept
        self.family = family
        self.link = link
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learn_bias = learn_bias
        self.fit_intercept = learn_bias  # Alias for backward compatibility
        self.power = power
        self.n_iter_ = 0

    def _get_link_function(self):
        """Get link and inverse link functions."""
        # Default to canonical links
        if self.link is None:
            if self.family == "gaussian":
                link = "identity"
            elif self.family == "poisson":
                link = "log"
            elif self.family == "gamma":
                link = "inverse"
            elif self.family == "tweedie":
                link = "log"
            else:
                link = "identity"
        else:
            link = self.link

        # Define link functions
        if link == "identity":

            def link_func(eta):
                return eta

            def inv_link_func(eta):
                return eta
        elif link == "log":

            def link_func(mu):
                return np.log(mu + 1e-10)

            def inv_link_func(eta):
                return np.exp(eta)
        elif link == "inverse":

            def link_func(mu):
                return 1.0 / (mu + 1e-10)

            def inv_link_func(eta):
                return 1.0 / (eta + 1e-10)
        elif link == "logit":

            def link_func(mu):
                return np.log(mu / (1 - mu + 1e-10) + 1e-10)

            def inv_link_func(eta):
                return 1.0 / (1 + np.exp(-eta))
        else:
            raise ValueError(f"Unknown link function: {link}")

        return link_func, inv_link_func

    def _get_variance_function(self, mu):
        """Get variance function for the distribution."""
        if self.family == "gaussian":
            return np.ones_like(mu)
        elif self.family == "poisson":
            return mu
        elif self.family == "gamma":
            return mu**2
        elif self.family == "tweedie":
            return mu**self.power
        else:
            return np.ones_like(mu)

    def _get_derivative_link(self, eta):
        """Get derivative of inverse link function."""
        if self.link is None:
            if self.family == "gaussian":
                link = "identity"
            elif self.family == "poisson":
                link = "log"
            elif self.family == "gamma":
                link = "inverse"
            elif self.family == "tweedie":
                link = "log"
            else:
                link = "identity"
        else:
            link = self.link

        if link == "identity":
            return np.ones_like(eta)
        elif link == "log":
            return np.exp(eta)
        elif link == "inverse":
            return -1.0 / (eta**2 + 1e-10)
        elif link == "logit":
            exp_eta = np.exp(-eta)
            return exp_eta / ((1 + exp_eta) ** 2 + 1e-10)
        else:
            return np.ones_like(eta)

    def train(self, X, y):
        """
        Train the GLM model using Iteratively Reweighted Least Squares (IRLS).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GeneralizedLinearModel
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Ensure y is positive for certain distributions
        if self.family in ["poisson", "gamma", "tweedie"]:
            y = np.maximum(y, 1e-10)

        # Get link functions
        link_func, inv_link_func = self._get_link_function()

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0 if self.fit_intercept else 0.0

        # Initialize mu (mean predictions)
        mu = y.copy()

        # IRLS algorithm
        for iteration in range(self.max_iter):
            # Compute eta (linear predictor)
            eta = link_func(mu)

            # Compute predictions
            y_pred = inv_link_func(X @ self.weights + self.bias)

            # Compute working response
            # z = eta + (y - mu) * d(eta)/d(mu)
            # For canonical links, this simplifies
            z = eta + (y - mu) / self._get_derivative_link(eta)

            # Compute weights
            # w = 1 / (V(mu) * (d(eta)/d(mu))^2)
            var = self._get_variance_function(mu)
            d_eta_d_mu = self._get_derivative_link(eta)
            w = 1.0 / (var * d_eta_d_mu**2 + 1e-10)

            # Weighted least squares update
            sqrt_w = np.sqrt(w)
            X_weighted = X * sqrt_w[:, np.newaxis]
            z_weighted = z * sqrt_w

            if self.fit_intercept:
                X_b = np.c_[np.ones(n_samples), X_weighted]
            else:
                X_b = X_weighted

            # Add regularization
            I = np.eye(X_b.shape[1])
            if self.fit_intercept:
                I[0, 0] = 0  # Don't regularize intercept

            theta = np.linalg.pinv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ z_weighted

            if self.fit_intercept:
                self.bias = theta[0]
                self.weights = theta[1:]
            else:
                self.bias = 0.0
                self.weights = theta

            # Update mu
            mu = inv_link_func(X @ self.weights + self.bias)

            # Check convergence
            if iteration > 0:
                delta = np.max(np.abs(mu - mu_old))
                if delta < self.tol:
                    self.n_iter_ = iteration + 1
                    break

            mu_old = mu.copy()

        self.n_iter_ = iteration + 1
        self._is_trained = True
        return self

    def __repr__(self):
        return f"GeneralizedLinearModel(family='{self.family}')"


class PoissonRegressor(GeneralizedLinearModel):
    """
    Poisson Regression for count data.

    Uses a log link function and Poisson distribution.
    Suitable for modeling count data with non-negative integer values.

    Parameters
    ----------
    alpha : float, default=0.0
        L2 regularization strength.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    """

    def __init__(
        self, alpha=0.0, max_iter=100, tol=1e-4, learn_bias=True, fit_intercept=None
    ):
        # Backward compatibility: fit_intercept overrides learn_bias if provided
        if fit_intercept is not None:
            learn_bias = fit_intercept
        super().__init__(
            family="poisson",
            link="log",
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            learn_bias=learn_bias,
            fit_intercept=fit_intercept,
        )

    def __repr__(self):
        return f"PoissonRegressor(alpha={self.alpha})"


class GammaRegressor(GeneralizedLinearModel):
    """
    Gamma Regression for positive continuous data.

    Uses an inverse link function and Gamma distribution.
    Suitable for modeling strictly positive continuous data.

    Parameters
    ----------
    alpha : float, default=0.0
        L2 regularization strength.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    """

    def __init__(
        self, alpha=0.0, max_iter=100, tol=1e-4, learn_bias=True, fit_intercept=None
    ):
        # Backward compatibility: fit_intercept overrides learn_bias if provided
        if fit_intercept is not None:
            learn_bias = fit_intercept
        super().__init__(
            family="gamma",
            link="inverse",
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            learn_bias=learn_bias,
            fit_intercept=fit_intercept,
        )

    def __repr__(self):
        return f"GammaRegressor(alpha={self.alpha})"


class TweedieRegressor(GeneralizedLinearModel):
    """
    Tweedie Regression for compound Poisson-Gamma distributions.

    Uses a log link function and Tweedie distribution.
    Suitable for modeling non-negative continuous data with mass at zero.

    Parameters
    ----------
    power : float, default=1.5
        Power parameter (1 < power < 2).
        power=1: Poisson, power=2: Gamma, power=3: Inverse Gaussian.
    alpha : float, default=0.0
        L2 regularization strength.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    """

    def __init__(
        self,
        power=1.5,
        alpha=0.0,
        max_iter=100,
        tol=1e-4,
        learn_bias=True,
        fit_intercept=None,
    ):
        # Backward compatibility: fit_intercept overrides learn_bias if provided
        if fit_intercept is not None:
            learn_bias = fit_intercept
        super().__init__(
            family="tweedie",
            link="log",
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            learn_bias=learn_bias,
            fit_intercept=fit_intercept,
            power=power,
        )

    def __repr__(self):
        return f"TweedieRegressor(power={self.power}, alpha={self.alpha})"
