"""
L-BFGS Optimizer - The Secret Sauce.

L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton
optimization method that approximates the Hessian matrix using limited memory.

This is the algorithm that makes scikit-learn's linear models fast!
It converges much faster than SGD for smooth convex problems.

Key advantages:
- Superlinear convergence (faster than SGD's linear convergence)
- No learning rate to tune (uses line search)
- Memory efficient (only stores last m gradients/positions)
- Works great for convex optimization (linear regression, logistic regression)

References:
- Nocedal & Wright, "Numerical Optimization", Chapter 7
- scikit-learn's implementation in sklearn.utils.optimize
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ._base import BaseOptimizer


class LBFGS(BaseOptimizer):
    """
    L-BFGS Optimizer - Quasi-Newton method with limited memory.

    L-BFGS approximates the inverse Hessian matrix using a history of
    gradient differences and position changes. This allows it to take
    near-optimal steps without computing or storing the full Hessian.

    Parameters
    ----------
    memory_size : int, default=10
        Number of previous gradients to store (m in L-BFGS literature).
        Larger values = better approximation but more memory.
    max_line_search_iters : int, default=20
        Maximum iterations for line search.
    c1 : float, default=1e-4
        Armijo condition parameter for line search.
    c2 : float, default=0.9
        Wolfe condition parameter for line search.
    max_iter : int, default=1000
        Maximum number of optimization iterations.
    tol : float, default=1e-6
        Convergence tolerance for gradient norm.
    verbose : bool, default=False
        Print convergence information.

    Attributes
    ----------
    iterations : int
        Number of iterations performed.
    converged : bool
        Whether optimization converged.
    gradient_norm : float
        Final gradient norm.

    Examples
    --------
    >>> from lynxlearn.neural_network.optimizers import LBFGS
    >>> optimizer = LBFGS(memory_size=10, tol=1e-6)

    Notes
    -----
    L-BFGS is ideal for:
    - Convex optimization (linear/logistic regression)
    - Small to medium datasets
    - Problems where Hessian is expensive to compute

    L-BFGS is NOT ideal for:
    - Non-convex problems (can get stuck in local minima)
    - Very large datasets (memory for gradient history)
    - Stochastic settings (needs full batch gradients)

    Performance tip: For linear regression on CPU, L-BFGS with Intel MKL
    can match or beat scikit-learn's performance!
    """

    def __init__(
        self,
        memory_size: int = 10,
        max_line_search_iters: int = 20,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_iter: int = 1000,
        tol: float = 1e-6,
        verbose: bool = False,
    ):
        super().__init__()

        if memory_size < 1:
            raise ValueError(f"memory_size must be >= 1, got {memory_size}")
        if not 0 < c1 < c2 < 1:
            raise ValueError(f"Must have 0 < c1 < c2 < 1, got c1={c1}, c2={c2}")

        self.memory_size = memory_size
        self.max_line_search_iters = max_line_search_iters
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # State
        self.iterations = 0
        self.converged = False
        self.gradient_norm = float("inf")

        # L-BFGS history (stored per-parameter-group)
        self._s_history: Dict[int, List[np.ndarray]] = {}  # Position differences
        self._y_history: Dict[int, List[np.ndarray]] = {}  # Gradient differences
        self._rho_history: Dict[int, List[float]] = {}  # 1 / (y^T s)

    def _get_flat_params(self, params: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten parameter dictionary into a single vector."""
        flat = []
        for key in sorted(params.keys()):
            flat.append(params[key].ravel())
        return np.concatenate(flat)

    def _set_flat_params(
        self, flat: np.ndarray, shapes: Dict[str, Tuple[int, ...]]
    ) -> Dict[str, np.ndarray]:
        """Unflatten vector back to parameter dictionary."""
        params = {}
        idx = 0
        for key in sorted(shapes.keys()):
            shape = shapes[key]
            size = np.prod(shape)
            params[key] = flat[idx : idx + size].reshape(shape)
            idx += size
        return params

    def _get_flat_grads(self, grads: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten gradient dictionary into a single vector."""
        flat = []
        for key in sorted(grads.keys()):
            flat.append(grads[key].ravel())
        return np.concatenate(flat)

    def _two_loop_recursion(
        self,
        grad: np.ndarray,
        s_history: List[np.ndarray],
        y_history: List[np.ndarray],
        rho_history: List[float],
    ) -> np.ndarray:
        """
        L-BFGS two-loop recursion to compute search direction.

        This is the core L-BFGS algorithm that approximates
        H^{-1} * grad without storing the full Hessian.

        Parameters
        ----------
        grad : ndarray
            Current gradient
        s_history : list of ndarray
            History of position differences (x_{k+1} - x_k)
        y_history : list of ndarray
            History of gradient differences (g_{k+1} - g_k)
        rho_history : list of float
            History of 1 / (y^T s)

        Returns
        -------
        direction : ndarray
            Search direction (-H^{-1} * grad)
        """
        q = grad.copy()
        m = len(s_history)

        if m == 0:
            # No history yet, use negative gradient (steepest descent)
            return -q

        alpha = np.zeros(m)

        # First loop (backward)
        for i in range(m - 1, -1, -1):
            alpha[i] = rho_history[i] * np.dot(s_history[i], q)
            q = q - alpha[i] * y_history[i]

        # Initial Hessian approximation: H_0 = gamma * I
        # gamma = (s_{m-1}^T y_{m-1}) / (y_{m-1}^T y_{m-1})
        gamma = np.dot(s_history[-1], y_history[-1]) / (
            np.dot(y_history[-1], y_history[-1]) + 1e-10
        )
        r = gamma * q

        # Second loop (forward)
        for i in range(m):
            beta = rho_history[i] * np.dot(y_history[i], r)
            r = r + (alpha[i] - beta) * s_history[i]

        return -r

    def _line_search(
        self,
        f: Callable,
        grad_f: Callable,
        x: np.ndarray,
        direction: np.ndarray,
        grad: np.ndarray,
        f_val: float,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Backtracking line search with Wolfe conditions.

        Finds step size alpha such that:
        - Armijo condition: f(x + alpha*d) <= f(x) + c1*alpha*grad^T*d
        - Curvature condition: |grad(x + alpha*d)^T*d| <= c2*|grad^T*d|

        Parameters
        ----------
        f : callable
            Function to minimize
        grad_f : callable
            Gradient function
        x : ndarray
            Current position
        direction : ndarray
            Search direction
        grad : ndarray
            Current gradient
        f_val : float
            Current function value

        Returns
        -------
        alpha : float
            Step size
        new_f : float
            New function value
        new_grad : ndarray
            New gradient
        """
        alpha = 1.0
        directional_deriv = np.dot(grad, direction)

        if directional_deriv >= 0:
            # Not a descent direction, use small step
            alpha = 1e-4
            new_x = x + alpha * direction
            new_f = f(new_x)
            new_grad = grad_f(new_x)
            return alpha, new_f, new_grad

        for _ in range(self.max_line_search_iters):
            new_x = x + alpha * direction
            new_f = f(new_x)

            # Armijo condition
            if new_f <= f_val + self.c1 * alpha * directional_deriv:
                new_grad = grad_f(new_x)
                new_directional_deriv = np.dot(new_grad, direction)

                # Strong Wolfe curvature condition
                if abs(new_directional_deriv) <= self.c2 * abs(directional_deriv):
                    return alpha, new_f, new_grad

            # Reduce step size
            alpha *= 0.5

        # Line search failed, use current alpha
        new_x = x + alpha * direction
        new_f = f(new_x)
        new_grad = grad_f(new_x)
        return alpha, new_f, new_grad

    def minimize(
        self,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Minimize a function using L-BFGS.

        This is the main optimization routine that can be used
        standalone for any optimization problem.

        Parameters
        ----------
        f : callable
            Function to minimize. Takes flat parameter vector, returns scalar.
        grad_f : callable
            Gradient function. Takes flat parameter vector, returns gradient.
        x0 : ndarray
            Initial parameter values.

        Returns
        -------
        x_opt : ndarray
            Optimal parameters
        f_opt : float
            Optimal function value
        info : dict
            Optimization info (iterations, converged, etc.)
        """
        x = x0.copy().astype(np.float64)

        # Initialize history
        s_history: List[np.ndarray] = []
        y_history: List[np.ndarray] = []
        rho_history: List[float] = []

        # Initial evaluation
        f_val = f(x)
        grad = grad_f(x)
        self.gradient_norm = np.linalg.norm(grad)

        for iteration in range(self.max_iter):
            self.iterations = iteration + 1

            # Check convergence
            if self.gradient_norm < self.tol:
                self.converged = True
                if self.verbose:
                    print(f"L-BFGS converged at iteration {iteration}")
                break

            # Compute search direction via two-loop recursion
            direction = self._two_loop_recursion(
                grad, s_history, y_history, rho_history
            )

            # Line search
            alpha, new_f, new_grad = self._line_search(
                f, grad_f, x, direction, grad, f_val
            )

            # Compute position and gradient differences
            s = alpha * direction
            y = new_grad - grad

            # Compute rho = 1 / (y^T s)
            yTs = np.dot(y, s)
            if yTs > 1e-10:
                rho = 1.0 / yTs

                # Update history
                s_history.append(s.copy())
                y_history.append(y.copy())
                rho_history.append(rho)

                # Limit history size
                if len(s_history) > self.memory_size:
                    s_history.pop(0)
                    y_history.pop(0)
                    rho_history.pop(0)

            # Update position
            x = x + s
            f_val = new_f
            grad = new_grad
            self.gradient_norm = np.linalg.norm(grad)

            if self.verbose and iteration % 100 == 0:
                print(
                    f"Iteration {iteration}: f = {f_val:.6e}, |grad| = {self.gradient_norm:.6e}"
                )

        info = {
            "iterations": self.iterations,
            "converged": self.converged,
            "gradient_norm": self.gradient_norm,
            "final_loss": f_val,
        }

        return x, f_val, info

    def update(self, layer: Any) -> None:
        """
        Update layer parameters using L-BFGS.

        Note: L-BFGS is designed for full-batch optimization.
        For mini-batch training, consider using SGD or Adam instead.

        Parameters
        ----------
        layer : BaseLayer
            Layer with parameters to update
        """
        if not hasattr(layer, "get_params") or not hasattr(layer, "get_gradients"):
            return

        params = layer.get_params()
        grads = layer.get_gradients()

        if not params or not grads:
            return

        layer_id = id(layer)

        # Initialize history for this layer if needed
        if layer_id not in self._s_history:
            self._s_history[layer_id] = []
            self._y_history[layer_id] = []
            self._rho_history[layer_id] = []

        # Flatten params and grads
        flat_grad = self._get_flat_grads(grads)

        # Compute search direction
        direction = self._two_loop_recursion(
            flat_grad,
            self._s_history[layer_id],
            self._y_history[layer_id],
            self._rho_history[layer_id],
        )

        # Simple step (for layer-by-layer L-BFGS)
        # Use a fixed step size since we don't have function access
        alpha = 1.0

        # Update parameters
        shapes = {k: v.shape for k, v in params.items()}
        flat_params = self._get_flat_params(params)
        new_flat_params = flat_params + alpha * direction
        new_params = self._set_flat_params(new_flat_params, shapes)

        layer.set_params(new_params)

        # Update history (approximate since we don't have old grads stored)
        # This is a simplified version for layer-by-layer updates
        s = alpha * direction
        y = -flat_grad  # Approximate: y ≈ -grad (assuming small change)

        yTs = np.dot(y, s)
        if yTs > 1e-10:
            self._s_history[layer_id].append(s)
            self._y_history[layer_id].append(y)
            self._rho_history[layer_id].append(1.0 / yTs)

            # Limit history
            if len(self._s_history[layer_id]) > self.memory_size:
                self._s_history[layer_id].pop(0)
                self._y_history[layer_id].pop(0)
                self._rho_history[layer_id].pop(0)

        self.iterations += 1

    def reset(self) -> None:
        """Reset optimizer state."""
        self.iterations = 0
        self.converged = False
        self.gradient_norm = float("inf")
        self._s_history.clear()
        self._y_history.clear()
        self._rho_history.clear()

    def get_state(self) -> Dict[str, Any]:
        """Get optimizer state for serialization."""
        return {
            "memory_size": self.memory_size,
            "max_line_search_iters": self.max_line_search_iters,
            "c1": self.c1,
            "c2": self.c2,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "iterations": self.iterations,
            "converged": self.converged,
            "gradient_norm": self.gradient_norm,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set optimizer state from serialization."""
        self.iterations = state.get("iterations", 0)
        self.converged = state.get("converged", False)
        self.gradient_norm = state.get("gradient_norm", float("inf"))

    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        return {
            "memory_size": self.memory_size,
            "max_line_search_iters": self.max_line_search_iters,
            "c1": self.c1,
            "c2": self.c2,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LBFGS":
        """Create optimizer from configuration."""
        return cls(**config)

    def __repr__(self) -> str:
        return f"LBFGS(memory_size={self.memory_size}, tol={self.tol})"


class LBFGSLinearRegression:
    """
    L-BFGS optimized Linear Regression - FASTEST linear regression in LynxLearn.

    This uses L-BFGS optimization to solve the linear regression problem,
    which can be faster than both Normal Equation and Gradient Descent
    for certain problem sizes.

    This is the algorithm that powers scikit-learn's fast linear regression!

    Parameters
    ----------
    tol : float, default=1e-6
        Convergence tolerance.
    max_iter : int, default=1000
        Maximum iterations.
    memory_size : int, default=10
        L-BFGS memory size.
    verbose : bool, default=False
        Print progress.

    Examples
    --------
    >>> from lynxlearn.neural_network.optimizers import LBFGSLinearRegression
    >>> model = LBFGSLinearRegression()
    >>> model.fit(X, y)
    >>> predictions = model.predict(X_test)

    Performance Notes
    -----------------
    With Intel MKL backend, this can match or beat scikit-learn's
    LinearRegression for medium-sized problems!

    - Small problems (< 1000 samples): Normal Equation is faster
    - Medium problems (1000-100000 samples): L-BFGS is fastest
    - Large problems (> 100000 samples): Use SGD or Normal Equation
    """

    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = 1000,
        memory_size: int = 10,
        verbose: bool = False,
    ):
        self.tol = tol
        self.max_iter = max_iter
        self.memory_size = memory_size
        self.verbose = verbose

        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.n_iter_: int = 0
        self.converged_: bool = False

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Add bias column to feature matrix."""
        return np.column_stack([X, np.ones(X.shape[0])])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LBFGSLinearRegression":
        """
        Fit linear regression using L-BFGS.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values

        Returns
        -------
        self : LBFGSLinearRegression
            Fitted model
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Add bias term
        X_bias = self._add_bias(X)

        def loss(params):
            pred = X_bias @ params
            return 0.5 * np.mean((y - pred) ** 2)

        def grad(params):
            pred = X_bias @ params
            error = pred - y
            return (X_bias.T @ error) / n_samples

        # Initialize weights
        params0 = np.zeros((n_features + 1, y.shape[1]))

        # Run L-BFGS for each output
        optimizer = LBFGS(
            memory_size=self.memory_size,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
        )

        params_opt = np.zeros((n_features + 1, y.shape[1]))
        for j in range(y.shape[1]):

            def loss_j(p):
                return loss(p.reshape(-1, 1)) if p.ndim == 1 else loss(p)

            def grad_j(p):
                g = grad(p.reshape(-1, 1)) if p.ndim == 1 else grad(p)
                return g[:, 0] if g.ndim > 1 else g

            p_opt, _, info = optimizer.minimize(loss_j, grad_j, params0[:, j])
            params_opt[:, j] = p_opt
            self.n_iter_ = max(self.n_iter_, info["iterations"])
            self.converged_ = info["converged"]

        # Extract weights and bias
        self.weights = params_opt[:-1]
        self.bias = params_opt[-1, 0] if y.shape[1] == 1 else params_opt[-1]

        return self

    def train(self, X: np.ndarray, y: np.ndarray) -> "LBFGSLinearRegression":
        """Alias for fit()."""
        return self.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data

        Returns
        -------
        predictions : ndarray
            Predicted values
        """
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Alias for score()."""
        return self.score(X, y)
