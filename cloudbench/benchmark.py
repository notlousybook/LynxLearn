"""
Cloud Benchmark: LynxLearn vs The World

This benchmark uses the PIP-INSTALLED lynxlearn package (not local code).
This is useful for testing the published package on different machines.

Usage:
    pip install lynxlearn
    python benchmark.py
    python benchmark.py --quick
"""

import argparse
import time
import warnings
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Try importing libraries - using PIP-INSTALLED packages
# =============================================================================

# LynxLearn - from pip, NOT local code
try:
    import lynxlearn
    from lynxlearn import metrics
    from lynxlearn.linear_model import (
        GradientDescentRegressor,
        LinearRegression,
    )
    from lynxlearn.neural_network import SGD, Dense, Sequential
    from lynxlearn.neural_network.optimizers import LBFGSLinearRegression

    LYNXLEARN_AVAILABLE = True
    LYNXLEARN_VERSION = getattr(lynxlearn, "__version__", "unknown")
except ImportError as e:
    LYNXLEARN_AVAILABLE = False
    LYNXLEARN_VERSION = "not installed"
    print(f"[!] LynxLearn not available: {e}")
    print("[!] Install with: pip install lynxlearn")

# Scikit-learn
try:
    from sklearn.linear_model import LinearRegression as SklearnLR
    from sklearn.linear_model import SGDRegressor
    from sklearn.neural_network import MLPRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[!] Scikit-learn not installed. Install with: pip install scikit-learn")

# PyTorch
try:
    import torch
    import torch.nn as nn

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("[!] PyTorch not installed. Install with: pip install torch")


# =============================================================================
# Timing utilities
# =============================================================================


def format_time(seconds: float) -> str:
    """Format time nicely."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}us"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.3f}s"
    else:
        return f"{seconds / 60:.2f}min"


def timeit(func: Callable, *args, n_runs: int = 3, **kwargs) -> Tuple[float, Any]:
    """Time a function call, returning median time and result."""
    times = []
    result = None

    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return float(np.median(times)), result


# =============================================================================
# Benchmark functions
# =============================================================================


def generate_data(
    n_samples: int,
    n_features: int,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic regression data with known ground truth."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    true_weights = rng.randn(n_features)
    true_bias = 5.0
    y = X @ true_weights + true_bias + rng.randn(n_samples) * noise
    return X, y, true_weights


def benchmark_lynxlearn_ols(X_train, y_train, X_test, y_test) -> Dict:
    """Benchmark LynxLearn OLS (Normal Equation)."""
    if not LYNXLEARN_AVAILABLE:
        return {"error": "LynxLearn not available"}

    model = LinearRegression()
    fit_time, _ = timeit(model.fit, X_train, y_train)
    predict_time, y_pred = timeit(model.predict, X_test)

    return {
        "name": "LynxLearn OLS",
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": metrics.mse(y_test, y_pred),
        "r2": metrics.r2_score(y_test, y_pred),
    }


def benchmark_lynxlearn_gd(X_train, y_train, X_test, y_test) -> Dict:
    """Benchmark LynxLearn Gradient Descent."""
    if not LYNXLEARN_AVAILABLE:
        return {"error": "LynxLearn not available"}

    model = GradientDescentRegressor(learning_rate=0.01, n_iterations=1000)
    fit_time, _ = timeit(model.fit, X_train, y_train)
    predict_time, y_pred = timeit(model.predict, X_test)

    return {
        "name": "LynxLearn GD",
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": metrics.mse(y_test, y_pred),
        "r2": metrics.r2_score(y_test, y_pred),
        "iterations": model.n_iter_,
    }


def benchmark_lynxlearn_lbfgs(X_train, y_train, X_test, y_test) -> Dict:
    """Benchmark LynxLearn L-BFGS."""
    if not LYNXLEARN_AVAILABLE:
        return {"error": "LynxLearn not available"}

    model = LBFGSLinearRegression(tol=1e-6, max_iter=1000)
    fit_time, _ = timeit(model.fit, X_train, y_train)
    predict_time, y_pred = timeit(model.predict, X_test)

    return {
        "name": "LynxLearn L-BFGS",
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": np.mean((y_test - y_pred) ** 2),
        "r2": model.score(X_test, y_test),
        "iterations": model.n_iter_,
        "converged": model.converged_,
    }


def benchmark_lynxlearn_nn(X_train, y_train, X_test, y_test) -> Dict:
    """Benchmark LynxLearn Neural Network."""
    if not LYNXLEARN_AVAILABLE:
        return {"error": "LynxLearn not available"}

    n_features = X_train.shape[1]

    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(n_features,)),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=SGD(learning_rate=0.01), loss="mse")

    fit_time, _ = timeit(model.train, X_train, y_train, epochs=50, verbose=0)
    predict_time, y_pred = timeit(model.predict, X_test)

    return {
        "name": "LynxLearn NN",
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": np.mean((y_test - y_pred.flatten()) ** 2),
        "r2": 1
        - np.sum((y_test - y_pred.flatten()) ** 2)
        / np.sum((y_test - y_test.mean()) ** 2),
    }


def benchmark_sklearn_ols(X_train, y_train, X_test, y_test) -> Dict:
    """Benchmark scikit-learn OLS."""
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not available"}

    model = SklearnLR()
    fit_time, _ = timeit(model.fit, X_train, y_train)
    predict_time, y_pred = timeit(model.predict, X_test)

    return {
        "name": "scikit-learn OLS",
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": np.mean((y_test - y_pred) ** 2),
        "r2": model.score(X_test, y_test),
    }


def benchmark_sklearn_sgd(X_train, y_train, X_test, y_test) -> Dict:
    """Benchmark scikit-learn SGD."""
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not available"}

    model = SGDRegressor(max_iter=1000, tol=1e-6, random_state=42)
    fit_time, _ = timeit(model.fit, X_train, y_train)
    predict_time, y_pred = timeit(model.predict, X_test)

    return {
        "name": "scikit-learn SGD",
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": np.mean((y_test - y_pred) ** 2),
        "r2": model.score(X_test, y_test),
    }


def benchmark_sklearn_mlp(X_train, y_train, X_test, y_test) -> Dict:
    """Benchmark scikit-learn MLP."""
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not available"}

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=50,
        random_state=42,
        early_stopping=True,
    )
    fit_time, _ = timeit(model.fit, X_train, y_train)
    predict_time, y_pred = timeit(model.predict, X_test)

    return {
        "name": "scikit-learn MLP",
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": np.mean((y_test - y_pred) ** 2),
        "r2": model.score(X_test, y_test),
    }


class PyTorchMLP(nn.Module):
    """Simple PyTorch MLP for benchmarking."""

    def __init__(self, n_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


def benchmark_pytorch_nn(X_train, y_train, X_test, y_test) -> Dict:
    """Benchmark PyTorch Neural Network."""
    if not PYTORCH_AVAILABLE:
        return {"error": "PyTorch not available"}

    n_features = X_train.shape[1]

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_t = torch.FloatTensor(X_test)

    model = PyTorchMLP(n_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def train():
        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
        return model

    fit_time, model = timeit(train)

    model.eval()
    with torch.no_grad():
        predict_time, y_pred_t = timeit(model, X_test_t)
        y_pred = y_pred_t.numpy().flatten()

    return {
        "name": "PyTorch NN",
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": np.mean((y_test - y_pred) ** 2),
        "r2": 1
        - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2),
    }


def benchmark_numpy_lstsq(X_train, y_train, X_test, y_test) -> Dict:
    """Benchmark pure NumPy least squares (reference)."""
    # Add bias column
    X_b = np.column_stack([X_train, np.ones(X_train.shape[0])])
    X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])

    def fit():
        return np.linalg.lstsq(X_b, y_train, rcond=None)[0]

    fit_time, weights = timeit(fit)

    def predict():
        return X_test_b @ weights

    predict_time, y_pred = timeit(predict)

    return {
        "name": "NumPy lstsq",
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": np.mean((y_test - y_pred) ** 2),
        "r2": 1
        - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2),
    }


# =============================================================================
# Main benchmark runner
# =============================================================================


def run_single_benchmark(
    name: str,
    n_samples: int,
    n_features: int,
    seed: int = 42,
) -> List[Dict]:
    """Run a single benchmark configuration."""
    print(f"\n{'=' * 70}")
    print(f"Benchmark: {name}")
    print(f"{'=' * 70}")
    print(f"Data: {n_samples:,} samples x {n_features} features")
    print("-" * 70)

    # Generate data
    X_train, y_train, _ = generate_data(int(n_samples * 0.8), n_features, seed=seed)
    X_test, y_test, _ = generate_data(int(n_samples * 0.2), n_features, seed=seed + 1)

    results = []

    # Run benchmarks
    benchmarks = [
        ("NumPy lstsq", benchmark_numpy_lstsq),
        ("LynxLearn OLS", benchmark_lynxlearn_ols),
        ("LynxLearn GD", benchmark_lynxlearn_gd),
        ("LynxLearn L-BFGS", benchmark_lynxlearn_lbfgs),
        ("scikit-learn OLS", benchmark_sklearn_ols),
        ("scikit-learn SGD", benchmark_sklearn_sgd),
    ]

    for bench_name, bench_func in benchmarks:
        print(f"\n  Testing {bench_name}...")
        try:
            result = bench_func(X_train, y_train, X_test, y_test)
            if "error" not in result:
                results.append(result)
                print(f"    Fit time: {format_time(result['fit_time'])}")
                print(f"    Predict time: {format_time(result['predict_time'])}")
                print(f"    MSE: {result['mse']:.6f}")
                print(f"    R2: {result['r2']:.6f}")
            else:
                print(f"    {result['error']}")
        except Exception as e:
            print(f"    Error: {e}")

    # Sort by fit time
    results.sort(key=lambda x: x.get("fit_time", float("inf")))

    return results


def run_nn_benchmark(
    name: str,
    n_samples: int,
    n_features: int,
    seed: int = 42,
) -> List[Dict]:
    """Run neural network benchmarks."""
    print(f"\n{'=' * 70}")
    print(f"Neural Network Benchmark: {name}")
    print(f"{'=' * 70}")
    print(f"Data: {n_samples:,} samples x {n_features} features")
    print("-" * 70)

    # Generate data
    X_train, y_train, _ = generate_data(int(n_samples * 0.8), n_features, seed=seed)
    X_test, y_test, _ = generate_data(int(n_samples * 0.2), n_features, seed=seed + 1)

    results = []

    # Run NN benchmarks
    benchmarks = [
        ("LynxLearn NN", benchmark_lynxlearn_nn),
        ("scikit-learn MLP", benchmark_sklearn_mlp),
        ("PyTorch NN", benchmark_pytorch_nn),
    ]

    for bench_name, bench_func in benchmarks:
        print(f"\n  Testing {bench_name}...")
        try:
            result = bench_func(X_train, y_train, X_test, y_test)
            if "error" not in result:
                results.append(result)
                print(f"    Fit time: {format_time(result['fit_time'])}")
                print(f"    Predict time: {format_time(result['predict_time'])}")
                print(f"    MSE: {result['mse']:.6f}")
                print(f"    R2: {result['r2']:.6f}")
            else:
                print(f"    {result['error']}")
        except Exception as e:
            print(f"    Error: {e}")

    # Sort by fit time
    results.sort(key=lambda x: x.get("fit_time", float("inf")))

    return results


def print_comparison_table(all_results: Dict[str, List[Dict]]) -> None:
    """Print comparison table of all results."""
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    for config_name, results in all_results.items():
        if not results:
            continue

        print(f"\n{config_name}:")
        print("-" * 60)
        print(f"{'Method':<25} {'Fit Time':<12} {'MSE':<12} {'R2':<10}")
        print("-" * 60)

        fastest_time = min(x.get("fit_time", float("inf")) for x in results)

        for r in results:
            name = r.get("name", "unknown")
            fit_time = r.get("fit_time", 0)
            mse = r.get("mse", 0)
            r2 = r.get("r2", 0)

            # Mark fastest
            marker = " *" if fit_time == fastest_time else ""

            print(
                f"{name:<25} {format_time(fit_time):<12} {mse:<12.6f} {r2:<10.6f}{marker}"
            )


def print_environment_info() -> None:
    """Print environment information."""
    print("\n" + "=" * 70)
    print("ENVIRONMENT INFORMATION")
    print("=" * 70)
    print(f"NumPy Version:    {np.__version__}")
    print(f"LynxLearn:        {LYNXLEARN_VERSION}")
    print(f"  - Available:    {'Yes' if LYNXLEARN_AVAILABLE else 'No'}")

    if SKLEARN_AVAILABLE:
        import sklearn

        print(f"scikit-learn:     {sklearn.__version__}")
    else:
        print("scikit-learn:     Not installed")

    if PYTORCH_AVAILABLE:
        print(f"PyTorch:          {torch.__version__}")
    else:
        print("PyTorch:          Not installed")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LynxLearn Cloud Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark")
    parser.add_argument(
        "--nn", action="store_true", help="Run neural network benchmarks"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("LynxLearn Cloud Benchmark (PIP-INSTALLED PACKAGE)")
    print("=" * 70)
    print("\nThis benchmark uses the pip-installed lynxlearn package.")
    print("Make sure you have installed it: pip install lynxlearn")

    # Print environment info
    print_environment_info()

    if not LYNXLEARN_AVAILABLE:
        print("\n[!] LynxLearn is not installed. Please install it first:")
        print("    pip install lynxlearn")
        return

    # Run benchmarks
    all_results = {}

    if args.quick:
        configs = [
            ("Small (1000 x 10)", 1000, 10),
            ("Medium (10000 x 50)", 10000, 50),
        ]
    else:
        configs = [
            ("Tiny (100 x 5)", 100, 5),
            ("Small (1000 x 10)", 1000, 10),
            ("Medium (10000 x 50)", 10000, 50),
            ("Large (100000 x 100)", 100000, 100),
        ]

    for config_name, n_samples, n_features in configs:
        results = run_single_benchmark(config_name, n_samples, n_features)
        all_results[config_name] = results

    # Print comparison
    print_comparison_table(all_results)

    # Run neural network benchmarks if requested
    if args.nn:
        nn_results = {}
        nn_configs = [
            ("NN Small (1000 x 10)", 1000, 10),
            ("NN Medium (5000 x 20)", 5000, 20),
        ]
        for config_name, n_samples, n_features in nn_configs:
            results = run_nn_benchmark(config_name, n_samples, n_features)
            nn_results[config_name] = results

        print_comparison_table(nn_results)

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
    print(f"\nLynxLearn version tested: {LYNXLEARN_VERSION}")
    print("View on PyPI: https://pypi.org/project/lynxlearn/")


if __name__ == "__main__":
    main()
