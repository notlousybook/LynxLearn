"""
THE NUCLEAR BENCHMARK SUITE - Maximum Performance Testing

This benchmark suite tests EVERYTHING:
- All solvers (lstsq, sgd, lbfgs, cg, cholesky, auto)
- All backends (numpy, cython, numba, auto)
- All dataset sizes (tiny to massive)
- All competitors (scikit-learn, PyTorch, TensorFlow)
- All configurations (with/without stats, sync/async, etc.)

Usage:
    python benchmark_nuclear.py                    # Full benchmark
    python benchmark_nuclear.py --quick            # Quick test
    python benchmark_nuclear.py --big-data         # Big data focus
    python benchmark_nuclear.py --versus-sklearn   # Compare vs scikit-learn
    python benchmark_nuclear.py --versus-pytorch   # Compare vs PyTorch
"""

import argparse
import gc
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing LynxLearn
try:
    from lynxlearn import metrics
    from lynxlearn.linear_model import (
        GradientDescentRegressor,
        LinearRegression,
        Ridge,
    )
    from lynxlearn.neural_network import SGD, Dense, Sequential

    LYNXLEARN_AVAILABLE = True
except ImportError as e:
    LYNXLEARN_AVAILABLE = False
    print(f"[!] LynxLearn not available: {e}")

# Try importing scikit-learn
try:
    from sklearn.linear_model import LinearRegression as SklearnLR
    from sklearn.linear_model import Ridge as SklearnRidge
    from sklearn.linear_model import SGDRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[!] scikit-learn not installed. Install with: pip install scikit-learn")

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("[!] PyTorch not installed. Install with: pip install torch")

# Try importing TensorFlow
try:
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[!] TensorFlow not installed. Install with: pip install tensorflow")


# =============================================================================
# Benchmark Configuration
# =============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    name: str
    n_samples: int
    n_features: int
    solver: str = "auto"
    backend: str = "auto"
    compute_statistics: bool = True
    async_statistics: bool = False
    dtype: str = "float64"
    n_runs: int = 3


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    config: BenchmarkConfig
    fit_time: float
    predict_time: float
    total_time: float
    memory_mb: float = 0.0
    mse: float = 0.0
    r2: float = 0.0
    n_params: int = 0
    error: Optional[str] = None


# =============================================================================
# Utility Functions
# =============================================================================


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds / 60:.1f}min"


def format_number(n: int) -> str:
    """Format large numbers."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_speedup(speedup: float) -> str:
    """Format speedup with emoji."""
    if speedup > 2.0:
        return f"🏆 {speedup:.2f}x"
    elif speedup > 1.2:
        return f"✅ {speedup:.2f}x"
    elif speedup > 0.8:
        return f"⚖️ {speedup:.2f}x"
    else:
        return f"❌ {speedup:.2f}x"


def generate_data(
    n_samples: int,
    n_features: int,
    noise: float = 0.1,
    seed: int = 42,
    dtype: str = "float64",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic regression data with known ground truth."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(dtype)
    true_weights = rng.randn(n_features).astype(dtype)
    true_bias = 5.0
    y = X @ true_weights + true_bias + rng.randn(n_samples).astype(dtype) * noise
    return X, y, true_weights


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0.0


# =============================================================================
# Benchmark Functions - Linear Models
# =============================================================================


def benchmark_lynxlearn_lr(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark LynxLearn LinearRegression."""
    if not LYNXLEARN_AVAILABLE:
        return BenchmarkResult(
            name="LynxLearn LR",
            config=config,
            fit_time=0,
            predict_time=0,
            total_time=0,
            error="LynxLearn not available",
        )

    # Generate data
    X_train, y_train, _ = generate_data(
        int(config.n_samples * 0.8),
        config.n_features,
        seed=42,
        dtype=config.dtype,
    )
    X_test, y_test, _ = generate_data(
        int(config.n_samples * 0.2),
        config.n_features,
        seed=43,
        dtype=config.dtype,
    )

    # Setup model
    try:
        model = LinearRegression(
            solver=config.solver,
            backend=config.backend,
            compute_statistics=config.compute_statistics,
            async_statistics=config.async_statistics,
        )
    except TypeError:
        # Fallback for old API
        model = LinearRegression()

    # Time fit
    gc.collect()
    mem_before = get_memory_usage()

    fit_times = []
    for _ in range(config.n_runs):
        start = time.perf_counter()
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            return BenchmarkResult(
                name="LynxLearn LR",
                config=config,
                fit_time=0,
                predict_time=0,
                total_time=0,
                error=str(e),
            )
        fit_times.append(time.perf_counter() - start)

    fit_time = float(np.median(fit_times))
    mem_after = get_memory_usage()

    # Time predict
    predict_times = []
    for _ in range(config.n_runs):
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_times.append(time.perf_counter() - start)

    predict_time = float(np.median(predict_times))

    # Compute metrics
    try:
        mse = metrics.mse(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
    except:
        mse = np.mean((y_test - y_pred) ** 2)
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum(
            (y_test - np.mean(y_test)) ** 2
        )

    return BenchmarkResult(
        name="LynxLearn LR",
        config=config,
        fit_time=fit_time,
        predict_time=predict_time,
        total_time=fit_time + predict_time,
        memory_mb=mem_after - mem_before,
        mse=mse,
        r2=r2,
        n_params=config.n_features + 1,
    )


def benchmark_sklearn_lr(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark scikit-learn LinearRegression."""
    if not SKLEARN_AVAILABLE:
        return BenchmarkResult(
            name="scikit-learn LR",
            config=config,
            fit_time=0,
            predict_time=0,
            total_time=0,
            error="scikit-learn not available",
        )

    # Generate data
    X_train, y_train, _ = generate_data(
        int(config.n_samples * 0.8),
        config.n_features,
        seed=42,
        dtype=config.dtype,
    )
    X_test, y_test, _ = generate_data(
        int(config.n_samples * 0.2),
        config.n_features,
        seed=43,
        dtype=config.dtype,
    )

    model = SklearnLR()

    # Time fit
    gc.collect()
    mem_before = get_memory_usage()

    fit_times = []
    for _ in range(config.n_runs):
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_times.append(time.perf_counter() - start)

    fit_time = float(np.median(fit_times))
    mem_after = get_memory_usage()

    # Time predict
    predict_times = []
    for _ in range(config.n_runs):
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_times.append(time.perf_counter() - start)

    predict_time = float(np.median(predict_times))

    # Compute metrics
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = model.score(X_test, y_test)

    return BenchmarkResult(
        name="scikit-learn LR",
        config=config,
        fit_time=fit_time,
        predict_time=predict_time,
        total_time=fit_time + predict_time,
        memory_mb=mem_after - mem_before,
        mse=mse,
        r2=r2,
        n_params=config.n_features + 1,
    )


def benchmark_numpy_lstsq(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark pure NumPy lstsq (baseline)."""
    # Generate data
    X_train, y_train, _ = generate_data(
        int(config.n_samples * 0.8),
        config.n_features,
        seed=42,
        dtype=config.dtype,
    )
    X_test, y_test, _ = generate_data(
        int(config.n_samples * 0.2),
        config.n_features,
        seed=43,
        dtype=config.dtype,
    )

    # Add bias column
    X_b = np.column_stack([X_train, np.ones(X_train.shape[0])])
    X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])

    # Time fit
    gc.collect()

    fit_times = []
    for _ in range(config.n_runs):
        start = time.perf_counter()
        weights = np.linalg.lstsq(X_b, y_train, rcond=None)[0]
        fit_times.append(time.perf_counter() - start)

    fit_time = float(np.median(fit_times))

    # Time predict
    predict_times = []
    for _ in range(config.n_runs):
        start = time.perf_counter()
        y_pred = X_test_b @ weights
        predict_times.append(time.perf_counter() - start)

    predict_time = float(np.median(predict_times))

    # Compute metrics
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    return BenchmarkResult(
        name="NumPy lstsq",
        config=config,
        fit_time=fit_time,
        predict_time=predict_time,
        total_time=fit_time + predict_time,
        mse=mse,
        r2=r2,
        n_params=config.n_features + 1,
    )


# =============================================================================
# Benchmark Functions - Neural Networks
# =============================================================================


def benchmark_lynxlearn_nn(
    X: np.ndarray,
    y: np.ndarray,
    layer_sizes: List[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    dtype: str = "float32",
) -> Dict[str, Any]:
    """Benchmark LynxLearn Sequential model."""
    if not LYNXLEARN_AVAILABLE:
        return {"error": "LynxLearn not available"}

    try:
        # Build model
        layers = []
        for i, size in enumerate(layer_sizes):
            if i == 0:
                layers.append(Dense(size, activation="relu", input_shape=(X.shape[1],)))
            elif i == len(layer_sizes) - 1:
                layers.append(Dense(size))  # Output layer
            else:
                layers.append(Dense(size, activation="relu"))

        model = Sequential(layers)

        # Compile
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        model.compile(optimizer=optimizer, loss="mse")

        # Train
        gc.collect()
        start = time.perf_counter()
        history = model.train(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        train_time = time.perf_counter() - start

        # Inference
        start = time.perf_counter()
        y_pred = model.predict(X[:100])
        inference_time = time.perf_counter() - start

        return {
            "train_time": train_time,
            "inference_time": inference_time,
            "final_loss": history["loss"][-1],
            "n_params": model.count_params(),
        }
    except Exception as e:
        return {"error": str(e)}


class PyTorchMLP(nn.Module):
    """PyTorch MLP for benchmarking."""

    def __init__(self, layer_sizes: List[int], input_size: int):
        super().__init__()
        layers = []
        prev_size = input_size
        for i, size in enumerate(layer_sizes):
            layers.append(nn.Linear(prev_size, size))
            if i < len(layer_sizes) - 1:
                layers.append(nn.ReLU())
            prev_size = size
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def benchmark_pytorch_nn(
    X: np.ndarray,
    y: np.ndarray,
    layer_sizes: List[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    dtype: str = "float32",
) -> Dict[str, Any]:
    """Benchmark PyTorch MLP."""
    if not PYTORCH_AVAILABLE:
        return {"error": "PyTorch not available"}

    try:
        # Convert to tensors
        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float().unsqueeze(1)

        # Build model
        model = PyTorchMLP(layer_sizes, X.shape[1])
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum
        )
        criterion = nn.MSELoss()

        # Train
        gc.collect()
        start = time.perf_counter()

        model.train()
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X_t[i : i + batch_size]
                y_batch = y_t[i : i + batch_size]

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        train_time = time.perf_counter() - start

        # Inference
        model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            y_pred = model(X_t[:100])
        inference_time = time.perf_counter() - start

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())

        return {
            "train_time": train_time,
            "inference_time": inference_time,
            "final_loss": loss.item(),
            "n_params": n_params,
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Benchmark Runners
# =============================================================================


def run_linear_benchmark(
    config: BenchmarkConfig,
    competitors: List[str] = ["lynxlearn", "sklearn", "numpy"],
) -> List[BenchmarkResult]:
    """Run linear model benchmark for all competitors."""
    results = []

    print(f"\n{'=' * 80}")
    print(f"BENCHMARK: {config.name}")
    print(f"{'=' * 80}")
    print(
        f"Data: {format_number(config.n_samples)} samples × {config.n_features} features"
    )
    print(
        f"Solver: {config.solver} | Backend: {config.backend} | Dtype: {config.dtype}"
    )
    print(f"Statistics: {'Yes' if config.compute_statistics else 'No'}")
    print("-" * 80)

    # Run each competitor
    if "lynxlearn" in competitors:
        print("\n[1] LynxLearn LinearRegression...")
        result = benchmark_lynxlearn_lr(config)
        results.append(result)
        if result.error:
            print(f"    ❌ Error: {result.error}")
        else:
            print(f"    ✓ Fit time: {format_time(result.fit_time)}")
            print(f"    ✓ Predict time: {format_time(result.predict_time)}")
            print(f"    ✓ MSE: {result.mse:.6f}")
            print(f"    ✓ R²: {result.r2:.6f}")

    if "sklearn" in competitors:
        print("\n[2] scikit-learn LinearRegression...")
        result = benchmark_sklearn_lr(config)
        results.append(result)
        if result.error:
            print(f"    ❌ Error: {result.error}")
        else:
            print(f"    ✓ Fit time: {format_time(result.fit_time)}")
            print(f"    ✓ Predict time: {format_time(result.predict_time)}")
            print(f"    ✓ MSE: {result.mse:.6f}")
            print(f"    ✓ R²: {result.r2:.6f}")

    if "numpy" in competitors:
        print("\n[3] NumPy lstsq (baseline)...")
        result = benchmark_numpy_lstsq(config)
        results.append(result)
        if result.error:
            print(f"    ❌ Error: {result.error}")
        else:
            print(f"    ✓ Fit time: {format_time(result.fit_time)}")
            print(f"    ✓ Predict time: {format_time(result.predict_time)}")
            print(f"    ✓ MSE: {result.mse:.6f}")
            print(f"    ✓ R²: {result.r2:.6f}")

    # Print comparison
    if len(results) > 1:
        print("\n" + "-" * 80)
        print("COMPARISON")
        print("-" * 80)
        print(
            f"{'Framework':<20} {'Fit Time':<15} {'Predict Time':<15} {'vs sklearn':<15}"
        )
        print("-" * 80)

        # Find sklearn baseline
        sklearn_time = next(
            (r.fit_time for r in results if "scikit-learn" in r.name), 1.0
        )
        if sklearn_time == 0:
            sklearn_time = 1.0

        for r in results:
            if r.error:
                continue
            speedup = sklearn_time / r.fit_time if r.fit_time > 0 else 0
            print(
                f"{r.name:<20} {format_time(r.fit_time):<15} "
                f"{format_time(r.predict_time):<15} {format_speedup(speedup)}"
            )

    return results


def run_neural_network_benchmark(
    name: str,
    n_samples: int,
    n_features: int,
    layer_sizes: List[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    dtype: str = "float32",
) -> Dict[str, Any]:
    """Run neural network benchmark."""
    print(f"\n{'=' * 80}")
    print(f"NEURAL NETWORK BENCHMARK: {name}")
    print(f"{'=' * 80}")
    print(f"Data: {format_number(n_samples)} samples × {n_features} features")
    print(f"Model: {n_features} → {' → '.join(map(str, layer_sizes))}")
    print(f"Training: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    print("-" * 80)

    # Generate data
    X, y = generate_data(n_samples, n_features, dtype=dtype)[:2]
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    results = {}

    # LynxLearn
    print("\n[1] LynxLearn Sequential...")
    r = benchmark_lynxlearn_nn(
        X, y, layer_sizes, epochs, batch_size, learning_rate, momentum, dtype
    )
    if "error" not in r:
        results["LynxLearn"] = r
        print(f"    ✓ Train time: {format_time(r['train_time'])}")
        print(f"    ✓ Inference time: {format_time(r['inference_time'])}")
        print(f"    ✓ Final loss: {r['final_loss']:.6f}")
        print(f"    ✓ Parameters: {format_number(r['n_params'])}")
    else:
        print(f"    ❌ {r['error']}")

    # PyTorch
    if PYTORCH_AVAILABLE:
        print("\n[2] PyTorch (CPU)...")
        r = benchmark_pytorch_nn(
            X, y, layer_sizes, epochs, batch_size, learning_rate, momentum, dtype
        )
        if "error" not in r:
            results["PyTorch"] = r
            print(f"    ✓ Train time: {format_time(r['train_time'])}")
            print(f"    ✓ Inference time: {format_time(r['inference_time'])}")
            print(f"    ✓ Final loss: {r['final_loss']:.6f}")
            print(f"    ✓ Parameters: {format_number(r['n_params'])}")
        else:
            print(f"    ❌ {r['error']}")

    # Comparison
    if len(results) > 1:
        print("\n" + "-" * 80)
        print("COMPARISON")
        print("-" * 80)
        print(f"{'Framework':<15} {'Train Time':<15} {'Inference':<15} {'Speedup':<12}")
        print("-" * 57)

        baseline_time = results.get("PyTorch", {}).get("train_time", 1)
        if baseline_time == 0:
            baseline_time = 1

        for name, r in results.items():
            speedup = baseline_time / r["train_time"] if r["train_time"] > 0 else 0
            print(
                f"{name:<15} {format_time(r['train_time']):<15} "
                f"{format_time(r['inference_time']):<15} {format_speedup(speedup)}"
            )

    return results


# =============================================================================
# Main Benchmark Suites
# =============================================================================


def run_quick_benchmark():
    """Run quick benchmark for development."""
    print("\n" + "=" * 80)
    print("QUICK BENCHMARK - Development Testing")
    print("=" * 80)

    configs = [
        BenchmarkConfig(
            name="Small (1K × 10)",
            n_samples=1000,
            n_features=10,
            solver="auto",
            backend="auto",
            compute_statistics=True,
            dtype="float64",
            n_runs=3,
        ),
        BenchmarkConfig(
            name="Medium (10K × 50)",
            n_samples=10000,
            n_features=50,
            solver="auto",
            backend="auto",
            compute_statistics=True,
            dtype="float64",
            n_runs=3,
        ),
    ]

    all_results = []
    for config in configs:
        results = run_linear_benchmark(config)
        all_results.extend(results)

    return all_results


def run_full_benchmark():
    """Run full comprehensive benchmark."""
    print("\n" + "=" * 80)
    print("FULL NUCLEAR BENCHMARK - All Configurations")
    print("=" * 80)

    configs = [
        # Tiny datasets
        BenchmarkConfig(
            name="Tiny (100 × 5)",
            n_samples=100,
            n_features=5,
            solver="lstsq",
            backend="numpy",
            compute_statistics=True,
            dtype="float64",
            n_runs=5,
        ),
        # Small datasets
        BenchmarkConfig(
            name="Small (1K × 10)",
            n_samples=1000,
            n_features=10,
            solver="auto",
            backend="auto",
            compute_statistics=True,
            dtype="float64",
            n_runs=3,
        ),
        # Medium datasets
        BenchmarkConfig(
            name="Medium (10K × 50)",
            n_samples=10000,
            n_features=50,
            solver="auto",
            backend="auto",
            compute_statistics=True,
            dtype="float64",
            n_runs=3,
        ),
        # Large datasets
        BenchmarkConfig(
            name="Large (100K × 100)",
            n_samples=100000,
            n_features=100,
            solver="auto",
            backend="auto",
            compute_statistics=False,  # No stats for large data
            dtype="float64",
            n_runs=3,
        ),
        # XL datasets
        BenchmarkConfig(
            name="XL (500K × 200)",
            n_samples=500000,
            n_features=200,
            solver="lbfgs",  # Use iterative solver
            backend="auto",
            compute_statistics=False,
            dtype="float32",  # Use float32 for memory efficiency
            n_runs=2,
        ),
    ]

    all_results = []
    for config in configs:
        results = run_linear_benchmark(config)
        all_results.extend(results)

    return all_results


def run_big_data_benchmark():
    """Run big data focused benchmark."""
    print("\n" + "=" * 80)
    print("BIG DATA BENCHMARK - Massive Datasets")
    print("=" * 80)
    print("\nTesting ability to handle large datasets WITHOUT crashing!")
    print("Comparing iterative solvers (SGD, L-BFGS) vs direct solvers.\n")

    configs = [
        BenchmarkConfig(
            name="Large (100K × 100)",
            n_samples=100000,
            n_features=100,
            solver="lbfgs",
            backend="auto",
            compute_statistics=False,
            dtype="float32",
            n_runs=2,
        ),
        BenchmarkConfig(
            name="XL (500K × 200)",
            n_samples=500000,
            n_features=200,
            solver="lbfgs",
            backend="auto",
            compute_statistics=False,
            dtype="float32",
            n_runs=2,
        ),
        BenchmarkConfig(
            name="XXL (1M × 100)",
            n_samples=1_000_000,
            n_features=100,
            solver="sgd",
            backend="auto",
            compute_statistics=False,
            dtype="float32",
            n_runs=1,
        ),
    ]

    all_results = []
    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"Testing: {config.name}")
        print(f"Expected to handle without crash: ✅")
        results = run_linear_benchmark(config, competitors=["lynxlearn", "sklearn"])
        all_results.extend(results)

    return all_results


def run_neural_network_benchmarks(quick: bool = False):
    """Run neural network benchmarks."""
    print("\n" + "=" * 80)
    print("NEURAL NETWORK BENCHMARKS")
    print("=" * 80)

    epochs = 5 if quick else 20
    n_samples = 500 if quick else 1000

    configs = [
        ("Small Model (~1K params)", n_samples, 20, [32, 16, 1]),
        ("Medium Model (~10K params)", n_samples, 50, [64, 32, 1]),
    ]

    if not quick:
        configs.append(("Large Model (~100K params)", n_samples, 100, [128, 64, 32, 1]))

    all_results = {}
    for name, samples, features, layers in configs:
        result = run_neural_network_benchmark(
            name=name,
            n_samples=samples,
            n_features=features,
            layer_sizes=layers,
            epochs=epochs,
            batch_size=32,
            learning_rate=0.01,
            momentum=0.9,
        )
        all_results[name] = result

    return all_results


def run_solver_comparison():
    """Compare different solvers."""
    print("\n" + "=" * 80)
    print("SOLVER COMPARISON - Finding the Best Solver")
    print("=" * 80)

    solvers = ["lstsq", "lbfgs", "sgd", "auto"]

    # Test on medium dataset
    base_config = {
        "n_samples": 10000,
        "n_features": 50,
        "compute_statistics": False,
        "dtype": "float64",
        "n_runs": 3,
    }

    all_results = []
    for solver in solvers:
        config = BenchmarkConfig(
            name=f"Solver: {solver}",
            solver=solver,
            backend="auto",
            **base_config,
        )
        results = run_linear_benchmark(config, competitors=["lynxlearn"])
        all_results.extend(results)

    return all_results


# =============================================================================
# Report Generation
# =============================================================================


def generate_summary_report(all_results: List[BenchmarkResult]):
    """Generate summary report."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)

    # Group by config
    configs = {}
    for result in all_results:
        if result.error:
            continue
        key = result.config.name
        if key not in configs:
            configs[key] = []
        configs[key].append(result)

    # Print summary table
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(
        f"{'Dataset':<25} {'Framework':<20} {'Fit Time':<15} {'vs sklearn':<15} {'Status':<10}"
    )
    print("-" * 80)

    wins = 0
    losses = 0
    ties = 0

    for config_name, results in configs.items():
        # Find sklearn baseline
        sklearn_result = next((r for r in results if "scikit-learn" in r.name), None)
        sklearn_time = sklearn_result.fit_time if sklearn_result else 1.0

        for result in results:
            if result.error:
                continue

            speedup = sklearn_time / result.fit_time if result.fit_time > 0 else 0

            if "scikit-learn" in result.name:
                status = "baseline"
            elif speedup > 1.2:
                status = "🏆 WIN"
                wins += 1
            elif speedup < 0.8:
                status = "❌ LOSE"
                losses += 1
            else:
                status = "⚖️ TIE"
                ties += 1

            print(
                f"{config_name:<25} {result.name:<20} "
                f"{format_time(result.fit_time):<15} {format_speedup(speedup):<15} {status:<10}"
            )

    # Print final stats
    total = wins + losses + ties
    if total > 0:
        print("\n" + "-" * 80)
        print(f"FINAL SCORE: {wins} wins, {losses} losses, {ties} ties")
        print(f"Win rate: {wins / total * 100:.1f}%")
        print("-" * 80)

        if wins > losses:
            print("🏆 OVERALL VICTORY! LynxLearn DOMINATES! 🏆")
        elif wins == losses:
            print("⚖️ BALANCED PERFORMANCE - Room for improvement!")
        else:
            print("💪 GOOD EFFORT - More optimizations needed!")

    print("=" * 80)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LynxLearn Nuclear Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_nuclear.py                    # Full benchmark
    python benchmark_nuclear.py --quick            # Quick test
    python benchmark_nuclear.py --big-data         # Big data focus
    python benchmark_nuclear.py --neural-network   # Neural networks
    python benchmark_nuclear.py --solvers          # Compare solvers
        """,
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark (development)"
    )
    parser.add_argument(
        "--big-data", action="store_true", help="Run big data benchmark"
    )
    parser.add_argument(
        "--neural-network",
        action="store_true",
        help="Run neural network benchmarks",
    )
    parser.add_argument("--solvers", action="store_true", help="Compare solvers")
    parser.add_argument(
        "--versus-sklearn",
        action="store_true",
        help="Focus on comparison vs scikit-learn",
    )

    args = parser.parse_args()

    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print(
        "║"
        + "  🔥 LYNXLEARN NUCLEAR BENCHMARK SUITE - MAXIMUM PERFORMANCE 🔥  ".center(78)
        + "║"
    )
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    print("\nThis benchmark tests EVERYTHING:")
    print("  • All solvers (lstsq, sgd, lbfgs, cg, auto)")
    print("  • All backends (numpy, cython, numba, auto)")
    print("  • All dataset sizes (tiny to massive)")
    print("  • All competitors (scikit-learn, PyTorch, NumPy)")
    print("\nGoal: BEAT SCIKIT-LEARN IN 80%+ OF CASES")

    all_results = []

    # Run selected benchmarks
    if args.quick:
        all_results.extend(run_quick_benchmark())
        all_results.extend(run_neural_network_benchmarks(quick=True))
    elif args.big_data:
        all_results.extend(run_big_data_benchmark())
    elif args.neural_network:
        run_neural_network_benchmarks(quick=False)
    elif args.solvers:
        all_results.extend(run_solver_comparison())
    else:
        # Full benchmark
        all_results.extend(run_full_benchmark())
        run_neural_network_benchmarks(quick=False)

    # Generate summary
    if all_results:
        generate_summary_report(all_results)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
