"""
Benchmark Runner for LynxLearn vs Scikit-learn, PyTorch, TensorFlow

This script benchmarks linear regression implementations across multiple libraries.
For libraries not installed locally (PyTorch, TensorFlow), use the Colab notebook:
    benchmark/benchmark_colab.ipynb

Usage:
    python benchmark_runner.py

For full benchmark with PyTorch + TensorFlow:
    1. Upload and run benchmark/benchmark_colab.ipynb in Google Colab
    2. Download colab_benchmark.json and place in benchmark/ folder
    3. Run: python benchmark_runner.py
"""

import numpy as np
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lynxlearn.linear_model import LinearRegression, GradientDescentRegressor, Ridge
from lynxlearn.model_selection import train_test_split
from lynxlearn import metrics

# Try to import scikit-learn
try:
    from sklearn.linear_model import LinearRegression as SklearnLR
    from sklearn.linear_model import Ridge as SklearnRidge
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[!] Scikit-learn not installed. Install with: pip install scikit-learn")

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Try to import TensorFlow
try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic regression data."""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features) * 2
    true_bias = 5.0
    y = X @ true_weights + true_bias + np.random.randn(n_samples) * noise
    return X, y, true_weights, true_bias


def benchmark_lynxlearn(X_train, y_train, X_test, y_test):
    """Benchmark LynxLearn models."""
    results = {}

    # Linear Regression (OLS)
    start = time.time()
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = lr.predict(X_test)
    predict_time = time.time() - start

    results["LynxLearn_OLS"] = {
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": metrics.mse(y_test, y_pred),
        "rmse": metrics.rmse(y_test, y_pred),
        "mae": metrics.mae(y_test, y_pred),
        "r2": metrics.r2_score(y_test, y_pred),
        "weights": lr.weights.tolist()
        if hasattr(lr.weights, "tolist")
        else list(lr.weights),
        "bias": float(lr.bias),
    }

    # Gradient Descent
    start = time.time()
    gd = GradientDescentRegressor(learning_rate=0.01, n_iterations=1000)
    gd.fit(X_train, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = gd.predict(X_test)
    predict_time = time.time() - start

    results["LynxLearn_GD"] = {
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": metrics.mse(y_test, y_pred),
        "rmse": metrics.rmse(y_test, y_pred),
        "mae": metrics.mae(y_test, y_pred),
        "r2": metrics.r2_score(y_test, y_pred),
        "iterations": gd.n_iter_,
        "final_cost": gd.cost_history[-1] if gd.cost_history else None,
    }

    # Ridge Regression
    start = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = ridge.predict(X_test)
    predict_time = time.time() - start

    results["LynxLearn_Ridge"] = {
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": metrics.mse(y_test, y_pred),
        "rmse": metrics.rmse(y_test, y_pred),
        "mae": metrics.mae(y_test, y_pred),
        "r2": metrics.r2_score(y_test, y_pred),
    }

    return results


def benchmark_sklearn(X_train, y_train, X_test, y_test):
    """Benchmark Scikit-learn models."""
    if not SKLEARN_AVAILABLE:
        return {}

    results = {}

    # Linear Regression
    start = time.time()
    lr = SklearnLR()
    lr.fit(X_train, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = lr.predict(X_test)
    predict_time = time.time() - start

    results["Sklearn_OLS"] = {
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": np.mean(np.abs(y_test - y_pred)),
        "r2": r2_score(y_test, y_pred),
        "weights": lr.coef_.tolist() if hasattr(lr.coef_, "tolist") else list(lr.coef_),
        "bias": float(lr.intercept_),
    }

    # SGD Regressor (Gradient Descent)
    start = time.time()
    sgd = SGDRegressor(max_iter=1000, tol=1e-6, learning_rate="constant", eta0=0.01)
    sgd.fit(X_train, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = sgd.predict(X_test)
    predict_time = time.time() - start

    results["Sklearn_SGD"] = {
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": np.mean(np.abs(y_test - y_pred)),
        "r2": r2_score(y_test, y_pred),
        "n_iter": sgd.n_iter_,
    }

    # Ridge Regression
    start = time.time()
    ridge = SklearnRidge(alpha=1.0)
    ridge.fit(X_train, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = ridge.predict(X_test)
    predict_time = time.time() - start

    results["Sklearn_Ridge"] = {
        "fit_time": fit_time,
        "predict_time": predict_time,
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": np.mean(np.abs(y_test - y_pred)),
        "r2": r2_score(y_test, y_pred),
    }

    return results


def load_colab_results():
    """Load unified Colab benchmark results."""
    results = {}

    # Look for the new single file
    colab_file = os.path.join(os.path.dirname(__file__), "colab_benchmark.json")

    # Fallback to old files for backwards compatibility
    if not os.path.exists(colab_file):
        old_file = os.path.join(os.path.dirname(__file__), "colab_results.json")
        if os.path.exists(old_file):
            colab_file = old_file

    if os.path.exists(colab_file):
        with open(colab_file, "r") as f:
            data = json.load(f)
            # The colab_benchmark.json has a 'results' key with all model results
            if "results" in data:
                results = data["results"]
            else:
                results = data  # Fallback for old format
        print(f"[OK] Loaded Colab results from {colab_file}")
        print(f"    Environment: {data.get('environment', 'Unknown')}")
        print(f"    Timestamp: {data.get('timestamp', 'Unknown')}")
    else:
        print(f"[!] Colab results not found at {colab_file}")
        print(
            "    Run benchmark_colab.ipynb in Google Colab and save colab_benchmark.json"
        )

    return results


def load_colab_data():
    """Load the exact dataset used in Colab for fair comparison."""
    data_file = os.path.join(os.path.dirname(__file__), "benchmark_data.json")

    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            data = json.load(f)

        X_train = np.array(data["X_train"])
        X_test = np.array(data["X_test"])
        y_train = np.array(data["y_train"])
        y_test = np.array(data["y_test"])

        print(f"[OK] Loaded Colab dataset from {data_file}")
        print(f"    Same data as Colab - fair comparison guaranteed!")

        return X_train, X_test, y_train, y_test, data

    return None, None, None, None, None


def print_results_table(all_results, dataset_info, external_libs=None):
    """Print formatted benchmark results with best values marked."""
    if external_libs is None:
        external_libs = set()

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    print(
        f"Dataset: {dataset_info['n_samples']} samples, {dataset_info['n_features']} features"
    )
    print(
        f"Train size: {dataset_info['train_size']}, Test size: {dataset_info['test_size']}"
    )
    print("-" * 100)

    # Find best values across all results
    best_mse = float("inf")
    best_r2 = float("-inf")

    for result in all_results.values():
        mse = result.get("mse")
        r2 = result.get("r2")
        if isinstance(mse, (int, float)) and mse < best_mse:
            best_mse = mse
        if isinstance(r2, (int, float)) and r2 > best_r2:
            best_r2 = r2

    # Header
    print(
        f"{'Model':<30} {'Fit Time (s)':<18} {'Predict (s)':<18} {'MSE':<14} {'R²':<12}"
    )
    print("-" * 100)

    # Group results by library
    libraries = {"LynxLearn": [], "Scikit-learn": [], "PyTorch": [], "TensorFlow": []}

    for model_name, result in all_results.items():
        if "LynxLearn" in model_name:
            libraries["LynxLearn"].append((model_name, result, False))
        elif "Sklearn" in model_name:
            libraries["Scikit-learn"].append((model_name, result, False))
        elif "PyTorch" in model_name:
            libraries["PyTorch"].append(
                (model_name, result, "PyTorch" in external_libs)
            )
        elif "TensorFlow" in model_name:
            libraries["TensorFlow"].append(
                (model_name, result, "TensorFlow" in external_libs)
            )

    # Print each library section
    for lib_name, models in libraries.items():
        if not models:
            continue
        print(f"\n[{lib_name}]")
        for model_name, result, is_external in models:
            fit_time = result.get("fit_time", "N/A")
            predict_time = result.get("predict_time", "N/A")
            mse = result.get("mse", "N/A")
            r2 = result.get("r2", "N/A")

            # Add (Colab) marker for external results on TIME metrics only
            if is_external:
                fit_str = (
                    f"{fit_time:.6f} (Colab)"
                    if isinstance(fit_time, (int, float))
                    else f"{fit_time} (Colab)"
                )
                pred_str = (
                    f"{predict_time:.6f} (Colab)"
                    if isinstance(predict_time, (int, float))
                    else f"{predict_time} (Colab)"
                )
            else:
                fit_str = (
                    f"{fit_time:.6f}"
                    if isinstance(fit_time, (int, float))
                    else str(fit_time)
                )
                pred_str = (
                    f"{predict_time:.6f}"
                    if isinstance(predict_time, (int, float))
                    else str(predict_time)
                )

            # Format MSE and R² with * marker for best values
            if isinstance(mse, (int, float)):
                mse_str = f"{mse:.6f}*" if abs(mse - best_mse) < 1e-10 else f"{mse:.6f}"
            else:
                mse_str = str(mse)

            if isinstance(r2, (int, float)):
                r2_str = f"{r2:.6f}*" if abs(r2 - best_r2) < 1e-10 else f"{r2:.6f}"
            else:
                r2_str = str(r2)

            print(
                f"{model_name:<30} {fit_str:<18} {pred_str:<18} {mse_str:<14} {r2_str:<12}"
            )

    # Add note about best values and Colab results
    print("\n" + "-" * 100)
    print("NOTE: Values marked with '*' indicate the best (lowest MSE / highest R²)")
    if external_libs:
        print("      Results marked with '(Colab)' were run on Google Colab hardware.")
        print(
            "      Time comparisons between local and Colab are not directly comparable."
        )

    print("=" * 100)


def save_results(all_results, dataset_info):
    """Save benchmark results to JSON file."""
    output = {
        "dataset_info": dataset_info,
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_file = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results saved to: {output_file}")


def main():
    """Main benchmark function."""
    print("=" * 100)
    print("LINEAR REGRESSION BENCHMARK")
    print("LynxLearn vs Scikit-learn vs PyTorch vs TensorFlow")
    print("=" * 100)

    # Try to load Colab data first (for fair comparison)
    print("\n[1] Checking for Colab dataset...")
    X_train_colab, X_test_colab, y_train_colab, y_test_colab, colab_data = (
        load_colab_data()
    )

    if X_train_colab is not None:
        # Use the exact same data as Colab
        X_train, X_test, y_train, y_test = (
            X_train_colab,
            X_test_colab,
            y_train_colab,
            y_test_colab,
        )
        dataset_info = {
            "n_samples": colab_data["n_samples"],
            "n_features": colab_data["n_features"],
            "train_size": colab_data["train_size"],
            "test_size": colab_data["test_size"],
            "true_weights": colab_data["true_weights"],
            "true_bias": colab_data["true_bias"],
        }
        print(f"    Using Colab dataset for fair comparison!")
    else:
        # Generate new data
        print("    [i] Colab data not found. Generating new dataset...")
        print("    (For fair comparison, run benchmark_colab.ipynb in Colab first)")
        X, y, true_weights, true_bias = generate_data(
            n_samples=5000, n_features=20, noise=0.5
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        dataset_info = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "train_size": len(y_train),
            "test_size": len(y_test),
            "true_weights": true_weights.tolist(),
            "true_bias": float(true_bias),
        }

    print(
        f"    Samples: {dataset_info['n_samples']}, Features: {dataset_info['n_features']}"
    )
    print(f"    Train: {dataset_info['train_size']}, Test: {dataset_info['test_size']}")

    all_results = {}

    # Benchmark LynxLearn
    print("\n[2] Benchmarking LynxLearn...")
    lb_results = benchmark_lynxlearn(X_train, y_train, X_test, y_test)
    all_results.update(lb_results)
    print(f"    [OK] Benchmarked {len(lb_results)} models")

    # Benchmark Scikit-learn
    if SKLEARN_AVAILABLE:
        print("\n[3] Benchmarking Scikit-learn...")
        sklearn_results = benchmark_sklearn(X_train, y_train, X_test, y_test)
        all_results.update(sklearn_results)
        print(f"    [OK] Benchmarked {len(sklearn_results)} models")
    else:
        print("\n[3] Skipping Scikit-learn (not installed)")

    # Load unified Colab results (PyTorch + TensorFlow + optional LynxLearn)
    print("\n[4] Loading Colab benchmark results...")
    colab_results = load_colab_results()

    # Check for missing Colab results and provide instructions
    if not colab_results:
        print("\n[!] Colab results not found.")
        print("    To benchmark PyTorch and TensorFlow:")
        print("    1. Open Google Colab: https://colab.research.google.com")
        print("    2. Upload and run: benchmark/benchmark_colab.ipynb")
        print("    3. Download colab_benchmark.json")
        print("    4. Place both files in the benchmark/ folder")
        print("    5. Re-run this script")

    # Track that all Colab results are external
    external_libs = set(["PyTorch", "TensorFlow"])

    # Merge Colab results
    for model_name, result in colab_results.items():
        all_results[model_name] = result

    # Print results
    print_results_table(all_results, dataset_info, external_libs)

    # Save results
    save_results(all_results, dataset_info)

    # Summary
    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE")
    print("=" * 100)
    print(f"Total models benchmarked: {len(all_results)}")
    print(f"Results saved to: benchmark/benchmark_results.json")

    if not PYTORCH_AVAILABLE and "PyTorch_LSTSQ" not in colab_results:
        print(
            "\n[NOTE] PyTorch results missing - run benchmark_colab.ipynb in Google Colab"
        )
    if not TENSORFLOW_AVAILABLE and "TensorFlow_LSTSQ" not in colab_results:
        print(
            "[NOTE] TensorFlow results missing - run benchmark_colab.ipynb in Google Colab"
        )


if __name__ == "__main__":
    main()
