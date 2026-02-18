"""
CPU Benchmark: LynxLearn vs PyTorch Neural Networks

This benchmark compares the CPU performance of LynxLearn's pure NumPy
implementation against PyTorch on CPU.

Run this to see how different model sizes affect performance.

Usage:
    python benchmark_nn.py
    python benchmark_nn.py --epochs 50 --max-params 10000000
"""

import argparse
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lynxlearn.neural_network import SGD, Dense, MeanSquaredError, Sequential


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        return f"{seconds / 60:.1f} min"


def format_params(n: int) -> str:
    """Format parameter count in human-readable format."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def count_params_lynxlearn(model: Sequential) -> int:
    """Count parameters in LynxLearn model."""
    return model.count_params()


def create_lynxlearn_model(layer_sizes: List[int], input_dim: int) -> Sequential:
    """Create a LynxLearn Sequential model with given layer sizes."""
    layers = []

    # First layer
    layers.append(Dense(layer_sizes[0], activation="relu", input_shape=(input_dim,)))

    # Hidden layers
    for i in range(1, len(layer_sizes)):
        layers.append(Dense(layer_sizes[i], activation="relu"))

    # Output layer
    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss="mse")

    return model


def benchmark_lynxlearn_train(
    model: Sequential, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int
) -> Tuple[float, Dict]:
    """Benchmark LynxLearn training."""

    start_time = time.perf_counter()
    history = model.train(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    elapsed = time.perf_counter() - start_time

    return elapsed, history


def benchmark_lynxlearn_inference(
    model: Sequential, X: np.ndarray, n_runs: int = 10
) -> float:
    """Benchmark LynxLearn inference."""

    # Warmup
    model.predict(X[:10])

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - start)

    return np.median(times)


# ============================================================================
# PyTorch Benchmarks (optional)
# ============================================================================

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    pass


if TORCH_AVAILABLE:

    class PyTorchMLP(nn.Module):
        """PyTorch MLP matching LynxLearn architecture."""

        def __init__(self, layer_sizes: List[int], input_dim: int):
            super().__init__()

            layers = []
            prev_dim = input_dim

            for size in layer_sizes:
                layers.append(nn.Linear(prev_dim, size))
                layers.append(nn.ReLU())
                prev_dim = size

            layers.append(nn.Linear(prev_dim, 1))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    def count_params_pytorch(model: nn.Module) -> int:
        """Count parameters in PyTorch model."""
        return sum(p.numel() for p in model.parameters())

    def create_pytorch_model(layer_sizes: List[int], input_dim: int) -> nn.Module:
        """Create a PyTorch MLP model."""
        return PyTorchMLP(layer_sizes, input_dim)

    def benchmark_pytorch_train(
        model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int
    ) -> Tuple[float, Dict]:
        """Benchmark PyTorch training on CPU."""

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Setup
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.MSELoss()

        n_samples = X.shape[0]
        losses = []

        start_time = time.perf_counter()

        for epoch in range(epochs):
            # Shuffle
            indices = torch.randperm(n_samples)
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i : i + batch_size]
                batch_y = y_shuffled[i : i + batch_size]

                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            losses.append(epoch_loss / n_batches)

        elapsed = time.perf_counter() - start_time

        return elapsed, {"loss": losses}

    def benchmark_pytorch_inference(
        model: nn.Module, X: np.ndarray, n_runs: int = 10
    ) -> float:
        """Benchmark PyTorch inference on CPU."""

        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Warmup
        with torch.no_grad():
            model(X_tensor[:10])

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            with torch.no_grad():
                model(X_tensor)
            times.append(time.perf_counter() - start)

        return np.median(times)


# ============================================================================
# Main Benchmark
# ============================================================================


def get_model_configs(max_params: int) -> List[Tuple[str, List[int], int, int]]:
    """
    Generate model configurations up to max_params.
    Returns: (name, layer_sizes, input_dim, expected_params)
    """
    configs = []

    # Small models
    configs.append(("Tiny MLP", [32], 10, 32 * 10 + 32 + 32 + 1))  # ~353 params

    # Medium models
    configs.append(("Small MLP", [64, 32], 20, None))  # ~4K params
    configs.append(("Medium MLP", [128, 64, 32], 50, None))  # ~15K params
    configs.append(("Large MLP", [256, 128, 64], 100, None))  # ~65K params

    # Bigger models
    configs.append(("XL MLP", [512, 256, 128, 64], 200, None))  # ~260K params
    configs.append(("XXL MLP", [1024, 512, 256, 128], 500, None))  # ~1.3M params

    # Very large models (for CPU torture test)
    configs.append(("Huge MLP", [2048, 1024, 512, 256], 1000, None))  # ~5.2M params
    configs.append(("Massive MLP", [4096, 2048, 1024], 2000, None))  # ~25M params

    # Filter by max_params
    filtered = []
    for name, layers, input_dim, expected in configs:
        # Calculate actual params
        params = layers[0] * input_dim + layers[0]  # First layer
        for i in range(1, len(layers)):
            params += layers[i - 1] * layers[i] + layers[i]
        params += layers[-1] + 1  # Output layer

        if params <= max_params:
            filtered.append((name, layers, input_dim, params))

    return filtered


def run_benchmark(
    epochs: int = 10,
    max_params: int = 10_000_000,
    batch_size: int = 32,
    n_samples: int = 1000,
    n_runs: int = 3,
) -> None:
    """Run the full benchmark."""

    print("=" * 80)
    print("CPU Neural Network Benchmark: LynxLearn vs PyTorch")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Max params: {format_params(max_params)}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Samples: {n_samples}")
    print(f"  - Runs per config: {n_runs}")
    print(f"  - PyTorch available: {TORCH_AVAILABLE}")

    if not TORCH_AVAILABLE:
        print("\n  [!] PyTorch not installed. Install with: pip install torch")
        print("      Benchmarking LynxLearn only.\n")

    configs = get_model_configs(max_params)

    print(f"\n  Models to benchmark: {len(configs)}")
    print()

    # Results storage
    results = []

    for name, layer_sizes, input_dim, params in configs:
        print("=" * 80)
        print(f"Benchmarking: {name}")
        print(
            f"  Architecture: {input_dim} -> {' -> '.join(map(str, layer_sizes))} -> 1"
        )
        print(f"  Parameters: {format_params(params)} ({params:,})")
        print("-" * 80)

        # Generate data
        np.random.seed(42)
        X = np.random.randn(n_samples, input_dim).astype(np.float64)
        y = (
            np.sum(X[:, : min(5, input_dim)] ** 2, axis=1, keepdims=True)
            + np.random.randn(n_samples, 1) * 0.1
        )

        # LynxLearn benchmark
        print("\n  [1/2] LynxLearn (NumPy, float64)...")
        lynxlearn_times = []

        for run in range(n_runs):
            model_ll = create_lynxlearn_model(layer_sizes, input_dim)
            elapsed, history = benchmark_lynxlearn_train(
                model_ll, X, y, epochs, batch_size
            )
            lynxlearn_times.append(elapsed)
            print(
                f"        Run {run + 1}: {format_time(elapsed)} (loss: {history['loss'][-1]:.4f})"
            )

        ll_train_time = np.median(lynxlearn_times)
        ll_inference_time = benchmark_lynxlearn_inference(model_ll, X)

        print(f"        Median train time: {format_time(ll_train_time)}")
        print(f"        Inference time: {format_time(ll_inference_time)}")

        # PyTorch benchmark (if available)
        if TORCH_AVAILABLE:
            print("\n  [2/2] PyTorch (CPU, float32)...")
            pytorch_times = []

            for run in range(n_runs):
                model_pt = create_pytorch_model(layer_sizes, input_dim)
                model_pt.eval()
                elapsed, history = benchmark_pytorch_train(
                    model_pt, X, y, epochs, batch_size
                )
                pytorch_times.append(elapsed)
                print(
                    f"        Run {run + 1}: {format_time(elapsed)} (loss: {history['loss'][-1]:.4f})"
                )

            pt_train_time = np.median(pytorch_times)
            pt_inference_time = benchmark_pytorch_inference(model_pt, X)

            print(f"        Median train time: {format_time(pt_train_time)}")
            print(f"        Inference time: {format_time(pt_inference_time)}")

            # Comparison
            speedup = pt_train_time / ll_train_time
            print(
                f"\n  📊 Result: LynxLearn is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
            )

            results.append(
                {
                    "name": name,
                    "params": params,
                    "ll_train": ll_train_time,
                    "ll_inference": ll_inference_time,
                    "pt_train": pt_train_time,
                    "pt_inference": pt_inference_time,
                    "speedup": speedup,
                }
            )
        else:
            results.append(
                {
                    "name": name,
                    "params": params,
                    "ll_train": ll_train_time,
                    "ll_inference": ll_inference_time,
                    "pt_train": None,
                    "pt_inference": None,
                    "speedup": None,
                }
            )

        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if TORCH_AVAILABLE:
        header = f"{'Model':<15} {'Params':<12} {'LynxLearn':<12} {'PyTorch':<12} {'Speedup':<10}"
        print(header)
        print("-" * len(header))

        for r in results:
            print(
                f"{r['name']:<15} {format_params(r['params']):<12} "
                f"{format_time(r['ll_train']):<12} {format_time(r['pt_train']):<12} "
                f"{r['speedup']:.2f}x"
            )
    else:
        header = f"{'Model':<15} {'Params':<12} {'LynxLearn Train':<18} {'LynxLearn Inference':<20}"
        print(header)
        print("-" * len(header))

        for r in results:
            print(
                f"{r['name']:<15} {format_params(r['params']):<12} "
                f"{format_time(r['ll_train']):<18} {format_time(r['ll_inference']):<20}"
            )

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if TORCH_AVAILABLE:
        avg_speedup = np.mean([r["speedup"] for r in results])
        print(f"""
Average speedup: {avg_speedup:.2f}x

Why LynxLearn can be faster on CPU:

1. NO AUTOGRAD OVERHEAD
   - PyTorch tracks every operation for gradients
   - LynxLearn computes gradients directly with pre-derived formulas

2. NO DYNAMIC GRAPH
   - PyTorch builds computation graph each forward pass
   - LynxLearn has static, simple forward/backward functions

3. FEWER PYTHON CALLS
   - PyTorch: Python → C++ ATen → autograd → Python
   - LynxLearn: Python → NumPy (C) → BLAS → done

4. NO FRAMEWORK OVERHEAD
   - No CUDA checks, no device transfers, no distributed hooks
   - No safety checks, assertions, or debugging infrastructure

5. OPTIMIZED BATCH OPERATIONS
   - Single matrix multiply per layer (pure BLAS)
   - PyTorch has dispatch overhead for every op

When PyTorch wins:
- GPU acceleration (CUDA cuBLAS is 10-100x faster than CPU BLAS)
- Very large models (>100M params)
- Complex architectures (transformers, CNNs with custom ops)
- Mixed precision training
- Distributed training across multiple GPUs/TPUs
""")
    else:
        print("""
To compare with PyTorch, install it:
    pip install torch

Then run this benchmark again to see the comparison.
""")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LynxLearn vs PyTorch on CPU"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--max-params", type=int, default=10_000_000, help="Maximum parameters"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of training samples"
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration")

    args = parser.parse_args()

    run_benchmark(
        epochs=args.epochs,
        max_params=args.max_params,
        batch_size=args.batch_size,
        n_samples=args.samples,
        n_runs=args.runs,
    )


if __name__ == "__main__":
    main()
