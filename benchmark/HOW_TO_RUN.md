# How to Run Benchmark

## Quick Start

### Option 1: Local Only (No Colab needed)
```bash
cd c:\Users\lousy\Documents\Projects\LousyBookML
python benchmark/benchmark_runner.py
```

### Option 2: Full Benchmark (With PyTorch + TensorFlow)

**Step 1: Run in Google Colab**
1. Open https://colab.research.google.com
2. Upload `benchmark/benchmark_colab.ipynb`
3. Click Runtime → Run all
4. Download `colab_benchmark.json`

**Step 2: Run Locally**
```bash
# Copy colab_benchmark.json to benchmark/ folder
python benchmark/benchmark_runner.py
```

## What Gets Tested

| Library | Models |
|---------|--------|
| LousyBookML | OLS, Gradient Descent, Ridge |
| Scikit-learn | LinearRegression, SGD, Ridge |
| PyTorch | LSTSQ, SGD |
| TensorFlow | LSTSQ, Keras |

## Understanding Results

- `*` = Best performer (lowest MSE / highest R²)
- `(Colab)` = Run on different hardware (time not comparable)
- MSE/R² are comparable across all libraries (same data!)
