# Benchmark Suite for LynxLearn

This directory contains benchmarking tools to compare LynxLearn against other popular ML libraries using the **EXACT same dataset** for fair comparison.

## Supported Libraries

- **LynxLearn** (always runs locally)
- **Scikit-learn** (runs locally if installed)
- **PyTorch** (runs in Google Colab)
- **TensorFlow** (runs in Google Colab)

## Quick Start

### 1. Run Local Benchmarks

```bash
cd benchmark
python benchmark_runner.py
```

This will benchmark:
- LynxLearn (OLS, Gradient Descent, Ridge)
- Scikit-learn (if installed)

### 2. Run Colab Benchmark (Recommended)

The Jupyter notebook benchmarks **PyTorch + TensorFlow** all together using the **same dataset**:

1. Go to [Google Colab](https://colab.research.google.com)
2. Upload `benchmark/benchmark_colab.ipynb`
3. Click Runtime → Run all
4. Download `colab_benchmark.json`
5. Place the file in this `benchmark/` folder
6. Re-run `benchmark_runner.py`

**Why this is better:**
- All libraries use the **identical train/test split**
- Same random seed guarantees **exact same data**
- Fair comparison of accuracy (MSE/R²)
- Time comparisons are still marked as "(Colab)" to indicate different hardware

## Benchmark Results Format

After running all benchmarks, you'll have:

```
benchmark/
├── benchmark_results.json      # Combined results
├── colab_benchmark.json        # From Colab run (single file)
└── ...
```

## What Each Benchmark Measures

### Models Tested

| Library | OLS/Closed-form | Gradient Descent | Ridge Regression |
|---------|-----------------|------------------|------------------|
| LynxLearn | ✅ Normal Equation | ✅ Batch GD | ✅ L2 Regularized |
| Scikit-learn | ✅ LinearRegression | ✅ SGDRegressor | ✅ Ridge |
| PyTorch | ✅ torch.linalg.lstsq | ✅ nn.Linear + SGD | ❌ (not tested) |
| TensorFlow | ✅ tf.linalg.lstsq | ✅ Keras Functional | ❌ (not tested) |

### Metrics Collected

- **Fit Time**: Time to train the model (marked with "(Colab)" for Colab runs)
- **Predict Time**: Time to make predictions (marked with "(Colab)" for Colab runs)
- **MSE**: Mean Squared Error (comparable across all environments)
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination (comparable across all environments)

### Dataset

Default benchmark uses:
- 5,000 samples
- 20 features
- 80/20 train/test split
- Fixed random seed (42) for reproducibility
- Synthetic data with known ground truth

## Interpreting Results

The benchmark runner will display a formatted table comparing:

1. **Accuracy (MSE/R²)**: Fair comparison since all libraries use identical data
2. **Speed**: Time comparisons only meaningful within same environment
   - Local times: Your machine
   - (Colab) times: Google Colab hardware (different, not directly comparable)

## Troubleshooting

### "Colab results not found"

Run `benchmark_colab.ipynb` in Google Colab and download the result file.

### Missing JSON files

If the benchmark runner can't find `colab_benchmark.json`, it will still run local benchmarks but skip PyTorch/TensorFlow. Follow the Colab instructions above to add them.

## Customizing Benchmarks

Edit `benchmark_runner.py` to change:
- Dataset size (`n_samples`, `n_features`)
- Noise level
- Train/test split ratio
- Models to benchmark

## Example Output

```
====================================================================================================
BENCHMARK RESULTS
====================================================================================================
Dataset: 5000 samples, 20 features
Train size: 4000, Test size: 1000
----------------------------------------------------------------------------------------------------
Model                          Fit Time (s)       Predict (s)        MSE          R²
----------------------------------------------------------------------------------------------------

[LynxLearn]
LynxLearn_OLS                  0.004175           0.000050           0.266795     0.997593
LynxLearn_GD                   0.182553           0.000041           0.267215     0.997589
LynxLearn_Ridge                0.002463           0.000045           0.266837     0.997592

[Scikit-learn]
Sklearn_OLS                    0.006334           0.000311           0.266795     0.997593

[PyTorch]
PyTorch_LSTSQ                  0.001234 (Colab)   0.000123 (Colab)   0.266800     0.997590
PyTorch_SGD                    0.456789 (Colab)   0.000234 (Colab)   0.267500     0.997580

NOTE: Results marked with '(Colab)' were run on Google Colab hardware.
      Time comparisons between local and Colab are not directly comparable.
      Focus on MSE/R² for accuracy comparison, time for same-environment only.
====================================================================================================
```

## Fair Comparison Guarantee

When you use the Colab notebook:
1. **Same Data**: `colab_benchmark.json` ensures identical X_train, X_test, y_train, y_test
2. **Same Seed**: Random seed 42 used for all data generation
3. **Same Split**: 80/20 split with same indices
4. **Verifiable**: You can inspect `colab_benchmark.json` to verify the data

This means MSE and R² comparisons are completely fair across all libraries!
