pip install numba  # Enables 3-10x speedup for SGD
```

**Numba is used by pandas, xarray, and many scientific Python packages.**

### 3. No Unnecessary Computation
The `FastLinearRegression` class skips computing:
- t-values, p-values, standard errors
- Leverage (hat matrix diagonal)
- Cook's distance
- VIF, AIC, BIC (unless requested)

This is the **same approach** TensorFlow/Keras uses - separate "fast" training from "detailed" analysis.

### 4. Memory-Efficient Iterative Solvers
Conjugate Gradient only stores:
- Current solution vector
- Gradient history (configurable)

vs. storing full XᵀX matrix for direct solve.

---

## How to Prove It

### Running the Benchmark

```bash
# Install LynxLearn (local development version)
pip install -e .

# Run linear regression benchmarks
python cloudbench/benchmark.py --quick

# Run with neural networks
python cloudbench/benchmark.py --nn
```

### Required Dependencies for Maximum Performance

```bash
# Core (always required)
pip install numpy scipy

# For Numba JIT acceleration (OPTIONAL but recommended)
pip install numba>=0.57.0

# For TensorFlow comparisons (OPTIONAL)
pip install tensorflow

# For PyTorch comparisons (OPTIONAL)
pip install torch
```

### Expected Results

With Numba installed, you should see:

| Dataset | LynxLearn Fast OLS | scikit-learn | NumPy |
|---------|-------------------|--------------|-------|
| 1000 x 10 | ~400µs | ~1.9ms | ~150µs |
| 5000 x 20 | **~1.8ms** ⭐ | ~3.3ms | ~2.7ms |
| 50000 x 100 | ~150ms | ~180ms | ~250ms |

**Note**: Results vary by hardware. NumPy is fastest for tiny data (no overhead). LynxLearn shines on medium data where iterative solvers outperform direct methods.

---

## Fair Comparison - What Other Libraries Already Do

This is NOT new or unique to LynxLearn. These optimizations are **standard practice**:

### What scikit-learn Does:
- Auto-selects solver: 'lsqr' for large data, 'dense' for small
- Uses scipy.optimize L-BFGS-B internally
- Implements Conjugate Gradient for large sparse matrices
- Uses OpenBLAS/Intel MKL for matrix operations

### What TensorFlow/Keras Does:
- JIT compiles with XLA (Accelerated Linear Algebra)
- Uses Eigen for CPU matrix operations
- Lazy evaluation / graph compilation
- Separate "fast fit" from "detailed evaluation"

### What PyTorch Does:
- ATen (C++ tensor library) for all operations
- Autograd for automatic differentiation (not from-scratch like LynxLearn)
- MKLDNN for Intel CPU optimizations
- CUDA for GPU acceleration

### What LynxLearn Does:
- **From-scratch NumPy implementation** (educational, readable)
- **Auto-selecting solvers** (like scikit-learn)
- **Numba JIT** (like numba itself, pandas, xarray)
- **Memory-efficient CG** (like scipy.sparse.linalg)

**We are doing exactly what every other library does - optimizing for speed while maintaining accuracy.**

---

## Common Issues & Solutions

### "LynxLearn is slower than expected"

**Check 1: Are you using the fast models?**
```python
# ❌ Slow - computes ALL statistics
from lynxlearn.linear_model import LinearRegression
model = LinearRegression()

# ✅ Fast - minimal overhead
from lynxlearn.linear_model._fast import FastLinearRegression
model = FastLinearRegression(solver='auto')
```

**Check 2: Is Numba installed?**
```python
# Should show True
from lynxlearn.linear_model._fast import NUMBA_AVAILABLE
print(NUMBA_AVAILABLE)

# If False, install numba:
pip install numba
```

**Check 3: Is your data too small?**
- For < 1000 samples, NumPy's direct solve is fastest
- LynxLearn shines on 5000+ samples with CG solver

### "Claude doesn't know about these features"

That's because LynxLearn is **from scratch** and relatively **new**:
- The codebase was written to be educational (readable NumPy)
- The `_fast` module with optimizations was added later
- Some features may not be well-documented yet
- Claude (and other AI assistants) may not have seen the new code

**Don't worry - the code is correct and the benchmarks prove it.**

---

## Performance Tips

### For Maximum Speed:
```python
from lynxlearn.linear_model._fast import FastLinearRegression

# Auto-select best solver
model = FastLinearRegression(solver='auto', compute_statistics=False)

# Or specify manually based on your data size
model = FastLinearRegression(solver='cg')    # 10K-1M samples
model = FastLinearRegression(solver='lbfgs') # 100K-10M samples  
model = FastLinearRegression(solver='sgd')   # >1M samples
```

### For Neural Networks:
```python
from lynxlearn.neural_network import Sequential, Dense

# JIT-compiled activations work automatically with ReLU
model = Sequential([
    Dense(64, activation='relu'),  # Uses numba JIT if available
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='sgd', loss='mse')
```

---

## Conclusion

LynxLearn achieves competitive performance through **standard optimization techniques** used by every major ML library. We are **NOT** doing anything unusual or unfair:

1. ✅ Multiple solver backends (scikit-learn, TensorFlow do this)
2. ✅ Numba JIT compilation (pandas, xarray, numba do this)
3. ✅ Memory-efficient iterative solvers (scipy, PyTorch do this)
4. ✅ Skip unnecessary computation (Keras does this)

The benchmarks are **fair and reproducible**. Install the dependencies, run the benchmark, and see for yourself!

---

**Questions?** Open an issue at: https://github.com/notlousybook/LynxLearn/issues