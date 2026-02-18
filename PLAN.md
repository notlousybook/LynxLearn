# Neural Network Implementation Plan for LynxLearn

**Version:** 1.0  
**Author:** Senior Frontend Architect & Neural Network Engineer  
**Status:** PLANNING PHASE  
**Estimated Timeline:** 6 weeks  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Deep Reasoning Chain](#deep-reasoning-chain)
3. [Technical Architecture](#technical-architecture)
4. [API Design](#api-design)
5. [Edge Case Analysis](#edge-case-analysis)
6. [Implementation Phases](#implementation-phases)
7. [Performance Optimization](#performance-optimization)
8. [Testing Strategy](#testing-strategy)
9. [Documentation Plan](#documentation-plan)
10. [Benchmarking Strategy](#benchmarking-strategy)
11. [Risk Mitigation](#risk-mitigation)
12. [Success Metrics](#success-metrics)

---

## Executive Summary

### Objective

Implement a comprehensive neural network framework for LynxLearn that maintains the library's core philosophy: **educational, beginner-friendly, performant, and built from scratch using NumPy**.

### Key Goals

1. **Performance Parity**: Match or exceed TensorFlow training speeds while maintaining accuracy
2. **Educational Value**: Clear, readable code that teaches neural network concepts
3. **API Consistency**: Maintain LynxLearn's `train()`, `predict()`, `evaluate()` pattern
4. **Professional Capabilities**: Advanced optimizers, regularization, and diagnostics
5. **Extensibility**: Modular architecture supporting future enhancements

### Scope

**In Scope (Phase 1):**
- Feedforward neural networks (Dense/Fully Connected)
- Multiple optimizers (SGD, Adam, RMSprop, AdaGrad, AdamW)
- Comprehensive activation functions
- Regularization techniques (Dropout, Batch Norm, L1/L2)
- Classification and regression support
- Training diagnostics and visualization
- Model serialization

**Out of Scope (Future Phases):**
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs, LSTMs)
- Transformer architectures
- GPU acceleration (CUDA/OpenCL)

---

## Deep Reasoning Chain

### Multi-Dimensional Analysis

#### 1. Psychological Dimension (User Sentiment & Cognitive Load)

**User Motivation Analysis:**
- The request for "actual neural networks" indicates the user wants to expand beyond linear models
- They value the educational, from-scratch approach (core LynxLearn philosophy)
- Performance matters: the 200x faster than TensorFlow claim must be maintained
- They want to compete with major ML libraries while staying beginner-friendly

**Cognitive Load Design Principles:**

```
Principle 1: Progressive Disclosure
├── Basic API: Sequential model with sensible defaults
├── Intermediate: Custom layers, optimizers, callbacks
└── Advanced: Custom layer development, gradient inspection

Principle 2: Consistent Mental Models
├── Same API as linear models: train(), predict(), evaluate()
├── Consistent naming: weights, bias (not coef_, intercept_)
└── Similar error messages and debugging patterns

Principle 3: Educational Transparency
├── Clear variable names (not obscure abbreviations)
├── Comments explaining mathematical operations
├── Reference to papers/textbooks in docstrings
└── Visual learning aids in documentation
```

**Design Decision: Hybrid API Approach**

We provide TWO APIs with identical capabilities:

```python
# Beginner API (Sequential)
model = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Advanced API (Functional) - Future
input_layer = Input(shape=(784,))
x = Dense(128, activation='relu')(input_layer)
x = Dropout(0.2)(x)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output)
```

**Rationale:** Beginners start simple, power users have flexibility.

---

#### 2. Technical Dimension (Performance & Numerical Stability)

**Performance-Critical Architecture Decisions:**

```
Decision 1: Vectorized Operations Only
├── NO loops over samples (only over epochs/batches)
├── All operations on (batch_size, features) matrices
├── Use NumPy broadcasting and einsum for complex operations
└── Benchmark: Must match TensorFlow speed on CPU

Decision 2: Memory-Efficient Backpropagation
├── Store only necessary intermediate activations
├── Clear gradients after weight updates
├── Use in-place operations where safe
└── Support gradient checkpointing for large networks

Decision 3: Numerical Stability Guarantees
├── Log-sum-exp trick for softmax (prevent overflow)
├── Epsilon clipping in divisions (prevent division by zero)
├── Gradient clipping (prevent exploding gradients)
└── NaN detection with helpful error messages
```

**Backpropagation Implementation Strategy:**

```
Forward Pass:
┌─────────────────────────────────────────────────┐
│ Input X (batch_size, input_features)            │
│   ↓                                             │
│ Layer 1: Z1 = X @ W1 + b1                       │
│          A1 = activation(Z1)                    │
│   ↓ [Store Z1, A1 for backward pass]           │
│ Layer 2: Z2 = A1 @ W2 + b2                      │
│          A2 = activation(Z2)                    │
│   ↓ [Store Z2, A2]                             │
│ Output: Y_pred = A2                             │
│ Loss: L = loss_function(Y_true, Y_pred)        │
└─────────────────────────────────────────────────┘

Backward Pass:
┌─────────────────────────────────────────────────┐
│ dL/dA2 = loss_gradient(Y_true, Y_pred)         │
│   ↓                                             │
│ dL/dZ2 = dL/dA2 ⊙ activation'(Z2)               │
│ dL/dW2 = A1.T @ dL/dZ2                          │
│ dL/db2 = sum(dL/dZ2, axis=0)                    │
│   ↓                                             │
│ dL/dA1 = dL/dZ2 @ W2.T                          │
│ dL/dZ1 = dL/dA1 ⊙ activation'(Z1)               │
│ dL/dW1 = X.T @ dL/dZ1                           │
│ dL/db1 = sum(dL/dZ1, axis=0)                    │
└─────────────────────────────────────────────────┘
```

**Memory Optimization Techniques:**

1. **Lazy Gradient Computation**: Only compute gradients when needed
2. **Intermediate Clearing**: Delete Z values after computing dZ
3. **Batch-wise Processing**: Don't store all batch activations simultaneously
4. **Shared Buffers**: Reuse memory for temporary arrays

**Numerical Stability Implementation:**

```python
# Example: Stable Softmax Implementation
def softmax(x):
    """
    Numerically stable softmax.
    
    Trick: Subtract max(x) to prevent overflow in exp()
    """
    # Shift for numerical stability
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Example: Stable Log Softmax
def log_softmax(x):
    """
    Numerically stable log softmax.
    
    Uses log-sum-exp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_x = np.max(x, axis=1, keepdims=True)
    return x - max_x - np.log(np.sum(np.exp(x - max_x), axis=1, keepdims=True))

# Example: Stable Sigmoid
def sigmoid(x):
    """
    Numerically stable sigmoid.
    
    Separate handling for positive and negative values to prevent overflow.
    """
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    
    result = np.zeros_like(x)
    
    # For positive values: 1 / (1 + exp(-x))
    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    
    # For negative values: exp(x) / (1 + exp(x))
    exp_x = np.exp(x[neg_mask])
    result[neg_mask] = exp_x / (1 + exp_x)
    
    return result
```

---

#### 3. Accessibility Dimension (Learning & Documentation)

**Documentation Accessibility Framework:**

```
Layer 1: API Reference
├── Every public method has complete docstring
├── Type hints for all parameters and returns
├── Example usage in docstring
└── Common pitfalls and solutions

Layer 2: Conceptual Explanations
├── "How Neural Networks Learn" guide
├── "Understanding Backpropagation" tutorial
├── Visual diagrams of forward/backward pass
└── Mathematical derivations with intuition

Layer 3: Practical Examples
├── Beginner: MNIST classification (Hello World)
├── Intermediate: Regression with custom architecture
├── Advanced: Custom layer development
└── Expert: Optimizer implementation

Layer 4: Interactive Learning
├── Jupyter notebooks with step-by-step explanations
├── Visualization of training progress
├── Gradient flow animations
└── Interactive hyperparameter exploration
```

**Error Message Design Principles:**

```
Principle 1: Identify the Problem Clearly
❌ Bad: "ValueError: shapes not aligned"
✅ Good: "ShapeError: Layer 2 expects input shape (batch, 128) but received (batch, 64). 
         Did you mean to set Dense(units=128) instead of Dense(units=64)?"

Principle 2: Provide Actionable Solutions
❌ Bad: "NaN detected in gradients"
✅ Good: "NumericalError: NaN detected in gradients during training.
         Common causes:
         1. Learning rate too high (try reducing by 10x)
         2. Input data not normalized (try StandardScaler)
         3. Exploding gradients (try gradient clipping)
         
         To debug, try: model.train(..., debug=True)"

Principle 3: Educational Context
❌ Bad: "Gradient vanishing detected"
✅ Good: "Warning: Gradient Vanishing Detected
         Layer 1 gradients are < 1e-7, which may prevent learning.
         
         This often happens with:
         - Deep networks using sigmoid/tanh (try ReLU)
         - Improper weight initialization (try He initialization)
         
         Learn more: docs/troubleshooting.md#vanishing-gradients"
```

---

#### 4. Scalability Dimension (Modularity & Extensibility)

**Modular Architecture Design:**

```
lynxlearn/
├── neural_network/                    # NEW MODULE
│   ├── __init__.py
│   ├── _base.py                       # BaseNeuralNetwork, BaseLayer, BaseOptimizer
│   ├── _model.py                      # Sequential model
│   ├── layers/                        # Layer implementations
│   │   ├── __init__.py
│   │   ├── _base.py                   # BaseLayer class
│   │   ├── _dense.py                  # Dense/Fully Connected
│   │   ├── _activation.py             # Activation layer wrapper
│   │   ├── _dropout.py                # Dropout regularization
│   │   ├── _batch_norm.py             # Batch Normalization
│   │   └── _input.py                  # Input layer (Functional API)
│   ├── activations/                   # Activation functions
│   │   ├── __init__.py
│   │   ├── _base.py                   # BaseActivation class
│   │   ├── _relu.py                   # ReLU, LeakyReLU, PReLU
│   │   ├── _sigmoid.py                # Sigmoid
│   │   ├── _tanh.py                   # Tanh
│   │   ├── _softmax.py                # Softmax
│   │   ├── _elu.py                    # ELU, SELU
│   │   └── _swish.py                  # Swish, GELU
│   ├── optimizers/                    # Optimization algorithms
│   │   ├── __init__.py
│   │   ├── _base.py                   # BaseOptimizer class
│   │   ├── _sgd.py                    # SGD, Momentum, Nesterov
│   │   ├── _adam.py                   # Adam, AdamW
│   │   ├── _rmsprop.py                # RMSprop
│   │   └── _adagrad.py                # AdaGrad
│   ├── losses/                        # Loss functions
│   │   ├── __init__.py
│   │   ├── _base.py                   # BaseLoss class
│   │   ├── _mse.py                    # Mean Squared Error
│   │   ├── _mae.py                    # Mean Absolute Error
│   │   ├── _binary_crossentropy.py    # Binary Cross-Entropy
│   │   ├── _categorical_crossentropy.py # Categorical Cross-Entropy
│   │   ├── _sparse_crossentropy.py    # Sparse Categorical Cross-Entropy
│   │   └── _huber.py                  # Huber Loss
│   ├── callbacks/                     # Training callbacks
│   │   ├── __init__.py
│   │   ├── _base.py                   # BaseCallback class
│   │   ├── _early_stopping.py         # Early stopping
│   │   ├── _model_checkpoint.py       # Save best model
│   │   ├── _lr_scheduler.py           # Learning rate scheduling
│   │   └── _progress_bar.py           # Training progress display
│   ├── initializers/                  # Weight initialization
│   │   ├── __init__.py
│   │   ├── _base.py                   # BaseInitializer class
│   │   ├── _xavier.py                 # Xavier/Glorot initialization
│   │   ├── _he.py                     # He initialization
│   │   └── _lecun.py                  # LeCun initialization
│   ├── _utils.py                      # Neural network utilities
│   └── _serialization.py              # Save/load functionality
├── metrics/                           # EXISTING (extend for classification)
│   ├── _classification.py             # NEW: Accuracy, Precision, Recall, F1
│   └── ...
└── visualizations/                    # EXISTING (extend for neural networks)
    ├── _neural_net.py                 # NEW: Training curves, confusion matrix
    └── ...
```

**Extensibility Mechanisms:**

```python
# 1. Custom Layer Creation (Easy)
class CustomLayer(BaseLayer):
    def __init__(self, units):
        super().__init__()
        self.units = units
    
    def build(self, input_shape):
        # Initialize weights
        self.weights = np.random.randn(input_shape[-1], self.units)
        self.bias = np.zeros(self.units)
    
    def forward(self, x):
        # Forward computation
        return x @ self.weights + self.bias
    
    def backward(self, grad_output):
        # Backward computation
        self.grad_weights = self.input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        return grad_output @ self.weights.T

# 2. Custom Optimizer Creation
class CustomOptimizer(BaseOptimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
    
    def update(self, layer):
        # Custom update rule
        layer.weights -= self.learning_rate * layer.grad_weights
        layer.bias -= self.learning_rate * layer.grad_bias

# 3. Custom Loss Function Creation
class CustomLoss(BaseLoss):
    def compute(self, y_true, y_pred):
        # Custom loss computation
        return np.mean((y_true - y_pred) ** 2)
    
    def gradient(self, y_true, y_pred):
        # Gradient of loss
        return 2 * (y_pred - y_true) / y_true.size

# 4. Custom Callback Creation
class CustomCallback(BaseCallback):
    def on_epoch_end(self, epoch, logs=None):
        # Custom behavior after each epoch
        if logs['val_loss'] < 0.01:
            self.model.stop_training = True
```

---

## Technical Architecture

### Core Class Hierarchy

```
BaseModel (abstract)
├── train(X, y)           # Train the model
├── predict(X)            # Make predictions
├── evaluate(X, y)        # Evaluate performance
├── summary()             # Print model summary
└── get_params()          # Get model parameters

├── BaseRegressor (existing)
│   └── Linear models...
│
└── BaseNeuralNetwork (NEW)
    ├── compile(optimizer, loss, metrics)  # Configure model
    ├── fit(X, y)                          # Alias for train()
    ├── train(X, y, epochs, batch_size)    # Train with epochs
    ├── predict(X)                         # Predict
    ├── predict_proba(X)                   # Predict probabilities
    ├── evaluate(X, y)                     # Evaluate
    ├── summary()                          # Architecture summary
    ├── save(filepath)                     # Save model
    ├── load(filepath)                     # Load model (classmethod)
    ├── layers                             # List of layers
    ├── optimizer                          # Optimizer instance
    ├── loss                               # Loss function instance
    ├── history                            # Training history
    └── stop_training                      # Flag to stop training
    
    └── Sequential (NEW)
        ├── __init__(layers)               # Initialize with layer list
        ├── add(layer)                     # Add layer
        ├── compile(optimizer, loss, metrics)
        ├── train(X, y, epochs, batch_size, validation_data, callbacks)
        └── ... (inherited methods)
```

### Layer Architecture

```
BaseLayer (abstract)
├── __init__()                 # Initialize layer parameters
├── build(input_shape)         # Create weights (lazy initialization)
├── forward(x)                 # Forward pass
├── backward(grad_output)      # Backward pass
├── get_params()               # Get weights and biases
├── set_params(params)         # Set weights and biases
├── get_gradients()            # Get gradients
├── trainable                  # Boolean: can weights be updated?
├── input_shape                # Shape of input
├── output_shape               # Shape of output
└── training                   # Boolean: training or inference mode?

├── Dense
│   ├── units                  # Number of neurons
│   ├── activation             # Activation function (optional)
│   ├── use_bias               # Whether to use bias
│   ├── kernel_initializer     # Weight initialization
│   ├── bias_initializer       # Bias initialization
│   ├── kernel_regularizer     # Weight regularization
│   └── bias_regularizer       # Bias regularization
│
├── Activation
│   └── activation             # Activation function
│
├── Dropout
│   ├── rate                   # Dropout rate
│   └── seed                   # Random seed
│
├── BatchNormalization
│   ├── momentum               # Momentum for running statistics
│   ├── epsilon                # Small constant for numerical stability
│   ├── center                 # Whether to use beta offset
│   ├── scale                  # Whether to use gamma scaling
│   └── training               # Training vs inference mode
│
└── Input (Functional API)
    └── shape                  # Input shape
```

### Optimizer Architecture

```
BaseOptimizer (abstract)
├── __init__(learning_rate)    # Initialize optimizer
├── update(layer)              # Update layer parameters
├── get_config()               # Get optimizer configuration
├── set_config(config)         # Set optimizer configuration
├── get_state()                # Get optimizer state (momentum, etc.)
├── set_state(state)           # Set optimizer state
└── lr                         # Learning rate

├── SGD
│   ├── momentum               # Momentum coefficient
│   ├── nesterov               # Whether to use Nesterov momentum
│   └── velocities             # Velocity terms (state)
│
├── Adam
│   ├── beta_1                 # Exponential decay rate for 1st moment
│   ├── beta_2                 # Exponential decay rate for 2nd moment
│   ├── epsilon                # Small constant for numerical stability
│   ├── m                      # 1st moment estimates (state)
│   └── v                      # 2nd moment estimates (state)
│
├── AdamW
│   ├── (same as Adam)
│   └── weight_decay           # Weight decay coefficient
│
├── RMSprop
│   ├── rho                    # Decay rate
│   ├── epsilon                # Small constant
│   └── cache                  # Accumulated squared gradients (state)
│
└── AdaGrad
    ├── epsilon                # Small constant
    └── cache                  # Accumulated squared gradients (state)
```

### Loss Function Architecture

```
BaseLoss (abstract)
├── __init__()                 # Initialize loss
├── compute(y_true, y_pred)    # Compute loss value
├── gradient(y_true, y_pred)   # Compute gradient of loss
└── name                       # Loss function name

├── MeanSquaredError
│   ├── compute(y_true, y_pred)    # MSE = mean((y_true - y_pred)^2)
│   └── gradient(y_true, y_pred)   # d(MSE)/dy_pred = 2*(y_pred - y_true)/n
│
├── MeanAbsoluteError
│   ├── compute(y_true, y_pred)    # MAE = mean(|y_true - y_pred|)
│   └── gradient(y_true, y_pred)   # d(MAE)/dy_pred = sign(y_pred - y_true)/n
│
├── BinaryCrossEntropy
│   ├── from_logits                # Whether inputs are logits or probabilities
│   ├── compute(y_true, y_pred)    # BCE = -mean(y_true*log(y_pred) + (1-y_true)*log(1-y_pred))
│   └── gradient(y_true, y_pred)   # Gradient w.r.t. y_pred
│
├── CategoricalCrossEntropy
│   ├── from_logits                # Whether inputs are logits or probabilities
│   ├── compute(y_true, y_pred)    # CCE = -sum(y_true * log(y_pred))
│   └── gradient(y_true, y_pred)   # Gradient w.r.t. y_pred
│
├── SparseCategoricalCrossEntropy
│   ├── from_logits                # Whether inputs are logits or probabilities
│   ├── compute(y_true, y_pred)    # SCCE for integer labels
│   └── gradient(y_true, y_pred)   # Gradient w.r.t. y_pred
│
└── HuberLoss
    ├── delta                      # Threshold for quadratic/linear transition
    ├── compute(y_true, y_pred)    # Huber loss formula
    └── gradient(y_true, y_pred)   # Gradient w.r.t. y_pred
```

### Callback Architecture

```
BaseCallback (abstract)
├── on_train_begin(logs)           # Called at the start of training
├── on_train_end(logs)             # Called at the end of training
├── on_epoch_begin(epoch, logs)    # Called at the start of each epoch
├── on_epoch_end(epoch, logs)      # Called at the end of each epoch
├── on_batch_begin(batch, logs)    # Called at the start of each batch
├── on_batch_end(batch, logs)      # Called at the end of each batch
└── model                          # Reference to the model

├── EarlyStopping
│   ├── monitor                    # Metric to monitor (e.g., 'val_loss')
│   ├── patience                   # Number of epochs with no improvement
│   ├── min_delta                  # Minimum change to qualify as improvement
│   ├── mode                       # 'min', 'max', or 'auto'
│   └── restore_best_weights       # Whether to restore best weights
│
├── ModelCheckpoint
│   ├── filepath                   # Path to save model
│   ├── monitor                    # Metric to monitor
│   ├── save_best_only             # Only save if monitored metric improves
│   └── mode                       # 'min', 'max', or 'auto'
│
├── LearningRateScheduler
│   ├── schedule                   # Function: epoch -> learning_rate
│   └── verbose                    # Verbosity mode
│
├── ReduceLROnPlateau
│   ├── monitor                    # Metric to monitor
│   ├── factor                     # Factor by which LR is reduced
│   ├── patience                   # Number of epochs with no improvement
│   └── min_lr                     # Lower bound on learning rate
│
└── ProgressBar
    ├── verbose                    # Verbosity level (0, 1, 2)
    └── show_metrics               # Metrics to display
```

---

## API Design

### Beginner API (Sequential Model)

**Example 1: Classification (MNIST)**

```python
from lynxlearn.neural_network import Sequential, Dense, Dropout
from lynxlearn.neural_network.optimizers import Adam
from lynxlearn.neural_network.losses import CategoricalCrossEntropy
from lynxlearn.metrics import accuracy_score

# Step 1: Create model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Step 2: Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CategoricalCrossEntropy(),
    metrics=['accuracy']
)

# Step 3: View architecture
model.summary()

# Output:
# ┌─────────────────────────────────────────────────────────────┐
# │ Model: Sequential                                           │
# ├─────────────────────────────────────────────────────────────┤
# │ Layer (type)               Output Shape          Param #    │
# ├─────────────────────────────────────────────────────────────┤
# │ dense_1 (Dense)            (None, 128)           100,480    │
# │ dropout_1 (Dropout)        (None, 128)           0          │
# │ dense_2 (Dense)            (None, 64)            8,256      │
# │ dropout_2 (Dropout)        (None, 64)            0          │
# │ dense_3 (Dense)            (None, 10)            650        │
# ├─────────────────────────────────────────────────────────────┤
# │ Total params: 109,386                                       │
# │ Trainable params: 109,386                                   │
# │ Non-trainable params: 0                                     │
# └─────────────────────────────────────────────────────────────┘

# Step 4: Train model
history = model.train(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# Step 5: Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Step 6: Predict
predictions = model.predict(X_test)
predicted_classes = model.predict_classes(X_test)
probabilities = model.predict_proba(X_test)

# Step 7: Save/Load
model.save('mnist_model.json')
loaded_model = Sequential.load('mnist_model.json')
```

**Example 2: Regression**

```python
from lynxlearn.neural_network import Sequential, Dense
from lynxlearn.neural_network.optimizers import SGD
from lynxlearn.neural_network.losses import MeanSquaredError
from lynxlearn import metrics

# Create model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1)  # No activation for regression
])

# Compile
model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    loss=MeanSquaredError(),
    metrics=['mse', 'mae']
)

# Train
history = model.train(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2
)

# Evaluate
mse, mae, r2 = model.evaluate(X_test, y_test, metrics=['mse', 'mae', 'r2'])

# Predict
predictions = model.predict(X_test)
```

**Example 3: With Callbacks**

```python
from lynxlearn.neural_network import Sequential, Dense
from lynxlearn.neural_network.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    LearningRateScheduler
)

# Define learning rate schedule
def lr_schedule(epoch):
    if epoch < 10:
        return 0.001
    elif epoch < 20:
        return 0.0005
    else:
        return 0.0001

# Create callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.json', monitor='val_loss', save_best_only=True),
    LearningRateScheduler(lr_schedule, verbose=1)
]

# Train with callbacks
history = model.train(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)
```

### Advanced API (Custom Components)

**Example 4: Custom Layer**

```python
from lynxlearn.neural_network.layers import BaseLayer
import numpy as np

class CustomDense(BaseLayer):
    """
    A custom dense layer with some special behavior.
    """
    
    def __init__(self, units, activation=None, name=None):
        super().__init__(name=name)
        self.units = units
        self.activation = activation
    
    def build(self, input_shape):
        """Initialize weights when input shape is known."""
        input_dim = input_shape[-1]
        
        # He initialization for ReLU
        self.weights = np.random.randn(input_dim, self.units) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros(self.units)
        
        # Initialize gradients
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        
        self.built = True
    
    def forward(self, x):
        """Forward pass: output = x @ W + b"""
        self.input = x  # Store for backward pass
        self.z = x @ self.weights + self.bias
        
        if self.activation is not None:
            self.output = self.activation.forward(self.z)
        else:
            self.output = self.z
        
        return self.output
    
    def backward(self, grad_output):
        """Backward pass: compute gradients"""
        # Gradient through activation
        if self.activation is not None:
            grad_output = self.activation.backward(grad_output, self.z)
        
        # Compute gradients
        self.grad_weights = self.input.T @ grad_output / self.input.shape[0]
        self.grad_bias = np.mean(grad_output, axis=0)
        
        # Gradient for previous layer
        grad_input = grad_output @ self.weights.T
        
        return grad_input
    
    def get_params(self):
        """Get layer parameters"""
        return {'weights': self.weights, 'bias': self.bias}
    
    def set_params(self, params):
        """Set layer parameters"""
        self.weights = params['weights']
        self.bias = params['bias']
    
    def get_gradients(self):
        """Get parameter gradients"""
        return {'weights': self.grad_weights, 'bias': self.grad_bias}

# Use custom layer
model = Sequential([
    CustomDense(128, activation='relu'),
    CustomDense(10, activation='softmax')
])
```

**Example 5: Custom Optimizer**

```python
from lynxlearn.neural_network.optimizers import BaseOptimizer
import numpy as np

class NovoGrad(BaseOptimizer):
    """
    NovoGrad optimizer: SGD with normalized gradients.
    """
    
    def __init__(self, learning_rate=0.01, beta=0.95, epsilon=1e-8):
        super().__init__(learning_rate=learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.v = {}  # Accumulated squared gradients
    
    def update(self, layer):
        """Update layer parameters using NovoGrad rule"""
        layer_id = id(layer)
        grads = layer.get_gradients()
        params = layer.get_params()
        
        # Initialize state if needed
        if layer_id not in self.v:
            self.v[layer_id] = {}
            for key in grads:
                self.v[layer_id][key] = np.zeros_like(grads[key])
        
        # Update each parameter
        for key in grads:
            grad = grads[key]
            
            # Update accumulator: v = beta * v + (1 - beta) * grad^2
            self.v[layer_id][key] = self.beta * self.v[layer_id][key] + (1 - self.beta) * grad**2
            
            # Normalized gradient update
            normalized_grad = grad / (np.sqrt(self.v[layer_id][key]) + self.epsilon)
            
            # Update parameter
            params[key] -= self.learning_rate * normalized_grad
        
        layer.set_params(params)
    
    def get_state(self):
        """Get optimizer state for serialization"""
        return {'v': self.v, 'lr': self.learning_rate}
    
    def set_state(self, state):
        """Restore optimizer state"""
        self.v = state['v']
        self.learning_rate = state['lr']

# Use custom optimizer
model.compile(optimizer=NovoGrad(learning_rate=0.001), loss='mse')
```

### API Consistency with Existing LynxLearn

**Method Naming:**

| Existing Linear Models | Neural Networks | Description |
|------------------------|-----------------|-------------|
| `train(X, y)` | `train(X, y, epochs, batch_size, ...)` | Train model |
| `fit(X, y)` | `fit(X, y, ...)` | Alias for train() |
| `predict(X)` | `predict(X)` | Make predictions |
| - | `predict_proba(X)` | Predict probabilities |
| - | `predict_classes(X)` | Predict class labels |
| `evaluate(X, y)` | `evaluate(X, y)` | Evaluate model |
| `score(X, y)` | `score(X, y)` | Alias for evaluate() |
| `summary()` | `summary()` | Print summary |
| `get_params()` | `get_params()` | Get parameters |
| - | `get_weights()` | Get all layer weights |
| - | `set_weights(weights)` | Set all layer weights |
| - | `save(filepath)` | Save model to file |
| - | `load(filepath)` | Load model from file |
| - | `compile(...)` | Configure model |

**Attribute Naming:**

| Existing | Neural Networks | Description |
|----------|-----------------|-------------|
| `weights` | `weights` / `W` | Model weights |
| `bias` | `bias` / `b` | Model bias |
| `coef_` | - | Scikit-learn alias (linear only) |
| `intercept_` | - | Scikit-learn alias (linear only) |
| - | `layers` | List of layers |
| - | `optimizer` | Optimizer instance |
| - | `loss` | Loss function instance |
| - | `history` | Training history |
| - | `stop_training` | Stop training flag |

---

## Edge Case Analysis

### Edge Case 1: Vanishing Gradients

**Problem:**
Deep networks with sigmoid/tanh activations experience exponentially decreasing gradients, preventing early layers from learning.

**Detection:**
```python
def detect_vanishing_gradients(model, threshold=1e-7):
    """Check if gradients are vanishing."""
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'grad_weights'):
            grad_norm = np.linalg.norm(layer.grad_weights)
            if grad_norm < threshold:
                print(f"Warning: Layer {i} gradient norm = {grad_norm:.2e} (vanishing)")
                return True
    return False
```

**Prevention Strategies:**

1. **Better Activations:**
```python
# Instead of Sigmoid/Tanh
model = Sequential([
    Dense(128, activation='sigmoid'),  # ❌ Bad for deep networks
])

# Use ReLU variants
model = Sequential([
    Dense(128, activation='relu'),     # ✅ Good
    Dense(64, activation='leaky_relu'), # ✅ Better
])
```

2. **Proper Initialization:**
```python
from lynxlearn.neural_network.initializers import HeInitializer

# He initialization for ReLU
model = Sequential([
    Dense(128, activation='relu', kernel_initializer=HeInitializer()),
])
```

3. **Batch Normalization:**
```python
from lynxlearn.neural_network import BatchNormalization

model = Sequential([
    Dense(128),
    BatchNormalization(),  # Normalizes activations
    Activation('relu'),
])
```

**Automatic Warnings:**
```python
# During training, automatically warn users
if epoch > 5 and detect_vanishing_gradients(model):
    print("\n" + "="*70)
    print("⚠️  WARNING: Vanishing Gradients Detected")
    print("="*70)
    print("Layer 1 gradients are < 1e-7, which may prevent learning.")
    print("\nSuggested fixes:")
    print("  1. Use ReLU/LeakyReLU instead of sigmoid/tanh")
    print("  2. Add BatchNormalization layers")
    print("  3. Use He initialization for ReLU")
    print("  4. Reduce network depth")
    print("\nLearn more: docs/troubleshooting.md#vanishing-gradients")
    print("="*70)
```

---

### Edge Case 2: Exploding Gradients

**Problem:**
Very large gradients cause weight updates that lead to NaN values and training collapse.

**Detection:**
```python
def detect_exploding_gradients(model, threshold=1e3):
    """Check if gradients are exploding."""
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'grad_weights'):
            grad_norm = np.linalg.norm(layer.grad_weights)
            if grad_norm > threshold:
                print(f"Warning: Layer {i} gradient norm = {grad_norm:.2e} (exploding)")
                return True
    return False

def detect_nan(model):
    """Check for NaN in weights or gradients."""
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'weights'):
            if np.any(np.isnan(layer.weights)):
                return True
    return False
```

**Prevention Strategies:**

1. **Gradient Clipping:**
```python
# Clip by norm
model.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss='categorical_crossentropy'
)

# Clip by value
model.compile(
    optimizer=Adam(learning_rate=0.001, clipvalue=0.5),
    loss='categorical_crossentropy'
)
```

2. **Lower Learning Rate:**
```python
# If gradients explode, reduce learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # 10x lower
    loss='categorical_crossentropy'
)
```

3. **Better Initialization:**
```python
# Xavier for sigmoid/tanh, He for ReLU
from lynxlearn.neural_network.initializers import XavierInitializer

model = Sequential([
    Dense(128, activation='tanh', kernel_initializer=XavierInitializer()),
])
```

**Automatic Recovery:**
```python
# During training, if NaN detected
if detect_nan(model):
    print("\n" + "="*70)
    print("🚨 CRITICAL: NaN Detected in Weights")
    print("="*70)
    print("Training has become numerically unstable.")
    print("\nCommon causes:")
    print("  1. Learning rate too high")
    print("  2. Input data not normalized")
    print("  3. Exploding gradients")
    print("\nRecovery options:")
    print("  - Restart with lower learning rate (10x lower)")
    print("  - Normalize input data")
    print("  - Enable gradient clipping")
    print("="*70)
    
    # Option: Auto-recover by reducing LR and restarting
    # model.optimizer.learning_rate *= 0.1
    # model.restore_best_weights()
```

---

### Edge Case 3: Dying ReLU

**Problem:**
ReLU neurons output 0 for all inputs, causing them to "die" and stop learning.

**Detection:**
```python
def detect_dying_relu(model, X_sample, threshold=0.9):
    """
    Detect if ReLU neurons are dying.
    
    A neuron is "dying" if it outputs 0 for > threshold fraction of samples.
    """
    dying_neurons = []
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'activation') and layer.activation.name == 'relu':
            # Forward pass to get activations
            activations = layer.forward(X_sample)
            
            # Check fraction of zeros
            zero_fraction = np.mean(activations == 0)
            
            if zero_fraction > threshold:
                dying_count = np.sum(np.all(activations == 0, axis=0))
                dying_neurons.append({
                    'layer': i,
                    'dying_fraction': zero_fraction,
                    'dying_count': dying_count
                })
    
    return dying_neurons
```

**Prevention Strategies:**

1. **Use LeakyReLU:**
```python
from lynxlearn.neural_network.layers import LeakyReLU

model = Sequential([
    Dense(128),
    LeakyReLU(alpha=0.01),  # Small gradient for negative inputs
])
```

2. **Lower Learning Rate:**
```python
# High LR can push ReLU into negative region permanently
model.compile(optimizer=Adam(learning_rate=0.0001))
```

3. **Better Initialization:**
```python
# He initialization for ReLU
model = Sequential([
    Dense(128, activation='relu', kernel_initializer='he_normal'),
])
```

**Automatic Monitoring:**
```python
# During training, periodically check for dying ReLU
if epoch % 10 == 0:
    dying = detect_dying_relu(model, X_train[:1000])
    if dying:
        print(f"\n⚠️  Warning: {len(dying)} layers have dying ReLU neurons")
        for info in dying:
            print(f"  Layer {info['layer']}: {info['dying_count']} neurons dying "
                  f"({info['dying_fraction']*100:.1f}% zeros)")
```

---

### Edge Case 4: Imbalanced Classes

**Problem:**
Classification with highly imbalanced classes leads to biased models.

**Detection:**
```python
def check_class_balance(y):
    """Check if classes are imbalanced."""
    from collections import Counter
    
    counts = Counter(y)
    n_classes = len(counts)
    total = len(y)
    
    imbalance_ratios = {}
    for cls, count in counts.items():
        imbalance_ratios[cls] = count / total
    
    # Check if any class is < 10% of majority class
    max_ratio = max(imbalance_ratios.values())
    min_ratio = min(imbalance_ratios.values())
    
    if min_ratio < 0.1 * max_ratio:
        return True, imbalance_ratios
    
    return False, imbalance_ratios
```

**Solutions:**

1. **Class Weights:**
```python
# Automatically compute class weights
from lynxlearn.utils import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Use in training
model.train(X_train, y_train, class_weight=class_weight_dict)
```

2. **Sample Weights:**
```python
# Custom sample weights
sample_weights = compute_sample_weight('balanced', y_train)
model.train(X_train, y_train, sample_weight=sample_weights)
```

3. **Balanced Metrics:**
```python
from lynxlearn.metrics import balanced_accuracy_score, f1_score

# Use metrics that handle imbalance
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', balanced_accuracy_score, f1_score]
)
```

---

### Edge Case 5: Small Batch Sizes

**Problem:**
Very small batch sizes (1-4) lead to noisy gradients and unstable training.

**Detection & Warning:**
```python
def validate_batch_size(batch_size, X_train):
    """Validate batch size and provide recommendations."""
    n_samples = X_train.shape[0]
    
    if batch_size < 16:
        print("\n" + "="*70)
        print("⚠️  Warning: Small Batch Size")
        print("="*70)
        print(f"Batch size {batch_size} is very small, which can cause:")
        print("  - Noisy gradient estimates")
        print("  - Unstable batch normalization statistics")
        print("  - Slower training (less efficient vectorization)")
        print(f"\nRecommended: batch_size >= 32 (you have {n_samples} samples)")
        print("="*70)
    
    if batch_size > n_samples:
        raise ValueError(
            f"Batch size ({batch_size}) cannot be larger than "
            f"number of training samples ({n_samples})"
        )
    
    # Check if batch size doesn't divide dataset evenly
    if n_samples % batch_size != 0:
        last_batch_size = n_samples % batch_size
        if last_batch_size < batch_size // 2:
            print(f"\nNote: Last batch will have only {last_batch_size} samples")
```

**Solutions:**

1. **Drop Last Batch:**
```python
model.train(X_train, y_train, batch_size=32, drop_last_batch=True)
```

2. **Batch Normalization Handling:**
```python
# For small batches, use larger momentum for running statistics
BatchNormalization(momentum=0.99)  # Higher momentum for stability
```

---

### Edge Case 6: NaN Loss Values

**Problem:**
Numerical instability causes loss to become NaN.

**Detection:**
```python
def detect_nan_loss(loss_value, epoch, batch):
    """Detect NaN loss and provide helpful error message."""
    if np.isnan(loss_value) or np.isinf(loss_value):
        raise ValueError(
            f"\n{'='*70}\n"
            f"🚨 NUMERICAL ERROR: Loss became NaN/Inf at epoch {epoch}, batch {batch}\n"
            f"{'='*70}\n"
            f"Common causes:\n"
            f"  1. Learning rate too high\n"
            f"     → Try: model.compile(optimizer=Adam(learning_rate=0.0001))\n"
            f"  2. Input data contains NaN/Inf\n"
            f"     → Check: np.any(np.isnan(X_train)) or np.any(np.isinf(X_train))\n"
            f"  3. Labels contain invalid values\n"
            f"     → For classification: labels should be 0 to n_classes-1\n"
            f"     → For regression: check for extreme outliers\n"
            f"  4. Numerical overflow in softmax/cross-entropy\n"
            f"     → Try: model.compile(loss='categorical_crossentropy', from_logits=True)\n"
            f"  5. Improper weight initialization\n"
            f"     → Try: Dense(128, kernel_initializer='he_normal')\n"
            f"\n"
            f"Debug mode: model.train(..., debug=True)\n"
            f"{'='*70}"
        )
```

**Prevention:**

```python
# 1. Use logits for numerical stability
model.compile(
    optimizer='adam',
    loss=CategoricalCrossEntropy(from_logits=True)  # More stable
)

# 2. Gradient clipping
model.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss='categorical_crossentropy'
)

# 3. Input normalization
from lynxlearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Debug mode
model.train(X_train, y_train, debug=True)  # Extra checks and logging
```

---

### Edge Case 7: Memory Issues

**Problem:**
Large networks or datasets cause memory errors.

**Detection & Estimation:**
```python
def estimate_memory_usage(model, batch_size):
    """
    Estimate GPU/CPU memory needed for training.
    
    Returns estimated memory in MB.
    """
    # Parameters
    param_count = model.count_params()
    param_memory = param_count * 8 / (1024**2)  # float64
    
    # Gradients (same size as params)
    grad_memory = param_memory
    
    # Optimizer state (Adam: 2x params)
    optimizer_memory = 2 * param_memory if isinstance(model.optimizer, Adam) else 0
    
    # Intermediate activations (rough estimate)
    activation_memory = 0
    for layer in model.layers:
        if hasattr(layer, 'units'):
            activation_memory += batch_size * layer.units * 8 / (1024**2)
    
    total_memory = param_memory + grad_memory + optimizer_memory + activation_memory
    
    print(f"\n{'='*70}")
    print("Memory Estimation")
    print(f"{'='*70}")
    print(f"Parameters:      {param_count:,} ({param_memory:.2f} MB)")
    print(f"Gradients:       {grad_memory:.2f} MB")
    print(f"Optimizer state: {optimizer_memory:.2f} MB")
    print(f"Activations:     {activation_memory:.2f} MB")
    print(f"{'─'*70}")
    print(f"Total estimated: {total_memory:.2f} MB")
    print(f"{'='*70}")
    
    return total_memory
```

**Solutions:**

1. **Gradient Checkpointing:**
```python
# Trade computation for memory
model = Sequential([
    Dense(512, activation='relu'),
    Dense(512, activation='relu', checkpoint=True),  # Recompute in backward
    Dense(10)
])
```

2. **Smaller Batch Size:**
```python
model.train(X_train, y_train, batch_size=16)  # Reduce from 32
```

3. **Mixed Precision (Future):**
```python
# Use float32 instead of float64
model.compile(..., dtype='float32')
```

---

### Edge Case 8: Incorrect Input Shapes

**Problem:**
Users provide wrong input dimensions, leading to cryptic errors.

**Solution: Clear Error Messages**

```python
def validate_input_shape(X, y, model):
    """Validate input shapes with helpful error messages."""
    
    # Check X shape
    if X.ndim != 2:
        raise ValueError(
            f"\n{'='*70}\n"
            f"Input Shape Error\n"
            f"{'='*70}\n"
            f"Expected X to be 2D array with shape (n_samples, n_features)\n"
            f"Received X with shape {X.shape} ({X.ndim}D)\n"
            f"\n"
            f"Fix:\n"
            f"  • If X has shape ({X.shape[0]},), reshape it:\n"
            f"    X = X.reshape(-1, 1)\n"
            f"  • If X has shape ({X.shape[0]}, {X.shape[1]}, {X.shape[2]}), you may need to flatten:\n"
            f"    X = X.reshape(X.shape[0], -1)\n"
            f"{'='*70}"
        )
    
    # Check if matches model input shape
    if model.layers[0].input_shape is not None:
        expected_features = model.layers[0].input_shape[-1]
        actual_features = X.shape[-1]
        
        if expected_features != actual_features:
            raise ValueError(
                f"\n{'='*70}\n"
                f"Input Shape Mismatch\n"
                f"{'='*70}\n"
                f"Model expects {expected_features} features, but got {actual_features}\n"
                f"\n"
                f"Layer 1: {model.layers[0]}\n"
                f"  Expected input shape: (batch_size, {expected_features})\n"
                f"  Actual input shape:    (batch_size, {actual_features})\n"
                f"\n"
                f"Fix:\n"
                f"  • Check your data preprocessing\n"
                f"  • Change first layer: Dense(units=..., input_shape=({actual_features},))\n"
                f"{'='*70}"
            )
```

---

### Edge Case 9: Mixed Data Types

**Problem:**
Integer vs float inputs, different dtypes causing precision issues.

**Solution: Automatic Conversion with Warnings**

```python
def ensure_float64(arr, name='array'):
    """Convert array to float64 with helpful warnings."""
    original_dtype = arr.dtype
    
    if arr.dtype == np.int64 or arr.dtype == np.int32:
        print(f"\nℹ️  Note: Converting {name} from {arr.dtype} to float64")
        arr = arr.astype(np.float64)
    
    elif arr.dtype == np.float32:
        print(f"\n⚠️  Warning: {name} has dtype float32, converting to float64")
        print("    This may cause minor precision differences.")
        arr = arr.astype(np.float64)
    
    elif arr.dtype != np.float64:
        print(f"\n⚠️  Warning: Converting {name} from {arr.dtype} to float64")
        arr = arr.astype(np.float64)
    
    return arr
```

---

### Edge Case 10: Overfitting

**Problem:**
Training loss decreases but validation loss increases.

**Detection:**
```python
def detect_overfitting(history, patience=3):
    """
    Detect overfitting by comparing training and validation loss.
    
    Returns True if validation loss has been increasing for 'patience' epochs.
    """
    if 'val_loss' not in history:
        return False
    
    val_losses = history['val_loss']
    
    if len(val_losses) < patience + 1:
        return False
    
    # Check if last 'patience' validation losses are increasing
    recent_losses = val_losses[-(patience+1):]
    
    # Check if monotonically increasing
    increasing = all(recent_losses[i] < recent_losses[i+1] for i in range(patience))
    
    # Check if training loss is still decreasing
    train_losses = history['loss'][-(patience+1):]
    decreasing = all(train_losses[i] > train_losses[i+1] for i in range(patience))
    
    return increasing and decreasing
```

**Solutions:**

```python
# 1. Early Stopping
from lynxlearn.neural_network.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.train(X_train, y_train, 
            validation_split=0.2,
            callbacks=[early_stop])

# 2. Dropout
model = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.5),  # Drop 50% of neurons
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# 3. L2 Regularization
from lynxlearn.neural_network.regularizers import L2

model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=L2(0.01)),
    Dense(10, activation='softmax')
])

# 4. Data Augmentation (for images)
# (Future implementation)

# 5. Reduce Model Complexity
model = Sequential([
    Dense(64, activation='relu'),   # Reduced from 128
    Dense(32, activation='relu'),   # Reduced from 64
    Dense(10, activation='softmax')
])
```

---

## Implementation Phases

### Phase 1: Core Foundation (Week 1, Days 1-5)

**Objective:** Establish the fundamental neural network infrastructure.

#### Day 1-2: Base Classes and Infrastructure

**Files to Create:**
- `lynxlearn/neural_network/__init__.py`
- `lynxlearn/neural_network/_base.py`
- `lynxlearn/neural_network/layers/__init__.py`
- `lynxlearn/neural_network/layers/_base.py`

**Tasks:**
1. Create `BaseNeuralNetwork` class
   - Implement `train()`, `predict()`, `evaluate()`, `summary()`
   - Add `compile()` method
   - Add `history` tracking

2. Create `BaseLayer` class
   - Define abstract methods: `forward()`, `backward()`, `build()`
   - Add parameter management methods
   - Add training/inference mode switching

3. Create `BaseOptimizer` class
   - Define `update()` method
   - Add state management

4. Create `BaseLoss` class
   - Define `compute()` and `gradient()` methods

**Code Example:**

```python
# lynxlearn/neural_network/_base.py

class BaseNeuralNetwork:
    """Base class for all neural network models in LynxLearn."""
    
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss = None
        self.metrics = []
        self.history = None
        self.stop_training = False
        self._is_compiled = False
        self._is_trained = False
    
    def compile(self, optimizer, loss, metrics=None, **kwargs):
        """
        Configure the model for training.
        
        Parameters
        ----------
        optimizer : str or BaseOptimizer
            Optimizer instance or string identifier ('adam', 'sgd', etc.)
        loss : str or BaseLoss
            Loss function instance or string identifier
        metrics : list
            List of metrics to track during training
        """
        # Handle string identifiers
        if isinstance(optimizer, str):
            optimizer = self._get_optimizer(optimizer)
        if isinstance(loss, str):
            loss = self._get_loss(loss)
        
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or []
        self._is_compiled = True
    
    def train(self, X, y, epochs=100, batch_size=32, 
              validation_data=None, validation_split=0.0,
              callbacks=None, verbose=1, **kwargs):
        """
        Train the neural network.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values
        epochs : int
            Number of epochs to train
        batch_size : int
            Number of samples per gradient update
        validation_data : tuple (X_val, y_val)
            Validation data
        validation_split : float
            Fraction of training data to use for validation
        callbacks : list
            List of Callback instances
        verbose : int
            Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
        
        Returns
        -------
        history : dict
            Training history with loss and metrics per epoch
        """
        # Implementation in Phase 6
        pass
    
    # Alias for compatibility
    def fit(self, X, y, **kwargs):
        """Alias for train()."""
        return self.train(X, y, **kwargs)
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        predictions : ndarray
            Predicted values
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained first! Call model.train(X, y)")
        
        X = np.asarray(X, dtype=np.float64)
        return self._forward_pass(X, training=False)
    
    def predict_proba(self, X):
        """
        Predict class probabilities (for classification).
        
        Parameters
        ----------
        X : array-like
            Input data
            
        Returns
        -------
        probabilities : ndarray
            Class probabilities
        """
        return self.predict(X)
    
    def predict_classes(self, X):
        """
        Predict class labels (for classification).
        
        Parameters
        ----------
        X : array-like
            Input data
            
        Returns
        -------
        classes : ndarray
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def evaluate(self, X, y, metrics=None):
        """
        Evaluate the model on test data.
        
        Parameters
        ----------
        X : array-like
            Test data
        y : array-like
            True labels
        metrics : list
            Additional metrics to compute
            
        Returns
        -------
        results : dict or float
            Loss and metric values
        """
        y_pred = self.predict(X)
        loss_value = self.loss.compute(y, y_pred)
        
        if metrics:
            results = {'loss': loss_value}
            for metric in metrics:
                results[metric.__name__] = metric(y, y_pred)
            return results
        
        return loss_value
    
    def summary(self):
        """Print a summary of the model architecture."""
        self._print_summary()
    
    def _forward_pass(self, X, training=True):
        """Execute forward pass through all layers."""
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output
    
    def _backward_pass(self, grad):
        """Execute backward pass through all layers."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def _get_optimizer(self, name):
        """Get optimizer instance from string name."""
        optimizers = {
            'sgd': SGD,
            'adam': Adam,
            'rmsprop': RMSprop,
            'adagrad': AdaGrad,
        }
        if name.lower() not in optimizers:
            raise ValueError(f"Unknown optimizer: {name}")
        return optimizers[name.lower()]()
    
    def _get_loss(self, name):
        """Get loss instance from string name."""
        losses = {
            'mse': MeanSquaredError,
            'mae': MeanAbsoluteError,
            'binary_crossentropy': BinaryCrossEntropy,
            'categorical_crossentropy': CategoricalCrossEntropy,
        }
        if name.lower() not in losses:
            raise ValueError(f"Unknown loss: {name}")
        return losses[name.lower()]()
```

#### Day 3: Dense Layer Implementation

**Files to Create:**
- `lynxlearn/neural_network/layers/_dense.py`

**Tasks:**
1. Implement `Dense` layer class
   - Weight initialization
   - Forward pass: `output = input @ weights + bias`
   - Backward pass: compute gradients
   - Parameter management

**Code Example:**

```python
# lynxlearn/neural_network/layers/_dense.py

import numpy as np
from ._base import BaseLayer
from ..initializers import HeInitializer, XavierInitializer

class Dense(BaseLayer):
    """
    Fully connected layer.
    
    Parameters
    ----------
    units : int
        Number of neurons in the layer
    activation : str or callable, optional
        Activation function to use
    use_bias : bool, default=True
        Whether to use bias term
    kernel_initializer : str or Initializer, default='he_normal'
        Initializer for weights
    bias_initializer : str or Initializer, default='zeros'
        Initializer for bias
    kernel_regularizer : Regularizer, optional
        Regularizer for weights
    input_shape : tuple, optional
        Input shape (for first layer)
    
    Examples
    --------
    >>> layer = Dense(128, activation='relu', input_shape=(784,))
    >>> layer.build((None, 784))
    >>> output = layer.forward(X)
    """
    
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='he_normal', bias_initializer='zeros',
                 kernel_regularizer=None, input_shape=None, name=None):
        super().__init__(name=name)
        
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.input_shape = input_shape
        
        # Will be initialized in build()
        self.weights = None
        self.bias = None
        self.grad_weights = None
        self.grad_bias = None
        
        # Store input for backward pass
        self.input_cache = None
    
    def build(self, input_shape):
        """
        Initialize layer parameters.
        
        Parameters
        ----------
        input_shape : tuple
            Shape of input (batch_size, input_features)
        """
        input_dim = input_shape[-1]
        
        # Get initializer
        if isinstance(self.kernel_initializer, str):
            if self.kernel_initializer in ['he_normal', 'he_uniform']:
                initializer = HeInitializer()
            elif self.kernel_initializer in ['xavier', 'glorot']:
                initializer = XavierInitializer()
            else:
                initializer = HeInitializer()
        else:
            initializer = self.kernel_initializer
        
        # Initialize weights
        self.weights = initializer.initialize((input_dim, self.units))
        
        # Initialize bias
        if self.use_bias:
            if self.bias_initializer == 'zeros':
                self.bias = np.zeros(self.units)
            else:
                self.bias = np.zeros(self.units)
        else:
            self.bias = None
        
        # Initialize gradient placeholders
        self.grad_weights = np.zeros_like(self.weights)
        if self.use_bias:
            self.grad_bias = np.zeros_like(self.bias)
        
        self.built = True
        self.output_shape = (input_shape[0], self.units)
    
    def forward(self, x, training=True):
        """
        Forward pass through the layer.
        
        Parameters
        ----------
        x : ndarray of shape (batch_size, input_features)
            Input data
        training : bool
            Whether in training mode
            
        Returns
        -------
        output : ndarray of shape (batch_size, units)
            Layer output
        """
        # Store input for backward pass
        self.input_cache = x
        
        # Linear transformation
        self.z = x @ self.weights
        if self.use_bias:
            self.z += self.bias
        
        # Apply activation
        if self.activation is not None:
            self.output = self._apply_activation(self.z)
        else:
            self.output = self.z
        
        return self.output
    
    def backward(self, grad_output):
        """
        Backward pass to compute gradients.
        
        Parameters
        ----------
        grad_output : ndarray of shape (batch_size, units)
            Gradient from next layer
            
        Returns
        -------
        grad_input : ndarray of shape (batch_size, input_features)
            Gradient for previous layer
        """
        batch_size = grad_output.shape[0]
        
        # Gradient through activation
        if self.activation is not None:
            grad_output = self._activation_gradient(grad_output, self.z)
        
        # Compute gradients (averaged over batch)
        self.grad_weights = self.input_cache.T @ grad_output / batch_size
        if self.use_bias:
            self.grad_bias = np.mean(grad_output, axis=0)
        
        # Add regularization gradient if present
        if self.kernel_regularizer is not None:
            self.grad_weights += self.kernel_regularizer.gradient(self.weights)
        
        # Gradient for previous layer
        grad_input = grad_output @ self.weights.T
        
        return grad_input
    
    def _apply_activation(self, z):
        """Apply activation function."""
        if isinstance(self.activation, str):
            if self.activation == 'relu':
                return np.maximum(0, z)
            elif self.activation == 'sigmoid':
                return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            elif self.activation == 'tanh':
                return np.tanh(z)
            elif self.activation == 'softmax':
                # Stable softmax
                shifted = z - np.max(z, axis=1, keepdims=True)
                exp_z = np.exp(shifted)
                return exp_z / np.sum(exp_z, axis=1, keepdims=True)
            else:
                return z
        else:
            # Custom activation function
            return self.activation.forward(z)
    
    def _activation_gradient(self, grad, z):
        """Compute gradient through activation."""
        if isinstance(self.activation, str):
            if self.activation == 'relu':
                return grad * (z > 0)
            elif self.activation == 'sigmoid':
                sig = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                return grad * sig * (1 - sig)
            elif self.activation == 'tanh':
                tanh = np.tanh(z)
                return grad * (1 - tanh ** 2)
            elif self.activation == 'softmax':
                # For softmax + cross-entropy, gradient is simplified
                # (handled in loss function)
                return grad
            else:
                return grad
        else:
            return self.activation.backward(grad, z)
    
    def get_params(self):
        """Get layer parameters."""
        if self.use_bias:
            return {'weights': self.weights, 'bias': self.bias}
        return {'weights': self.weights}
    
    def set_params(self, params):
        """Set layer parameters."""
        self.weights = params['weights']
        if self.use_bias and 'bias' in params:
            self.bias = params['bias']
    
    def get_gradients(self):
        """Get parameter gradients."""
        if self.use_bias:
            return {'weights': self.grad_weights, 'bias': self.grad_bias}
        return {'weights': self.grad_weights}
    
    def __repr__(self):
        return f"Dense(units={self.units}, activation='{self.activation}')"
```

#### Day 4: SGD Optimizer

**Files to Create:**
- `lynxlearn/neural_network/optimizers/__init__.py`
- `lynxlearn/neural_network/optimizers/_base.py`
- `lynxlearn/neural_network/optimizers/_sgd.py`

**Tasks:**
1. Implement `BaseOptimizer` class
2. Implement `SGD` optimizer with:
   - Basic gradient descent
   - Momentum
   - Nesterov accelerated gradient

**Code Example:**

```python
# lynxlearn/neural_network/optimizers/_sgd.py

import numpy as np
from ._base import BaseOptimizer

class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate
    momentum : float, default=0.0
        Momentum coefficient (0 = no momentum)
    nesterov : bool, default=False
        Whether to use Nesterov momentum
    clipnorm : float, optional
        Gradient clipping by norm
    clipvalue : float, optional
        Gradient clipping by value
    
    Examples
    --------
    >>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
    >>> optimizer.update(layer)
    
    # Nesterov momentum
    >>> optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False,
                 clipnorm=None, clipvalue=None):
        super().__init__(learning_rate=learning_rate)
        
        self.momentum = momentum
        self.nesterov = nesterov
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        
        # Velocity for momentum
        self.velocities = {}
    
    def update(self, layer):
        """
        Update layer parameters using SGD.
        
        Parameters
        ----------
        layer : BaseLayer
            Layer with parameters to update
        """
        layer_id = id(layer)
        params = layer.get_params()
        grads = layer.get_gradients()
        
        # Apply gradient clipping
        grads = self._clip_gradients(grads)
        
        # Initialize velocities if needed
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {}
            for key in grads:
                self.velocities[layer_id][key] = np.zeros_like(grads[key])
        
        # Update each parameter
        for key in params:
            grad = grads[key]
            
            if self.momentum > 0:
                # Compute velocity
                v = self.velocities[layer_id][key]
                v = self.momentum * v - self.learning_rate * grad
                self.velocities[layer_id][key] = v
                
                # Nesterov momentum
                if self.nesterov:
                    v = self.momentum * v - self.learning_rate * grad
                
                # Update parameter
                params[key] += v
            else:
                # Plain SGD
                params[key] -= self.learning_rate * grad
        
        layer.set_params(params)
    
    def _clip_gradients(self, grads):
        """Apply gradient clipping."""
        if self.clipnorm is not None:
            # Clip by global norm
            total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
            if total_norm > self.clipnorm:
                scale = self.clipnorm / total_norm
                grads = {k: v * scale for k, v in grads.items()}
        
        if self.clipvalue is not None:
            # Clip by value
            grads = {k: np.clip(v, -self.clipvalue, self.clipvalue) 
                     for k, v in grads.items()}
        
        return grads
    
    def get_state(self):
        """Get optimizer state."""
        return {
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'nesterov': self.nesterov,
            'velocities': self.velocities
        }
    
    def set_state(self, state):
        """Restore optimizer state."""
        self.learning_rate = state['learning_rate']
        self.momentum = state['momentum']
        self.nesterov = state['nesterov']
        self.velocities = state['velocities']
    
    def __repr__(self):
        return f"SGD(learning_rate={self.learning_rate}, momentum={self.momentum})"
```

#### Day 5: MSE Loss and Integration

**Files to Create:**
- `lynxlearn/neural_network/losses/__init__.py`
- `lynxlearn/neural_network/losses/_base.py`
- `lynxlearn/neural_network/losses/_mse.py`

**Tasks:**
1. Implement `BaseLoss` class
2. Implement `MeanSquaredError` loss
3. Create simple integration test

**Code Example:**

```python
# lynxlearn/neural_network/losses/_mse.py

import numpy as np
from ._base import BaseLoss

class MeanSquaredError(BaseLoss):
    """
    Mean Squared Error loss.
    
    loss = mean((y_true - y_pred)^2)
    
    Parameters
    ----------
    reduction : str, default='mean'
        Type of reduction ('mean' or 'sum')
    
    Examples
    --------
    >>> loss = MeanSquaredError()
    >>> value = loss.compute(y_true, y_pred)
    >>> grad = loss.gradient(y_true, y_pred)
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.name = 'mse'
    
    def compute(self, y_true, y_pred):
        """
        Compute MSE loss.
        
        Parameters
        ----------
        y_true : ndarray
            True labels
        y_pred : ndarray
            Predicted values
            
        Returns
        -------
        loss : float
            Mean squared error
        """
        error = y_true - y_pred
        squared_error = error ** 2
        
        if self.reduction == 'mean':
            return np.mean(squared_error)
        else:
            return np.sum(squared_error)
    
    def gradient(self, y_true, y_pred):
        """
        Compute gradient of MSE loss.
        
        d(MSE)/d(y_pred) = 2 * (y_pred - y_true) / n
        
        Parameters
        ----------
        y_true : ndarray
            True labels
        y_pred : ndarray
            Predicted values
            
        Returns
        -------
        gradient : ndarray
            Gradient of loss w.r.t. predictions
        """
        n = y_true.size
        return 2 * (y_pred - y_true) / n
    
    def __repr__(self):
        return "MeanSquaredError()"
```

**Integration Test:**

```python
# tests/test_neural_network/test_integration.py

import numpy as np
from lynxlearn.neural_network import Sequential, Dense
from lynxlearn.neural_network.optimizers import SGD
from lynxlearn.neural_network.losses import MeanSquaredError

def test_simple_regression():
    """Test simple regression task."""
    # Create simple linear data: y = 3x + 5
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 3 * X + 5 + np.random.randn(100, 1) * 0.1
    
    # Build model
    model = Sequential([
        Dense(1, input_shape=(1,))  # No activation for regression
    ])
    
    # Compile
    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss=MeanSquaredError()
    )
    
    # Train
    history = model.train(X, y, epochs=100, batch_size=32, verbose=0)
    
    # Check learned weights are close to [3, 5]
    learned_weights = model.layers[0].weights[0, 0]
    learned_bias = model.layers[0].bias[0]
    
    assert abs(learned_weights - 3.0) < 0.5, f"Weight should be ~3.0, got {learned_weights}"
    assert abs(learned_bias - 5.0) < 0.5, f"Bias should be ~5.0, got {learned_bias}"
    
    # Check final loss is small
    assert history['loss'][-1] < 0.1, f"Final loss should be < 0.1, got {history['loss'][-1]}"
    
    print("✓ Simple regression test passed!")

if __name__ == '__main__':
    test_simple_regression()
```

---

### Phase 2: Activation Functions (Week 1, Days 6-7 + Week 2, Days 1-2)

**Objective:** Implement comprehensive activation functions with proper gradient computation.

#### Files to Create:
- `lynxlearn/neural_network/activations/__init__.py`
- `lynxlearn/neural_network/activations/_base.py`
- `lynxlearn/neural_network/activations/_relu.py`
- `lynxlearn/neural_network/activations/_sigmoid.py`
- `lynxlearn/neural_network/activations/_tanh.py`
- `lynxlearn/neural_network/activations/_softmax.py`
- `lynxlearn/neural_network/activations/_elu.py`
- `lynxlearn/neural_network/activations/_swish.py`
- `lynxlearn/neural_network/layers/_activation.py`

#### Tasks:

1. **Create BaseActivation class**

```python
# lynxlearn/neural_network/activations/_base.py

from abc import ABC, abstractmethod
import numpy as np

class BaseActivation(ABC):
    """Base class for all activation functions."""
    
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : ndarray
            Input array
            
        Returns
        -------
        output : ndarray
            Activated output
        """
        pass
    
    @abstractmethod
    def backward(self, grad_output, x):
        """
        Backward pass.
        
        Parameters
        ----------
        grad_output : ndarray
            Gradient from next layer
        x : ndarray
            Original input (before activation)
            
        Returns
        -------
        grad_input : ndarray
            Gradient for previous layer
        """
        pass
    
    def __call__(self, x):
        """Make activation callable."""
        return self.forward(x)
    
    def __repr__(self):
        return f"{self.name}()"
```

2. **Implement ReLU family**

```python
# lynxlearn/neural_network/activations/_relu.py

import numpy as np
from ._base import BaseActivation

class ReLU(BaseActivation):
    """
    Rectified Linear Unit activation.
    
    ReLU(x) = max(0, x)
    
    Advantages:
    - No vanishing gradient problem (for positive inputs)
    - Computationally efficient
    - Sparse activation
    
    Disadvantages:
    - Dying ReLU problem (neurons can "die")
    - Not zero-centered
    
    Examples
    --------
    >>> activation = ReLU()
    >>> output = activation.forward(np.array([-1, 0, 1, 2]))
    array([0, 0, 1, 2])
    """
    
    def __init__(self):
        super().__init__(name='relu')
    
    def forward(self, x):
        """Forward pass: ReLU(x) = max(0, x)"""
        return np.maximum(0, x)
    
    def backward(self, grad_output, x):
        """Backward pass: gradient = 1 if x > 0 else 0"""
        return grad_output * (x > 0)


class LeakyReLU(BaseActivation):
    """
    Leaky ReLU activation.
    
    LeakyReLU(x) = x if x > 0 else alpha * x
    
    Parameters
    ----------
    alpha : float, default=0.01
        Slope for negative inputs
    
    Advantages:
    - No dying ReLU problem
    - Still computationally efficient
    
    Examples
    --------
    >>> activation = LeakyReLU(alpha=0.01)
    >>> output = activation.forward(np.array([-1, 0, 1]))
    array([-0.01, 0, 1])
    """
    
    def __init__(self, alpha=0.01):
        super().__init__(name='leaky_relu')
        self.alpha = alpha
    
    def forward(self, x):
        """Forward pass: LeakyReLU(x) = max(alpha*x, x)"""
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, grad_output, x):
        """Backward pass: gradient = 1 if x > 0 else alpha"""
        return grad_output * np.where(x > 0, 1, self.alpha)


class PReLU(BaseActivation):
    """
    Parametric ReLU activation.
    
    PReLU(x) = x if x > 0 else alpha * x
    
    Unlike LeakyReLU, alpha is learned during training.
    
    Parameters
    ----------
    alpha : float or ndarray, default=0.01
        Learnable slope parameter(s)
    
    Examples
    --------
    >>> activation = PReLU(alpha=0.01)
    """
    
    def __init__(self, alpha=0.01):
        super().__init__(name='prelu')
        self.alpha = alpha if isinstance(alpha, np.ndarray) else np.array(alpha)
        self.grad_alpha = None
    
    def forward(self, x):
        """Forward pass"""
        return np.where(x > 0, x, x * self.alpha)
    
    def backward(self, grad_output, x):
        """Backward pass including alpha gradient"""
        # Gradient for x
        grad_x = grad_output * np.where(x > 0, 1, self.alpha)
        
        # Gradient for alpha
        self.grad_alpha = np.sum(grad_output * np.where(x > 0, 0, x))
        
        return grad_x
    
    def get_params(self):
        """Get learnable parameters"""
        return {'alpha': self.alpha}
    
    def get_gradients(self):
        """Get gradients for learnable parameters"""
        return {'alpha': self.grad_alpha}
```

3. **Implement Softmax (numerically stable)**

```python
# lynxlearn/neural_network/activations/_softmax.py

import numpy as np
from ._base import BaseActivation

class Softmax(BaseActivation):
    """
    Softmax activation function.
    
    softmax(x)_i = exp(x_i) / sum(exp(x_j))
    
    Typically used for multi-class classification output layer.
    
    Note: Numerically stable implementation using log-sum-exp trick.
    
    Examples
    --------
    >>> activation = Softmax()
    >>> output = activation.forward(np.array([[1, 2, 3]]))
    # Output: [[0.09, 0.24, 0.67]]
    """
    
    def __init__(self):
        super().__init__(name='softmax')
    
    def forward(self, x):
        """
        Numerically stable softmax.
        
        Uses: softmax(x) = softmax(x - max(x))
        to prevent overflow in exp()
        """
        # Shift for numerical stability
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, grad_output, x):
        """
        Backward pass for softmax.
        
        Note: For softmax + cross-entropy loss, the combined gradient
        simplifies to (y_pred - y_true), which is computed in the loss function.
        
        This method is provided for standalone softmax gradient computation.
        """
        # For standalone softmax gradient (rarely used)
        # Jacobian is: diag(s) - s @ s.T
        # For batch processing, this is complex. Usually handled in loss.
        
        # Simplified version for common case where grad_output comes from cross-entropy
        return grad_output
    
    def forward_with_cache(self, x):
        """Forward pass that caches output for backward."""
        self.output = self.forward(x)
        return self.output
```

4. **Implement Sigmoid and Tanh**

```python
# lynxlearn/neural_network/activations/_sigmoid.py

import numpy as np
from ._base import BaseActivation

class Sigmoid(BaseActivation):
    """
    Sigmoid activation function.
    
    sigmoid(x) = 1 / (1 + exp(-x))
    
    Advantages:
    - Smooth gradient
    - Output bounded in (0, 1)
    
    Disadvantages:
    - Vanishing gradient problem
    - Not zero-centered
    
    Examples
    --------
    >>> activation = Sigmoid()
    >>> output = activation.forward(np.array([-1, 0, 1]))
    # Output: [0.269, 0.5, 0.731]
    """
    
    def __init__(self):
        super().__init__(name='sigmoid')
    
    def forward(self, x):
        """
        Numerically stable sigmoid.
        
        Separate handling for positive and negative values to prevent overflow.
        """
        result = np.zeros_like(x, dtype=np.float64)
        
        # For positive values: 1 / (1 + exp(-x))
        pos_mask = x >= 0
        result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        
        # For negative values: exp(x) / (1 + exp(x))
        neg_mask = ~pos_mask
        exp_x = np.exp(x[neg_mask])
        result[neg_mask] = exp_x / (1 + exp_x)
        
        return result
    
    def backward(self, grad_output, x):
        """Backward pass: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))"""
        s = self.forward(x)  # sigmoid(x)
        return grad_output * s * (1 - s)


class Tanh(BaseActivation):
    """
    Hyperbolic tangent activation.
    
    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Advantages:
    - Zero-centered output
    - Smooth gradient
    
    Disadvantages:
    - Vanishing gradient problem (less severe than sigmoid)
    
    Examples
    --------
    >>> activation = Tanh()
    >>> output = activation.forward(np.array([-1, 0, 1]))
    # Output: [-0.762, 0, 0.762]
    """
    
    def __init__(self):
        super().__init__(name='tanh')
    
    def forward(self, x):
        """Forward pass: use numpy's stable tanh"""
        return np.tanh(x)
    
    def backward(self, grad_output, x):
        """Backward pass: tanh'(x) = 1 - tanh(x)^2"""
        t = np.tanh(x)
        return grad_output * (1 - t ** 2)
```

5. **Implement ELU and SELU**

```python
# lynxlearn/neural_network/activations/_elu.py

import numpy as np
from ._base import BaseActivation

class ELU(BaseActivation):
    """
    Exponential Linear Unit activation.
    
    ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
    
    Parameters
    ----------
    alpha : float, default=1.0
        Scale factor for negative values
    
    Advantages:
    - No dying ReLU problem
    - Self-normalizing (for SELU)
    - Smooth at x=0
    
    Examples
    --------
    >>> activation = ELU(alpha=1.0)
    """
    
    def __init__(self, alpha=1.0):
        super().__init__(name='elu')
        self.alpha = alpha
    
    def forward(self, x):
        """Forward pass"""
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, grad_output, x):
        """Backward pass"""
        return grad_output * np.where(x > 0, 1, self.alpha * np.exp(x))


class SELU(BaseActivation):
    """
    Scaled Exponential Linear Unit activation.
    
    SELU(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
    
    Where:
    - alpha ≈ 1.6732632423543772
    - scale ≈ 1.0507009873554805
    
    These values are derived to achieve self-normalization.
    
    When combined with Lecun normal initialization and AlphaDropout,
    SELU networks self-normalize to mean=0, std=1.
    
    Examples
    --------
    >>> activation = SELU()
    """
    
    def __init__(self):
        super().__init__(name='selu')
        # Fixed parameters for self-normalization
        self.alpha = 1.6732632423543772
        self.scale = 1.0507009873554805
    
    def forward(self, x):
        """Forward pass"""
        return self.scale * np.where(
            x > 0, 
            x, 
            self.alpha * (np.exp(x) - 1)
        )
    
    def backward(self, grad_output, x):
        """Backward pass"""
        return grad_output * self.scale * np.where(
            x > 0, 
            1, 
            self.alpha * np.exp(x)
        )
```

6. **Implement Swish and GELU**

```python
# lynxlearn/neural_network/activations/_swish.py

import numpy as np
from ._base import BaseActivation
from ._sigmoid import Sigmoid

class Swish(BaseActivation):
    """
    Swish activation function.
    
    Swish(x) = x * sigmoid(x)
    
    Also known as SiLU (Sigmoid-weighted Linear Unit).
    
    Advantages:
    - Smooth, non-monotonic
    - Self-gated
    - Performs well in deep networks
    
    Examples
    --------
    >>> activation = Swish()
    """
    
    def __init__(self):
        super().__init__(name='swish')
        self.sigmoid = Sigmoid()
    
    def forward(self, x):
        """Forward pass: x * sigmoid(x)"""
        return x * self.sigmoid.forward(x)
    
    def backward(self, grad_output, x):
        """
        Backward pass.
        
        d(Swish)/dx = sigmoid(x) + x * sigmoid'(x)
                    = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                    = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        """
        sig = self.sigmoid.forward(x)
        grad = sig * (1 + x * (1 - sig))
        return grad_output * grad


class GELU(BaseActivation):
    """
    Gaussian Error Linear Unit activation.
    
    GELU(x) = x * Φ(x) where Φ is the Gaussian CDF
    
    Approximated as:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    Advantages:
    - Smooth, non-monotonic
    - Used in transformer models (BERT, GPT)
    - Better performance than ReLU in some cases
    
    Examples
    --------
    >>> activation = GELU()
    """
    
    def __init__(self):
        super().__init__(name='gelu')
    
    def forward(self, x):
        """
        Forward pass using approximation.
        
        More accurate than using actual Gaussian CDF.
        """
        # Approximation formula
        return 0.5 * x * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
        ))
    
    def backward(self, grad_output, x):
        """Backward pass"""
        # Derivative of approximation
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        pdf = 0.5 * np.sqrt(2 / np.pi) * (1 - np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
        ) ** 2) * (1 + 3 * 0.044715 * x ** 2)
        
        return grad_output * (cdf + x * pdf)
```

7. **Create Activation Layer wrapper**

```python
# lynxlearn/neural_network/layers/_activation.py

import numpy as np
from ._base import BaseLayer
from ..activations import (
    ReLU, LeakyReLU, PReLU, Sigmoid, Tanh, 
    Softmax, ELU, SELU, Swish, GELU
)

class Activation(BaseLayer):
    """
    Activation layer wrapper.
    
    Allows using activation functions as layers.
    
    Parameters
    ----------
    activation : str or BaseActivation
        Activation function name or instance
    
    Examples
    --------
    >>> layer = Activation('relu')
    >>> # or
    >>> layer = Activation(ReLU())
    """
    
    # Mapping from string names to classes
    _activation_map = {
        'relu': ReLU,
        'leaky_relu': LeakyReLU,
        'prelu': PReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'softmax': Softmax,
        'elu': ELU,
        'selu': SELU,
        'swish': Swish,
        'silu': Swish,  # Alias
        'gelu': GELU,
    }
    
    def __init__(self, activation, name=None):
        super().__init__(name=name)
        
        # Get activation function
        if isinstance(activation, str):
            if activation not in self._activation_map:
                raise ValueError(
                    f"Unknown activation: {activation}. "
                    f"Available: {list(self._activation_map.keys())}"
                )
            self.activation = self._activation_map[activation]()
        else:
            self.activation = activation
        
        self.name = self.activation.name
    
    def build(self, input_shape):
        """Activation layers have no parameters."""
        self.output_shape = input_shape
        self.built = True
    
    def forward(self, x, training=True):
        """Forward pass through activation."""
        self.input_cache = x
        return self.activation.forward(x)
    
    def backward(self, grad_output):
        """Backward pass through activation."""
        return self.activation.backward(grad_output, self.input_cache)
    
    def get_params(self):
        """Get parameters (if any, e.g., PReLU)."""
        if hasattr(self.activation, 'get_params'):
            return self.activation.get_params()
        return {}
    
    def set_params(self, params):
        """Set parameters (if any)."""
        if hasattr(self.activation, 'set_params'):
            self.activation.set_params(params)
    
    def get_gradients(self):
        """Get gradients (if any)."""
        if hasattr(self.activation, 'get_gradients'):
            return self.activation.get_gradients()
        return {}
    
    def __repr__(self):
        return f"Activation('{self.name}')"
```

**Week 2, Day 1-2 Tasks:**
- Implement gradient checking utility
- Write comprehensive tests for all activations
- Document mathematical formulas in docstrings

```python
# lynxlearn/neural_network/_utils.py

import numpy as np

def gradient_check(model, X, y, epsilon=1e-5, threshold=1e-6):
    """
    Perform numerical gradient checking.
    
    Compares analytical gradients (from backprop) with numerical gradients
    (finite differences).
    
    Parameters
    ----------
    model : BaseNeuralNetwork
        Model to check
    X : ndarray
        Input data
    y : ndarray
        Target labels
    epsilon : float
        Small value for numerical gradient computation
    threshold : float
        Maximum allowed relative error
        
    Returns
    -------
    passed : bool
        Whether gradient check passed
    max_error : float
        Maximum relative error found
    """
    # Forward pass to compute loss
    y_pred = model._forward_pass(X)
    loss = model.loss.compute(y, y_pred)
    
    # Backward pass to compute analytical gradients
    grad = model.loss.gradient(y, y_pred)
    model._backward_pass(grad)
    
    errors = []
    
    # Check each layer's gradients
    for i, layer in enumerate(model.layers):
        if not hasattr(layer, 'get_params'):
            continue
        
        params = layer.get_params()
        grads = layer.get_gradients()
        
        for key in params:
            param = params[key]
            analytical_grad = grads[key]
            
            # Compute numerical gradient
            numerical_grad = np.zeros_like(param)
            
            # Iterate through each element
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                original_value = param[idx]
                
                # Compute f(x + epsilon)
                param[idx] = original_value + epsilon
                layer.set_params(params)
                y_pred_plus = model._forward_pass(X)
                loss_plus = model.loss.compute(y, y_pred_plus)
                
                # Compute f(x - epsilon)
                param[idx] = original_value - epsilon
                layer.set_params(params)
                y_pred_minus = model._forward_pass(X)
                loss_minus = model.loss.compute(y, y_pred_minus)
                
                # Restore original value
                param[idx] = original_value
                layer.set_params(params)
                
                # Numerical gradient
                numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                
                it.iternext()
            
            # Compute relative error
            diff = np.abs(analytical_grad - numerical_grad)
            norm = np.abs(analytical_grad) + np.abs(numerical_grad) + 1e-8
            relative_error = np.max(diff / norm)
            
            errors.append(relative_error)
            
            print(f"Layer {i}, {key}: max relative error = {relative_error:.2e}")
    
    max_error = max(errors)
    passed = max_error < threshold
    
    print(f"\n{'='*60}")
    if passed:
        print(f"✓ Gradient check PASSED (max error: {max_error:.2e} < {threshold:.2e})")
    else:
        print(f"✗ Gradient check FAILED (max error: {max_error:.2e} > {threshold:.2e})")
    print(f"{'='*60}")
    
    return passed, max_error
```

---

### Phase 3: Advanced Optimizers (Week 2, Days 3-5)

**Objective:** Implement Adam, RMSprop, and AdaGrad optimizers.

#### Files to Create:
- `lynxlearn/neural_network/optimizers/_adam.py`
- `lynxlearn/neural_network/optimizers/_rmsprop.py`
- `lynxlearn/neural_network/optimizers/_adagrad.py`

#### Key Implementations:

1. **Adam Optimizer**

```python
# lynxlearn/neural_network/optimizers/_adam.py

import numpy as np
from ._base import BaseOptimizer

class Adam(BaseOptimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    Combines momentum and RMSprop for adaptive learning rates.
    
    Parameters
    ----------
    learning_rate : float, default=0.001
        Learning rate
    beta_1 : float, default=0.9
        Exponential decay rate for 1st moment estimates
    beta_2 : float, default=0.999
        Exponential decay rate for 2nd moment estimates
    epsilon : float, default=1e-8
        Small constant for numerical stability
    amsgrad : bool, default=False
        Whether to use AMSGrad variant
    
    References
    ----------
    Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
    arXiv preprint arXiv:1412.6980.
    
    Examples
    --------
    >>> optimizer = Adam(learning_rate=0.001)
    """
    
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, amsgrad=False):
        super().__init__(learning_rate=learning_rate)
        
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        
        # State
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.v_hat = {}  # For AMSGrad
        self.t = 0  # Timestep
    
    def update(self, layer):
        """
        Update layer parameters using Adam.
        
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * g
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * g^2
        m_hat = m_t / (1 - beta_1^t)
        v_hat = v_t / (1 - beta_2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
        """
        layer_id = id(layer)
        params = layer.get_params()
        grads = layer.get_gradients()
        
        self.t += 1
        
        # Initialize state if needed
        if layer_id not in self.m:
            self.m[layer_id] = {}
            self.v[layer_id] = {}
            for key in grads:
                self.m[layer_id][key] = np.zeros_like(grads[key])
                self.v[layer_id][key] = np.zeros_like(grads[key])
            if self.amsgrad:
                self.v_hat[layer_id] = {key: np.zeros_like(grads[key]) for key in grads}
        
        # Update each parameter
        for key in params:
            g = grads[key]
            
            # Update biased first moment estimate
            self.m[layer_id][key] = self.beta_1 * self.m[layer_id][key] + (1 - self.beta_1) * g
            
            # Update biased second raw moment estimate
            self.v[layer_id][key] = self.beta_2 * self.v[layer_id][key] + (1 - self.beta_2) * g**2
            
            # Bias correction
            m_hat = self.m[layer_id][key] / (1 - self.beta_1 ** self.t)
            v_hat = self.v[layer_id][key] / (1 - self.beta_2 ** self.t)
            
            if self.amsgrad:
                # AMSGrad: use maximum of past v_hat
                self.v_hat[layer_id][key] = np.maximum(self.v_hat[layer_id][key], v_hat)
                v_hat = self.v_hat[layer_id][key]
            
            # Update parameter
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        layer.set_params(params)
    
    def get_state(self):
        """Get optimizer state for serialization."""
        return {
            'learning_rate': self.learning_rate,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'epsilon': self.epsilon,
            'm': self.m,
            'v': self.v,
            't': self.t
        }
    
    def set_state(self, state):
        """Restore optimizer state."""
        self.learning_rate = state['learning_rate']
        self.beta_1 = state['beta_1']
        self.beta_2 = state['beta_2']
        self.epsilon = state['epsilon']
        self.m = state['m']
        self.v = state['v']
        self.t = state['t']
    
    def __repr__(self):
        return f"Adam(learning_rate={self.learning_rate})"


class AdamW(Adam):
    """
    Adam with decoupled weight decay.
    
    Separates weight decay from gradient update for better generalization.
    
    Parameters
    ----------
    learning_rate : float, default=0.001
    beta_1 : float, default=0.9
    beta_2 : float, default=0.999
    epsilon : float, default=1e-8
    weight_decay : float, default=0.01
        Weight decay coefficient
    
    References
    ----------
    Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization.
    arXiv preprint arXiv:1711.05101.
    
    Examples
    --------
    >>> optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
    """
    
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, weight_decay=0.01):
        super().__init__(learning_rate, beta_1, beta_2, epsilon)
        self.weight_decay = weight_decay
    
    def update(self, layer):
        """Update with decoupled weight decay."""
        layer_id = id(layer)
        params = layer.get_params()
        grads = layer.get_gradients()
        
        self.t += 1
        
        if layer_id not in self.m:
            self.m[layer_id] = {}
            self.v[layer_id] = {}
            for key in grads:
                self.m[layer_id][key] = np.zeros_like(grads[key])
                self.v[layer_id][key] = np.zeros_like(grads[key])
        
        for key in params:
            g = grads[key]
            
            # Weight decay (applied directly to weights, not bias)
            if 'weight' in key.lower():
                g = g + self.weight_decay * params[key]
            
            # Adam update
            self.m[layer_id][key] = self.beta_1 * self.m[layer_id][key] + (1 - self.beta_1) * g
            self.v[layer_id][key] = self.beta_2 * self.v[layer_id][key] + (1 - self.beta_2) * g**2
            
            m_hat = self.m[layer_id][key] / (1 - self.beta_1 ** self.t)
            v_hat = self.v[layer_id][key] / (1 - self.beta_2 ** self.t)
            
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        layer.set_params(params)
    
    def __repr__(self):
        return f"AdamW(learning_rate={self.learning_rate}, weight_decay={self.weight_decay})"
```

---

### Phase 4: Loss Functions (Week 2, Days 6-7 + Week 3, Day 1)

**Objective:** Implement classification and regression loss functions.

#### Files to Create:
- `lynxlearn/neural_network/losses/_binary_crossentropy.py`
- `lynxlearn/neural_network/losses/_categorical_crossentropy.py`
- `lynxlearn/neural_network/losses/_sparse_crossentropy.py`
- `lynxlearn/neural_network/losses/_huber.py`

#### Key Implementation (Cross-Entropy):

```python
# lynxlearn/neural_network/losses/_categorical_crossentropy.py

import numpy as np
from ._base import BaseLoss

class CategoricalCrossEntropy(BaseLoss):
    """
    Categorical cross-entropy loss.
    
    Used for multi-class classification with one-hot encoded labels.
    
    loss = -sum(y_true * log(y_pred))
    
    Parameters
    ----------
    from_logits : bool, default=False
        If True, inputs are logits (pre-softmax)
        If False, inputs are probabilities (post-softmax)
    
    Examples
    --------
    >>> loss = CategoricalCrossEntropy(from_logits=True)
    >>> # For model ending with Dense(n_classes) (no softmax activation)
    >>> model.compile(loss=loss)
    """
    
    def __init__(self, from_logits=False):
        super().__init__()
        self.from_logits = from_logits
        self.name = 'categorical_crossentropy'
    
    def compute(self, y_true, y_pred):
        """
        Compute categorical cross-entropy loss.
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples, n_classes)
            One-hot encoded true labels
        y_pred : ndarray of shape (n_samples, n_classes)
            Predicted probabilities or logits
            
        Returns
        -------
        loss : float
            Mean cross-entropy loss
        """
        # Clip predictions for numerical stability
        if not self.from_logits:
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
            loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        else:
            # Logits: use log-softmax for numerical stability
            log_softmax = y_pred - np.max(y_pred, axis=1, keepdims=True)
            log_softmax = log_softmax - np.log(np.sum(np.exp(log_softmax), axis=1, keepdims=True))
            loss = -np.sum(y_true * log_softmax) / y_true.shape[0]
        
        return loss
    
    def gradient(self, y_true, y_pred):
        """
        Compute gradient of cross-entropy loss.
        
        For softmax output: grad = y_pred - y_true
        For logits output: same (combined softmax + cross-entropy gradient)
        
        Parameters
        ----------
        y_true : ndarray
            One-hot encoded true labels
        y_pred : ndarray
            Predicted probabilities or logits
            
        Returns
        -------
        gradient : ndarray
            Gradient w.r.t. predictions
        """
        if not self.from_logits:
            # Apply softmax if not already applied
            shifted = y_pred - np.max(y_pred, axis=1, keepdims=True)
            exp_pred = np.exp(shifted)
            y_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
        # Combined gradient: softmax + cross-entropy
        return (y_pred - y_true) / y_true.shape[0]
    
    def __repr__(self):
        return f"CategoricalCrossEntropy(from_logits={self.from_logits})"
```

---

### Phase 5: Regularization (Week 3, Days 2-4)

**Objective:** Implement dropout and batch normalization.

#### Dropout Implementation:

```python
# lynxlearn/neural_network/layers/_dropout.py

import numpy as np
from ._base import BaseLayer

class Dropout(BaseLayer):
    """
    Dropout regularization layer.
    
    Randomly sets a fraction of inputs to 0 during training.
    
    Parameters
    ----------
    rate : float
        Fraction of inputs to drop (0.0 to 1.0)
    seed : int, optional
        Random seed for reproducibility
    
    Note: During inference, outputs are scaled by (1 - rate) to maintain
    the same expected value.
    
    Examples
    --------
    >>> layer = Dropout(0.5)  # Drop 50% of inputs
    """
    
    def __init__(self, rate, seed=None):
        super().__init__()
        
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate}")
        
        self.rate = rate
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Cache for backward pass
        self.mask = None
    
    def build(self, input_shape):
        """Dropout has no parameters."""
        self.output_shape = input_shape
        self.built = True
    
    def forward(self, x, training=True):
        """
        Forward pass.
        
        During training: randomly drop inputs and scale remaining
        During inference: return inputs unchanged (scaled during training)
        """
        if not training:
            return x
        
        # Create dropout mask
        self.mask = self.rng.rand(*x.shape) > self.rate
        
        # Scale by 1/(1-rate) to maintain expected value
        return x * self.mask / (1 - self.rate)
    
    def backward(self, grad_output):
        """Backward pass: gradient only flows through non-dropped units."""
        return grad_output * self.mask / (1 - self.rate)
    
    def __repr__(self):
        return f"Dropout(rate={self.rate})"
```

#### Batch Normalization Implementation:

```python
# lynxlearn/neural_network/layers/_batch_norm.py

import numpy as np
from ._base import BaseLayer

class BatchNormalization(BaseLayer):
    """
    Batch Normalization layer.
    
    Normalizes activations to have zero mean and unit variance,
    then scales and shifts with learnable parameters.
    
    Parameters
    ----------
    momentum : float, default=0.99
        Momentum for running statistics
    epsilon : float, default=1e-5
        Small constant for numerical stability
    center : bool, default=True
        Whether to use beta offset
    scale : bool, default=True
        Whether to use gamma scaling
    
    References
    ----------
    Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep
    network training by reducing internal covariate shift.
    
    Examples
    --------
    >>> layer = BatchNormalization(momentum=0.99)
    """
    
    def __init__(self, momentum=0.99, epsilon=1e-5, center=True, scale=True):
        super().__init__()
        
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        
        # Learnable parameters
        self.gamma = None  # Scale
        self.beta = None   # Shift
        
        # Running statistics (for inference)
        self.running_mean = None
        self.running_var = None
        
        # Gradients
        self.grad_gamma = None
        self.grad_beta = None
        
        # Cache for backward pass
        self.x_norm = None
        self.std = None
        self.x_centered = None
    
    def build(self, input_shape):
        """Initialize parameters."""
        n_features = input_shape[-1]
        
        # Initialize gamma and beta
        if self.scale:
            self.gamma = np.ones(n_features)
        else:
            self.gamma = None
        
        if self.center:
            self.beta = np.zeros(n_features)
        else:
            self.beta = None
        
        # Initialize running statistics
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        
        # Initialize gradients
        self.grad_gamma = np.zeros_like(self.gamma) if self.scale else None
        self.grad_beta = np.zeros_like(self.beta) if self.center else None
        
        self.output_shape = input_shape
        self.built = True
    
    def forward(self, x, training=True):
        """
        Forward pass.
        
        During training: use batch statistics
        During inference: use running statistics
        """
        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Normalize
            self.x_centered = x - batch_mean
            self.std = np.sqrt(batch_var + self.epsilon)
            self.x_norm = self.x_centered / self.std
        else:
            # Use running statistics for inference
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Scale and shift
        output = self.x_norm
        if self.scale:
            output = output * self.gamma
        if self.center:
            output = output + self.beta
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass for batch normalization.
        
        Computes gradients for gamma, beta, and input.
        """
        batch_size = grad_output.shape[0]
        
        # Gradients for gamma and beta
        if self.scale:
            self.grad_gamma = np.sum(grad_output * self.x_norm, axis=0)
        if self.center:
            self.grad_beta = np.sum(grad_output, axis=0)
        
        # Gradient for input
        if self.scale:
            dx_hat = grad_output * self.gamma
        else:
            dx_hat = grad_output
        
        # Gradient through normalization
        # See: https://kevinzakka.github.io/2016/09/14/batch_normalization/
        dx = (1. / batch_size) * (1. / self.std) * (
            batch_size * dx_hat 
            - np.sum(dx_hat, axis=0) 
            - self.x_norm * np.sum(dx_hat * self.x_norm, axis=0)
        )
        
        return dx
    
    def get_params(self):
        """Get learnable parameters."""
        params = {}
        if self.scale:
            params['gamma'] = self.gamma
        if self.center:
            params['beta'] = self.beta
        params['running_mean'] = self.running_mean
        params['running_var'] = self.running_var
        return params
    
    def set_params(self, params):
        """Set parameters."""
        if self.scale and 'gamma' in params:
            self.gamma = params['gamma']
        if self.center and 'beta' in params:
            self.beta = params['beta']
        if 'running_mean' in params:
            self.running_mean = params['running_mean']
        if 'running_var' in params:
            self.running_var = params['running_var']
    
    def get_gradients(self):
        """Get parameter gradients."""
        grads = {}
        if self.scale:
            grads['gamma'] = self.grad_gamma
        if self.center:
            grads['beta'] = self.grad_beta
        return grads
    
    def __repr__(self):
        return f"BatchNormalization(momentum={self.momentum})"
```

---

### Phase 6: Model Compilation & Training (Week 3, Days 5-7)

**Objective:** Implement the Sequential model with complete training loop.

#### Files to Create:
- `lynxlearn/neural_network/_model.py`

#### Sequential Model Implementation:

```python
# lynxlearn/neural_network/_model.py

import numpy as np
from ._base import BaseNeuralNetwork
from .layers import Input
from .callbacks import ProgressBar

class Sequential(BaseNeuralNetwork):
    """
    Sequential model - stack of layers.
    
    Parameters
    ----------
    layers : list, optional
        List of layers to add to the model
    
    Examples
    --------
    >>> model = Sequential([
    ...     Dense(128, activation='relu', input_shape=(784,)),
    ...     Dropout(0.2),
    ...     Dense(10, activation='softmax')
    ... ])
    >>> model.compile(optimizer='adam', loss='categorical_crossentropy')
    >>> model.train(X_train, y_train, epochs=10)
    """
    
    def __init__(self, layers=None):
        super().__init__()
        
        if layers is not None:
            for layer in layers:
                self.add(layer)
    
    def add(self, layer):
        """
        Add a layer to the model.
        
        Parameters
        ----------
        layer : BaseLayer
            Layer instance to add
        """
        # Handle input layer
        if len(self.layers) == 0 and hasattr(layer, 'input_shape') and layer.input_shape is not None:
            input_layer = Input(shape=layer.input_shape)
            self.layers.append(input_layer)
        
        self.layers.append(layer)
    
    def _build(self, input_shape):
        """Build all layers (initialize parameters)."""
        current_shape = input_shape
        
        for layer in self.layers:
            if not layer.built:
                layer.build(current_shape)
            current_shape = layer.output_shape
        
        self.built = True
    
    def compile(self, optimizer='adam', loss=None, metrics=None, **kwargs):
        """
        Configure the model for training.
        
        Parameters
        ----------
        optimizer : str or BaseOptimizer
            Optimizer to use
        loss : str or BaseLoss
            Loss function
        metrics : list
            Metrics to track
        """
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
        
        # Build model on first compile if input shape is known
        if not self.built and len(self.layers) > 0:
            first_layer = self.layers[0]
            if hasattr(first_layer, 'input_shape') and first_layer.input_shape is not None:
                self._build(first_layer.input_shape)
    
    def train(self, X, y, epochs=100, batch_size=32,
              validation_data=None, validation_split=0.0,
              callbacks=None, verbose=1, shuffle=True,
              initial_epoch=0, **kwargs):
        """
        Train the model.
        
        Parameters
        ----------
        X : ndarray
            Training data
        y : ndarray
            Target values
        epochs : int
            Number of epochs
        batch_size : int
            Batch size
        validation_data : tuple
            (X_val, y_val) for validation
        validation_split : float
            Fraction of training data for validation
        callbacks : list
            List of callback instances
        verbose : int
            0=silent, 1=progress bar, 2=one line per epoch
        shuffle : bool
            Whether to shuffle data each epoch
            
        Returns
        -------
        history : dict
            Training history
        """
        # Validate inputs
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Build model if not built
        if not self.built:
            self._build((None,) + X.shape[1:])
        
        # Handle validation split
        if validation_data is None and validation_split > 0:
            n_val = int(len(X) * validation_split)
            X_val, y_val = X[-n_val:], y[-n_val:]
            X, y = X[:-n_val], y[:-n_val]
            validation_data = (X_val, y_val)
        
        # Initialize history
        self.history = {
            'loss': [],
            'val_loss': [] if validation_data else None
        }
        
        # Add metrics to history
        for metric in self.metrics:
            self.history[metric.__name__ if callable(metric) else metric] = []
            if validation_data:
                self.history[f'val_{metric.__name__ if callable(metric) else metric}'] = []
        
        # Initialize callbacks
        callbacks = callbacks or []
        if verbose == 1:
            callbacks.insert(0, ProgressBar())
        
        for callback in callbacks:
            callback.model = self
        
        # Training loop
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Callback: on_train_begin
        logs = {}
        for callback in callbacks:
            callback.on_train_begin(logs)
        
        for epoch in range(initial_epoch, epochs):
            if self.stop_training:
                break
            
            # Shuffle data
            if shuffle:
                indices = np.random.permutation(n_samples)
                X, y = X[indices], y[indices]
            
            # Callback: on_epoch_begin
            logs = {'epoch': epoch}
            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)
            
            # Batch training
            epoch_loss = 0.0
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                X_batch = X[start:end]
                y_batch = y[start:end]
                
                # Callback: on_batch_begin
                logs = {'batch': batch_idx}
                for callback in callbacks:
                    callback.on_batch_begin(batch_idx, logs)
                
                # Forward pass
                y_pred = self._forward_pass(X_batch, training=True)
                
                # Compute loss
                batch_loss = self.loss.compute(y_batch, y_pred)
                epoch_loss += batch_loss
                
                # Backward pass
                grad = self.loss.gradient(y_batch, y_pred)
                self._backward_pass(grad)
                
                # Update parameters
                for layer in self.layers:
                    if hasattr(layer, 'get_gradients') and layer.get_gradients():
                        self.optimizer.update(layer)
                
                # Callback: on_batch_end
                logs = {'batch': batch_idx, 'loss': batch_loss}
                for callback in callbacks:
                    callback.on_batch_end(batch_idx, logs)
            
            # Compute epoch metrics
            epoch_loss /= n_batches
            self.history['loss'].append(epoch_loss)
            
            # Validation
            if validation_data:
                X_val, y_val = validation_data
                y_pred_val = self._forward_pass(X_val, training=False)
                val_loss = self.loss.compute(y_val, y_pred_val)
                self.history['val_loss'].append(val_loss)
            
            # Callback: on_epoch_end
            logs = {'epoch': epoch, 'loss': epoch_loss}
            if validation_data:
                logs['val_loss'] = val_loss
            
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)
        
        # Callback: on_train_end
        for callback in callbacks:
            callback.on_train_end(logs)
        
        self._is_trained = True
        return self.history
    
    def summary(self):
        """Print model summary."""
        print("=" * 70)
        print(f"Model: {self.__class__.__name__}")
        print("=" * 70)
        print(f"{'Layer (type)':<30} {'Output Shape':<20} {'Param #':<15}")
        print("=" * 70)
        
        total_params = 0
        trainable_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_type = layer.__class__.__name__
            output_shape = str(layer.output_shape) if hasattr(layer, 'output_shape') else 'Unknown'
            
            # Count parameters
            params = 0
            if hasattr(layer, 'get_params'):
                layer_params = layer.get_params()
                for key, value in layer_params.items():
                    if key not in ['running_mean', 'running_var']:  # Exclude non-trainable
                        params += np.prod(value.shape) if hasattr(value, 'shape') else 0
            
            total_params += params
            if hasattr(layer, 'trainable') and not layer.trainable:
                pass
            else:
                trainable_params += params
            
            print(f"{layer_type:<30} {output_shape:<20} {params:<15,}")
        
        print("=" * 70)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("=" * 70)
    
    def save(self, filepath):
        """
        Save model to file.
        
        Parameters
        ----------
        filepath : str
            Path to save model (will save as .json + .npz)
        """
        import json
        
        # Model architecture
        config = {
            'class_name': self.__class__.__name__,
            'config': {
                'layers': []
            }
        }
        
        weights_dict = {}
        
        for i, layer in enumerate(self.layers):
            layer_config = {
                'class_name': layer.__class__.__name__,
                'config': layer.__dict__.copy() if hasattr(layer, '__dict__') else {}
            }
            
            # Remove non-serializable items
            for key in ['weights', 'bias', 'gamma', 'beta', 'grad_weights', 'grad_bias']:
                layer_config['config'].pop(key, None)
            
            config['config']['layers'].append(layer_config)
            
            # Save weights
            if hasattr(layer, 'get_params'):
                params = layer.get_params()
                for key, value in params.items():
                    weights_dict[f'layer_{i}_{key}'] = value
        
        # Save architecture as JSON
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save weights as NPZ
        weights_path = filepath.replace('.json', '_weights.npz')
        np.savez(weights_path, **weights_dict)
    
    @classmethod
    def load(cls, filepath):
        """
        Load model from file.
        
        Parameters
        ----------
        filepath : str
            Path to model JSON file
            
        Returns
        -------
        model : Sequential
            Loaded model
        """
        import json
        
        # Load architecture
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Load weights
        weights_path = filepath.replace('.json', '_weights.npz')
        weights_data = np.load(weights_path)
        
        # Reconstruct model
        model = cls()
        
        for i, layer_config in enumerate(config['config']['layers']):
            layer_class = layer_config['class_name']
            # ... reconstruct layer
            # (Implementation details omitted for brevity)
        
        return model
    
    def __repr__(self):
        return f"Sequential(layers={len(self.layers)})"
```

---

### Phase 7: Classification Support (Week 4, Days 1-2)

**Objective:** Add classification metrics and utilities.

#### Files to Create/Modify:
- `lynxlearn/metrics/_classification.py`
- `lynxlearn/metrics/__init__.py` (update)

#### Classification Metrics Implementation:

```python
# lynxlearn/metrics/_classification.py

import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    
    Parameters
    ----------
    y_true : ndarray
        True labels (class indices or one-hot)
    y_pred : ndarray
        Predicted labels (class indices or probabilities)
        
    Returns
    -------
    accuracy : float
        Classification accuracy (0.0 to 1.0)
    """
    # Handle one-hot encoding
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average='macro'):
    """
    Compute precision score.
    
    Precision = TP / (TP + FP)
    
    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    average : str
        Averaging method: 'macro', 'micro', 'weighted'
        
    Returns
    -------
    precision : float
        Precision score
    """
    # Handle one-hot encoding
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    weights = []
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0.0)
        
        weights.append(np.sum(y_true == cls))
    
    if average == 'macro':
        return np.mean(precisions)
    elif average == 'weighted':
        return np.average(precisions, weights=weights)
    elif average == 'micro':
        # Global precision
        tp_total = sum(np.sum((y_pred == cls) & (y_true == cls)) for cls in classes)
        fp_total = sum(np.sum((y_pred == cls) & (y_true != cls)) for cls in classes)
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0


def recall_score(y_true, y_pred, average='macro'):
    """
    Compute recall score.
    
    Recall = TP / (TP + FN)
    
    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    average : str
        Averaging method
        
    Returns
    -------
    recall : float
        Recall score
    """
    # Handle one-hot encoding
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    weights = []
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        else:
            recalls.append(0.0)
        
        weights.append(np.sum(y_true == cls))
    
    if average == 'macro':
        return np.mean(recalls)
    elif average == 'weighted':
        return np.average(recalls, weights=weights)
    elif average == 'micro':
        tp_total = sum(np.sum((y_pred == cls) & (y_true == cls)) for cls in classes)
        fn_total = sum(np.sum((y_pred != cls) & (y_true == cls)) for cls in classes)
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0


def f1_score(y_true, y_pred, average='macro'):
    """
    Compute F1 score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    average : str
        Averaging method
        
    Returns
    -------
    f1 : float
        F1 score
    """
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix.
    
    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    labels : list, optional
        List of labels
        
    Returns
    -------
    cm : ndarray
        Confusion matrix
    """
    # Handle one-hot encoding
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[true], label_to_idx[pred]] += 1
    
    return cm
```

---

### Phase 8: Visualization & Diagnostics (Week 4, Days 3-5)

**Objective:** Create neural network-specific visualization tools.

#### Files to Create:
- `lynxlearn/visualizations/_neural_net.py`

#### Key Visualizations:

```python
# lynxlearn/visualizations/_neural_net.py

import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history, metrics=None, figsize=(12, 4)):
    """
    Plot training history (loss and metrics over epochs).
    
    Parameters
    ----------
    history : dict
        Training history from model.train()
    metrics : list, optional
        Metrics to plot
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if metrics is None:
        metrics = []
    
    n_plots = 1 + len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot loss
    ax = axes[0]
    epochs = range(1, len(history['loss']) + 1)
    ax.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot metrics
    for i, metric in enumerate(metrics):
        ax = axes[i + 1]
        metric_name = metric if isinstance(metric, str) else metric.__name__
        
        if metric_name in history:
            ax.plot(epochs, history[metric_name], 'b-', 
                   label=f'Training {metric_name}', linewidth=2)
            if f'val_{metric_name}' in history:
                ax.plot(epochs, history[f'val_{metric_name}'], 'r-', 
                       label=f'Validation {metric_name}', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f'{metric_name.capitalize()} over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, classes=None, normalize=False, 
                          cmap='Blues', figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    cm : ndarray
        Confusion matrix
    classes : list, optional
        Class names
    normalize : bool
        Whether to normalize
    cmap : str
        Colormap
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    if classes is None:
        classes = range(cm.shape[0])
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    
    return fig


def plot_weight_distribution(model, figsize=(12, 8)):
    """
    Plot weight distribution for each layer.
    
    Parameters
    ----------
    model : BaseNeuralNetwork
        Trained model
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    layers_with_weights = [
        (i, layer) for i, layer in enumerate(model.layers)
        if hasattr(layer, 'get_params') and layer.get_params()
    ]
    
    n_layers = len(layers_with_weights)
    if n_layers == 0:
        print("No layers with weights found.")
        return None
    
    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (i, layer) in enumerate(layers_with_weights):
        params = layer.get_params()
        weights = params.get('weights')
        
        if weights is not None:
            ax = axes[idx]
            ax.hist(weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'Layer {i}: {layer.__class__.__name__}')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean = np.mean(weights)
            std = np.std(weights)
            ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.4f}')
            ax.legend()
    
    # Hide extra subplots
    for idx in range(len(layers_with_weights), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Weight Distribution by Layer', fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_gradient_flow(model, figsize=(10, 6)):
    """
    Plot gradient flow through layers.
    
    Useful for detecting vanishing/exploding gradients.
    
    Parameters
    ----------
    model : BaseNeuralNetwork
        Model with computed gradients
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    grad_averages = []
    layer_names = []
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_gradients'):
            grads = layer.get_gradients()
            if grads:
                # Compute average gradient magnitude
                total_grad = np.concatenate([g.flatten() for g in grads.values()])
                avg_grad = np.mean(np.abs(total_grad))
                grad_averages.append(avg_grad)
                layer_names.append(f'Layer {i}')
    
    if not grad_averages:
        print("No gradients found. Run backward pass first.")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(layer_names, grad_averages, color='steelblue', alpha=0.7)
    
    # Color code by gradient magnitude
    for bar, grad in zip(bars, grad_averages):
        if grad < 1e-7:
            bar.set_color('red')  # Vanishing
        elif grad > 1e3:
            bar.set_color('orange')  # Exploding
        else:
            bar.set_color('steelblue')  # Normal
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Gradient Magnitude')
    ax.set_title('Gradient Flow Through Layers')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Normal'),
        Patch(facecolor='red', label='Vanishing (< 1e-7)'),
        Patch(facecolor='orange', label='Exploding (> 1e3)')
    ]
    ax.legend(handles=legend_elements)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig
```

---

### Phase 9: Serialization (Week 5, Days 1-3)

**Objective:** Implement model save/load functionality.

**Already covered in Phase 6 Sequential model implementation.**

Additional serialization utilities:

```python
# lynxlearn/neural_network/_serialization.py

import json
import numpy as np

def serialize_optimizer(optimizer):
    """Serialize optimizer to JSON-compatible dict."""
    config = {
        'class_name': optimizer.__class__.__name__,
        'config': {
            'learning_rate': optimizer.learning_rate,
        }
    }
    
    # Add optimizer-specific config
    if hasattr(optimizer, 'get_config'):
        config['config'].update(optimizer.get_config())
    
    return config


def deserialize_optimizer(config):
    """Deserialize optimizer from dict."""
    # Implementation
    pass


def save_weights(model, filepath):
    """
    Save model weights only.
    
    Parameters
    ----------
    model : BaseNeuralNetwork
        Model to save
    filepath : str
        Path to save weights (.npz)
    """
    weights_dict = {}
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_params'):
            params = layer.get_params()
            for key, value in params.items():
                weights_dict[f'layer_{i}_{key}'] = value
    
    np.savez(filepath, **weights_dict)


def load_weights(model, filepath):
    """
    Load weights into existing model.
    
    Parameters
    ----------
    model : BaseNeuralNetwork
        Model to load weights into
    filepath : str
        Path to weights file (.npz)
    """
    weights_data = np.load(filepath)
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'set_params'):
            params = {}
            for key in layer.get_params().keys():
                weight_key = f'layer_{i}_{key}'
                if weight_key in weights_data:
                    params[key] = weights_data[weight_key]
            
            if params:
                layer.set_params(params)
```

---

### Phase 10: Documentation & Testing (Week 5-6)

**Objective:** Comprehensive documentation and test suite.

#### Documentation Structure:

```
docs/
├── neural_networks/
│   ├── README.md
│   ├── quickstart.md
│   ├── layers.md
│   ├── activations.md
│   ├── optimizers.md
│   ├── losses.md
│   ├── regularization.md
│   ├── callbacks.md
│   ├── serialization.md
│   ├── troubleshooting.md
│   └── api/
│       ├── sequential.md
│       ├── layers/
│       ├── optimizers/
│       └── losses/
└── examples/
    ├── notebooks/
    │   ├── mnist_classification.ipynb
    │   ├── regression_example.ipynb
    │   ├── custom_layers.ipynb
    │   └── debugging.ipynb
    └── scripts/
        ├── compare_with_tensorflow.py
        └── benchmark.py
```

#### Test Structure:

```
tests/
├── neural_network/
│   ├── __init__.py
│   ├── test_layers.py
│   ├── test_activations.py
│   ├── test_optimizers.py
│   ├── test_losses.py
│   ├── test_model.py
│   ├── test_regularization.py
│   ├── test_callbacks.py
│   ├── test_serialization.py
│   ├── test_integration.py
│   └── test_utils.py
```

#### Sample Test File:

```python
# tests/neural_network/test_layers.py

import numpy as np
import pytest
from lynxlearn.neural_network.layers import Dense, Dropout, BatchNormalization
from lynxlearn.neural_network.activations import ReLU, Softmax


class TestDense:
    """Tests for Dense layer."""
    
    def test_initialization(self):
        """Test layer initialization."""
        layer = Dense(128, activation='relu', input_shape=(784,))
        assert layer.units == 128
        assert layer.activation == 'relu'
        assert layer.input_shape == (784,)
    
    def test_build(self):
        """Test layer building."""
        layer = Dense(64, input_shape=(32,))
        layer.build((None, 32))
        
        assert layer.built
        assert layer.weights.shape == (32, 64)
        assert layer.bias.shape == (64,)
    
    def test_forward_pass(self):
        """Test forward pass."""
        layer = Dense(10)
        layer.build((None, 5))
        
        X = np.random.randn(32, 5)
        output = layer.forward(X)
        
        assert output.shape == (32, 10)
    
    def test_backward_pass(self):
        """Test backward pass."""
        layer = Dense(10)
        layer.build((None, 5))
        
        X = np.random.randn(32, 5)
        output = layer.forward(X)
        
        grad_output = np.random.randn(32, 10)
        grad_input = layer.backward(grad_output)
        
        assert grad_input.shape == (32, 5)
        assert layer.grad_weights.shape == (5, 10)
        assert layer.grad_bias.shape == (10,)
    
    def test_gradient_check(self):
        """Test gradient computation with numerical gradient."""
        layer = Dense(5, activation=None)
        layer.build((None, 3))
        
        X = np.random.randn(10, 3)
        y = np.random.randn(10, 5)
        
        # Forward pass
        y_pred = layer.forward(X)
        
        # Analytical gradient
        loss_grad = 2 * (y_pred - y) / y.size
        layer.backward(loss_grad)
        
        # Numerical gradient (finite differences)
        epsilon = 1e-5
        numerical_grad = np.zeros_like(layer.weights)
        
        for i in range(layer.weights.shape[0]):
            for j in range(layer.weights.shape[1]):
                # f(x + epsilon)
                layer.weights[i, j] += epsilon
                y_pred_plus = layer.forward(X)
                loss_plus = np.mean((y - y_pred_plus) ** 2)
                
                # f(x - epsilon)
                layer.weights[i, j] -= 2 * epsilon
                y_pred_minus = layer.forward(X)
                loss_minus = np.mean((y - y_pred_minus) ** 2)
                
                # Restore
                layer.weights[i, j] += epsilon
                
                # Numerical gradient
                numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Compare
        relative_error = np.abs(layer.grad_weights - numerical_grad) / (
            np.abs(layer.grad_weights) + np.abs(numerical_grad) + 1e-8
        )
        
        assert np.max(relative_error) < 1e-5, f"Gradient check failed: {np.max(relative_error)}"


class TestDropout:
    """Tests for Dropout layer."""
    
    def test_training_mode(self):
        """Test dropout during training."""
        layer = Dropout(0.5, seed=42)
        layer.build((None, 10))
        
        X = np.ones((100, 10))
        output = layer.forward(X, training=True)
        
        # Should have approximately 50% zeros
        zero_fraction = np.mean(output == 0)
        assert 0.3 < zero_fraction < 0.7  # Allow some variance
    
    def test_inference_mode(self):
        """Test dropout during inference."""
        layer = Dropout(0.5, seed=42)
        layer.build((None, 10))
        
        X = np.ones((100, 10))
        output = layer.forward(X, training=False)
        
        # Should not modify input during inference
        np.testing.assert_array_equal(output, X)
    
    def test_scaling(self):
        """Test scaling during training."""
        layer = Dropout(0.5, seed=42)
        layer.build((None, 10))
        
        X = np.ones((1000, 10))
        output = layer.forward(X, training=True)
        
        # Expected value should be preserved
        expected_mean = 1.0  # Input mean
        actual_mean = np.mean(output[output != 0])  # Mean of non-zero values
        
        assert abs(actual_mean - 2.0) < 0.1  # Scaled by 1/(1-rate) = 2


class TestBatchNormalization:
    """Tests for BatchNormalization layer."""
    
    def test_normalization(self):
        """Test that output is normalized."""
        layer = BatchNormalization()
        layer.build((None, 10))
        
        X = np.random.randn(100, 10) * 10 + 5  # Mean ~5, std ~10
        output = layer.forward(X, training=True)
        
        # Check normalized
        assert np.abs(np.mean(output)) < 0.1
        assert np.abs(np.std(output) - 1.0) < 0.1
    
    def test_inference_uses_running_stats(self):
        """Test that inference uses running statistics."""
        layer = BatchNormalization(momentum=0.9)
        layer.build((None, 10))
        
        # Train on data with mean=5, std=2
        for _ in range(100):
            X = np.random.randn(32, 10) * 2 + 5
            layer.forward(X, training=True)
        
        # Inference with different data
        X_test = np.random.randn(10, 10) * 3 + 10
        output = layer.forward(X_test, training=False)
        
        # Should use running mean (~5) and std (~2), not test data stats
        assert not np.isclose(np.mean(output), 0, atol=0.1)
```

---

## Performance Optimization

### Vectorization Strategy

**Key Principle:** Never loop over samples. Use NumPy broadcasting.

```python
# ❌ BAD: Loop over samples
for i in range(n_samples):
    output[i] = X[i] @ weights + bias

# ✅ GOOD: Vectorized
output = X @ weights + bias
```

### Memory Optimization

1. **Clear intermediate values after backward pass**

```python
def backward(self, grad_output):
    # ... compute gradients ...
    
    # Clear cached values
    self.input_cache = None
    self.z = None
    
    return grad_input
```

2. **Use in-place operations where safe**

```python
# Instead of creating new arrays
params[key] = params[key] - learning_rate * grad

# Use in-place (when safe)
params[key] -= learning_rate * grad
```

### Numerical Stability

1. **Log-sum-exp trick for softmax**

```python
def log_softmax(x):
    """Numerically stable log softmax."""
    max_x = np.max(x, axis=1, keepdims=True)
    return x - max_x - np.log(np.sum(np.exp(x - max_x), axis=1, keepdims=True))
```

2. **Epsilon clipping**

```python
def safe_divide(a, b, epsilon=1e-8):
    """Safe division with epsilon."""
    return a / (b + epsilon)
```

3. **Gradient clipping**

```python
def clip_gradients(grads, max_norm=1.0):
    """Clip gradients by global norm."""
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
    
    if total_norm > max_norm:
        scale = max_norm / total_norm
        grads = {k: v * scale for k, v in grads.items()}
    
    return grads
```

---

## Testing Strategy

### Unit Tests

- Each layer has dedicated test class
- Test initialization, forward pass, backward pass
- Gradient checking for all layers
- Shape validation

### Integration Tests

- Full model training on simple datasets
- Compare with known solutions
- Test save/load functionality

### Performance Tests

- Benchmark against TensorFlow/PyTorch
- Memory usage profiling
- Training speed comparison

### Test Coverage Goals

- Code coverage: > 90%
- Branch coverage: > 85%
- All edge cases tested

---

## Documentation Plan

### API Documentation

- Every public method has complete docstring
- Type hints for all parameters
- Examples in docstrings

### Tutorials

1. **Getting Started with Neural Networks**
   - What is a neural network?
   - Building your first model
   - Training and evaluation

2. **Understanding Backpropagation**
   - Forward pass
   - Backward pass
   - Gradient flow

3. **Advanced Topics**
   - Custom layers
   - Custom optimizers
   - Debugging techniques

### Example Notebooks

- MNIST classification
- Regression example
- Custom layer development
- Training visualization
- Model comparison

---

## Benchmarking Strategy

### Comparison Targets

- **TensorFlow/Keras**: Main competitor
- **PyTorch**: Secondary competitor
- **Scikit-learn MLPRegressor**: Baseline

### Metrics

1. **Training Speed**
   - Time per epoch
   - Total training time
   - Batch processing speed

2. **Inference Speed**
   - Prediction time per sample
   - Batch prediction throughput

3. **Memory Usage**
   - Peak memory during training
   - Model size

4. **Accuracy**
   - Final loss value
   - Test accuracy
   - Convergence speed

### Benchmark Suite

```python
# benchmark/neural_network_benchmark.py

import time
import numpy as np

def benchmark_training(model_class, config, X, y, n_runs=5):
    """Benchmark model training."""
    times = []
    
    for _ in range(n_runs):
        model = model_class(**config)
        model.compile(optimizer='adam', loss='mse')
        
        start = time.time()
        model.train(X, y, epochs=10, batch_size=32, verbose=0)
        end = time.time()
        
        times.append(end - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def run_full_benchmark():
    """Run comprehensive benchmark suite."""
    results = {}
    
    # Test different network sizes
    sizes = [
        {'layers': [64, 32, 10], 'params': '~4K'},
        {'layers': [256, 128, 64, 10], 'params': '~40K'},
        {'layers': [512, 256, 128, 64, 10], 'params': '~170K'},
    ]
    
    for size in sizes:
        print(f"\nBenchmarking: {size['params']} parameters")
        
        # LynxLearn
        print("  LynxLearn...")
        lynxlearn_time = benchmark_training(create_lynxlearn_model, ...)
        
        # TensorFlow (if available)
        print("  TensorFlow...")
        tensorflow_time = benchmark_training(create_tensorflow_model, ...)
        
        results[size['params']] = {
            'LynxLearn': lynxlearn_time,
            'TensorFlow': tensorflow_time
        }
    
    # Print comparison
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    for size, data in results.items():
        print(f"\n{size}:")
        print(f"  LynxLearn:  {data['LynxLearn']['mean']:.4f}s (±{data['LynxLearn']['std']:.4f})")
        print(f"  TensorFlow: {data['TensorFlow']['mean']:.4f}s (±{data['TensorFlow']['std']:.4f})")
        speedup = data['TensorFlow']['mean'] / data['LynxLearn']['mean']
        print(f"  Speedup:    {speedup:.2f}x")
    print("="*70)
```

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Numerical instability | High | Medium | Comprehensive testing, stable implementations |
| Performance regression | Medium | Low | Benchmark suite, continuous profiling |
| Memory leaks | Medium | Medium | Memory profiling, proper cleanup |
| API breaking changes | High | Medium | Version compatibility, deprecation warnings |
| Integration issues | Medium | Low | Comprehensive integration tests |

### Project Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Scope creep | High | Medium | Strict phase boundaries, prioritized features |
| Documentation lag | Medium | Medium | Document as you code, review documentation |
| Testing gaps | High | Low | High coverage requirements, automated testing |
| Performance targets not met | High | Low | Early benchmarking, optimization focus |

---

## Success Metrics

### Performance Metrics

- ✅ Training speed: Match or exceed TensorFlow on CPU
- ✅ Memory efficiency: < 1.5x TensorFlow memory usage
- ✅ Accuracy: Match TensorFlow within 1% on standard benchmarks

### Quality Metrics

- ✅ Test coverage: > 90%
- ✅ Documentation coverage: 100% of public API
- ✅ Example coverage: All major use cases

### Usability Metrics

- ✅ API simplicity: New users can train a model in < 10 lines
- ✅ Error clarity: All error messages include suggestions
- ✅ Learning curve: Beginner tutorial completable in < 30 minutes

---

## Conclusion

This comprehensive plan provides a roadmap for implementing a professional-grade neural network framework in LynxLearn. The implementation maintains the library's core philosophy of being:

1. **Educational**: Clear code, extensive documentation, visual learning aids
2. **Beginner-Friendly**: Simple API, helpful errors, progressive disclosure
3. **Performant**: Optimized NumPy operations, competitive with TensorFlow
4. **Professional**: Advanced features, callbacks, serialization
5. **Extensible**: Modular architecture, easy to add new components

The phased approach allows for incremental development with working deliverables at each stage, enabling early testing and feedback integration.

---

**End of PLAN.md**