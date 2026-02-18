"""
Neural Network Demo script for LynxLearn.

This script demonstrates:
- Sequential model creation
- Dense layers with various activations
- SGD optimizer with momentum and Nesterov
- Loss functions (MSE, MAE, Huber)
- Binary and multi-class classification
- Regression with neural networks
- Training with validation
- Model evaluation and prediction
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lynxlearn import metrics
from lynxlearn.model_selection import train_test_split
from lynxlearn.neural_network import (
    SGD,
    Dense,
    HuberLoss,
    MeanAbsoluteError,
    MeanSquaredError,
    Sequential,
)
from lynxlearn.neural_network.initializers import HeNormal, XavierNormal

np.random.seed(42)

print("=" * 70)
print("LynxLearn - Neural Network Demo")
print("=" * 70)


# ============================================
# 1. Simple Regression with Neural Network
# ============================================
print("\n" + "=" * 70)
print("1. Simple Regression: Learning y = 3x + 5")
print("=" * 70)

# Generate linear data with noise
n_samples = 200
X_reg = np.random.randn(n_samples, 1) * 2
y_reg = 3 * X_reg.flatten() + 5 + np.random.randn(n_samples) * 0.5

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"   Data shape: X={X_reg.shape}, y={y_reg.shape}")
print(f"   True function: y = 3x + 5")

# Create simple neural network (single layer for linear regression)
model_reg = Sequential([Dense(1, input_shape=(1,))])
model_reg.compile(optimizer=SGD(learning_rate=0.1), loss=MeanSquaredError())

print("\n   Model Architecture:")
model_reg.summary()

# Train
print("\n   Training for 100 epochs...")
history_reg = model_reg.train(
    X_train_reg, y_train_reg, epochs=100, batch_size=32, verbose=0
)

# Evaluate
y_pred_reg = model_reg.predict(X_test_reg).flatten()
mse = metrics.mse(y_test_reg, y_pred_reg)
r2 = metrics.r2_score(y_test_reg, y_pred_reg)

print(f"\n   Results:")
print(f"   - Final Loss: {history_reg['loss'][-1]:.6f}")
print(f"   - Test MSE: {mse:.6f}")
print(f"   - Test R²: {r2:.6f}")
print(f"   - Learned weight: {model_reg.layers[0].weights[0, 0]:.4f} (true: 3.0)")
print(f"   - Learned bias: {model_reg.layers[0].bias[0]:.4f} (true: 5.0)")


# ============================================
# 2. Non-linear Regression (MLP)
# ============================================
print("\n" + "=" * 70)
print("2. Non-linear Regression: Learning y = sin(x) + noise")
print("=" * 70)

# Generate non-linear data
X_sin = np.linspace(-3, 3, 300).reshape(-1, 1)
y_sin = np.sin(X_sin).flatten() + np.random.randn(300) * 0.1

X_train_sin, X_test_sin, y_train_sin, y_test_sin = train_test_split(
    X_sin, y_sin, test_size=0.2, random_state=42
)

print(f"   Data shape: X={X_sin.shape}, y={y_sin.shape}")
print(f"   True function: y = sin(x) + noise")

# Create MLP with hidden layers
model_mlp = Sequential(
    [
        Dense(64, activation="tanh", input_shape=(1,)),
        Dense(32, activation="tanh"),
        Dense(1),
    ]
)

model_mlp.compile(
    optimizer=SGD(learning_rate=0.05, momentum=0.9), loss=MeanSquaredError()
)

print(f"\n   Model Parameters: {model_mlp.count_params()}")

# Train
print("   Training for 200 epochs with momentum...")
history_mlp = model_mlp.train(
    X_train_sin, y_train_sin, epochs=200, batch_size=32, verbose=0
)

# Evaluate
y_pred_sin = model_mlp.predict(X_test_sin).flatten()
mse_sin = metrics.mse(y_test_sin, y_pred_sin)
r2_sin = metrics.r2_score(y_test_sin, y_pred_sin)

print(f"\n   Results:")
print(f"   - Initial Loss: {history_mlp['loss'][0]:.6f}")
print(f"   - Final Loss: {history_mlp['loss'][-1]:.6f}")
print(f"   - Test MSE: {mse_sin:.6f}")
print(f"   - Test R²: {r2_sin:.6f}")


# ============================================
# 3. Binary Classification
# ============================================
print("\n" + "=" * 70)
print("3. Binary Classification: XOR Problem")
print("=" * 70)

# XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y_xor = np.array([[0], [1], [1], [0]], dtype=np.float64)

print(f"   XOR inputs: {X_xor.tolist()}")
print(f"   XOR outputs: {y_xor.flatten().tolist()}")

# Create network for XOR (needs hidden layer)
model_xor = Sequential(
    [
        Dense(8, activation="tanh", input_shape=(2,)),
        Dense(1, activation="sigmoid"),
    ]
)

model_xor.compile(
    optimizer=SGD(learning_rate=0.5, momentum=0.9), loss=MeanSquaredError()
)

print(f"\n   Model Parameters: {model_xor.count_params()}")

# Train (XOR needs more epochs)
print("   Training for 500 epochs...")
history_xor = model_xor.train(X_xor, y_xor, epochs=500, batch_size=4, verbose=0)

# Predictions
y_pred_xor = model_xor.predict(X_xor)
y_pred_classes = (y_pred_xor > 0.5).astype(int)

print(f"\n   Results:")
print(f"   - Final Loss: {history_xor['loss'][-1]:.6f}")
print(f"   - Predictions: {y_pred_xor.flatten().round(3).tolist()}")
print(f"   - Predicted Classes: {y_pred_classes.flatten().tolist()}")
print(f"   - Actual Classes: {y_xor.flatten().tolist()}")
accuracy = np.mean(y_pred_classes.flatten() == y_xor.flatten())
print(f"   - Accuracy: {accuracy * 100:.1f}%")


# ============================================
# 4. Multi-class Classification
# ============================================
print("\n" + "=" * 70)
print("4. Multi-class Classification: Synthetic Dataset")
print("=" * 70)

# Generate 3-class dataset
n_per_class = 100
n_features = 4
n_classes = 3

X_multi = []
y_multi = []

for i in range(n_classes):
    center = np.random.randn(n_features) * 3
    X_class = center + np.random.randn(n_per_class, n_features) * 0.5
    X_multi.append(X_class)
    # One-hot encoding
    y_onehot = np.zeros((n_per_class, n_classes))
    y_onehot[:, i] = 1
    y_multi.append(y_onehot)

X_multi = np.vstack(X_multi)
y_multi = np.vstack(y_multi)

# Shuffle
shuffle_idx = np.random.permutation(len(X_multi))
X_multi = X_multi[shuffle_idx]
y_multi = y_multi[shuffle_idx]

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

print(f"   Data shape: X={X_multi.shape}, y={y_multi.shape}")
print(f"   Number of classes: {n_classes}")
print(f"   Features: {n_features}")

# Create multi-class classifier
model_multi = Sequential(
    [
        Dense(32, activation="relu", input_shape=(n_features,)),
        Dense(16, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ]
)

model_multi.compile(
    optimizer=SGD(learning_rate=0.1, momentum=0.9), loss=MeanSquaredError()
)

print(f"\n   Model Parameters: {model_multi.count_params()}")

# Train
print("   Training for 100 epochs...")
history_multi = model_multi.train(
    X_train_multi, y_train_multi, epochs=100, batch_size=16, verbose=0
)

# Evaluate
y_pred_multi = model_multi.predict(X_test_multi)
y_pred_classes_multi = np.argmax(y_pred_multi, axis=1)
y_true_classes_multi = np.argmax(y_test_multi, axis=1)
accuracy_multi = np.mean(y_pred_classes_multi == y_true_classes_multi)

print(f"\n   Results:")
print(f"   - Final Loss: {history_multi['loss'][-1]:.6f}")
print(f"   - Test Accuracy: {accuracy_multi * 100:.1f}%")

# Show some predictions
print(f"\n   Sample Predictions (first 5):")
print(f"   {'True Class':<12} {'Predicted':<12} {'Confidence':<12}")
print("   " + "-" * 36)
for i in range(5):
    true_class = y_true_classes_multi[i]
    pred_class = y_pred_classes_multi[i]
    confidence = y_pred_multi[i, pred_class]
    print(f"   {true_class:<12} {pred_class:<12} {confidence:.3f}")


# ============================================
# 5. Comparing Optimizers
# ============================================
print("\n" + "=" * 70)
print("5. Comparing SGD Variants")
print("=" * 70)

# Create same dataset for comparison
X_comp = np.random.randn(200, 5)
y_comp = np.sum(X_comp**2, axis=1, keepdims=True) + np.random.randn(200, 1) * 0.1

print(f"   Data shape: X={X_comp.shape}, y={y_comp.shape}")

optimizers = {
    "Vanilla SGD": SGD(learning_rate=0.01, momentum=0.0),
    "SGD + Momentum": SGD(learning_rate=0.01, momentum=0.9),
    "SGD + Nesterov": SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
    "SGD + Clipping": SGD(learning_rate=0.01, momentum=0.9, clipnorm=1.0),
}

results = {}

print(
    f"\n   {'Optimizer':<20} {'Initial Loss':<15} {'Final Loss':<15} {'Improvement':<15}"
)
print("   " + "-" * 65)

for name, optimizer in optimizers.items():
    model = Sequential(
        [
            Dense(16, activation="relu", input_shape=(5,)),
            Dense(8, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(optimizer=optimizer, loss=MeanSquaredError())

    history = model.train(X_comp, y_comp, epochs=100, verbose=0)

    initial_loss = history["loss"][0]
    final_loss = history["loss"][-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100

    results[name] = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "improvement": improvement,
    }

    print(
        f"   {name:<20} {initial_loss:<15.4f} {final_loss:<15.4f} {improvement:<14.1f}%"
    )


# ============================================
# 6. Comparing Loss Functions
# ============================================
print("\n" + "=" * 70)
print("6. Comparing Loss Functions")
print("=" * 70)

# Create data with some outliers
X_loss = np.random.randn(100, 3)
y_loss = 2 * X_loss[:, 0] + 3 * X_loss[:, 1] - X_loss[:, 2] + np.random.randn(100) * 0.5
# Add outliers
y_loss[::10] += np.random.randn(10) * 10

y_loss = y_loss.reshape(-1, 1)

print(f"   Data shape: X={X_loss.shape}, y={y_loss.shape}")
print(f"   Added outliers: every 10th sample")

losses = {
    "MSE": MeanSquaredError(),
    "MAE": MeanAbsoluteError(),
    "Huber (δ=1.0)": HuberLoss(delta=1.0),
}

print(f"\n   {'Loss Function':<20} {'Final Loss':<15} {'Test MSE':<15}")
print("   " + "-" * 50)

for name, loss_fn in losses.items():
    model = Sequential([Dense(1, input_shape=(3,))])
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss=loss_fn)

    model.train(X_loss, y_loss, epochs=100, verbose=0)

    y_pred = model.predict(X_loss)
    test_mse = metrics.mse(y_loss, y_pred)

    print(
        f"   {name:<20} {model.evaluate(X_loss, y_loss)['loss']:<15.4f} {test_mse:<15.4f}"
    )


# ============================================
# 7. Training with Validation
# ============================================
print("\n" + "=" * 70)
print("7. Training with Validation Split")
print("=" * 70)

# Generate larger dataset
X_val = np.random.randn(500, 10)
y_val = (
    np.sum(X_val[:, :5] ** 2, axis=1)
    - np.sum(X_val[:, 5:], axis=1)
    + np.random.randn(500) * 0.5
)
y_val = y_val.reshape(-1, 1)

print(f"   Data shape: X={X_val.shape}, y={y_val.shape}")

model_val = Sequential(
    [
        Dense(32, activation="relu", input_shape=(10,)),
        Dense(16, activation="relu"),
        Dense(1),
    ]
)

model_val.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss="mse")

print("   Training with 20% validation split...")
history_val = model_val.train(
    X_val, y_val, epochs=50, batch_size=32, validation_split=0.2, verbose=0
)

print(f"\n   Training History:")
print(f"   - Initial Train Loss: {history_val['loss'][0]:.4f}")
print(f"   - Final Train Loss: {history_val['loss'][-1]:.4f}")
print(f"   - Initial Val Loss: {history_val['val_loss'][0]:.4f}")
print(f"   - Final Val Loss: {history_val['val_loss'][-1]:.4f}")

# Check for overfitting
if history_val["val_loss"][-1] > history_val["loss"][-1] * 1.5:
    print(f"   - Status: Potential overfitting detected")
else:
    print(f"   - Status: Good generalization")


# ============================================
# 8. Weight Initialization Impact
# ============================================
print("\n" + "=" * 70)
print("8. Impact of Weight Initialization")
print("=" * 70)

# Deep network to show initialization impact
X_init = np.random.randn(200, 20)
y_init = np.sum(X_init[:, :10], axis=1, keepdims=True)

print(f"   Deep network (5 hidden layers)")
print(f"   Data shape: X={X_init.shape}")

initializers = {
    "He Normal (for ReLU)": HeNormal(seed=42),
    "Xavier Normal (for tanh)": XavierNormal(seed=42),
}

print(f"\n   {'Initializer':<30} {'Final Loss':<15} {'Converged':<12}")
print("   " + "-" * 57)

for name, initializer in initializers.items():
    # Determine activation based on initializer
    activation = "relu" if "He" in name else "tanh"

    model = Sequential()

    # First layer
    model.add(
        Dense(
            64,
            activation=activation,
            kernel_initializer=initializer,
            input_shape=(20,),
        )
    )
    # Hidden layers
    for _ in range(4):
        model.add(Dense(32, activation=activation, kernel_initializer=initializer))
    # Output
    model.add(Dense(1))

    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss="mse")

    history = model.train(X_init, y_init, epochs=100, verbose=0)

    converged = "Yes" if history["loss"][-1] < 1.0 else "No"

    print(f"   {name:<30} {history['loss'][-1]:<15.4f} {converged:<12}")


# ============================================
# 9. Model Summary and Information
# ============================================
print("\n" + "=" * 70)
print("9. Model Summary Example")
print("=" * 70)

model_summary = Sequential(
    [
        Dense(128, activation="relu", input_shape=(784,), name="hidden1"),
        Dense(64, activation="relu", name="hidden2"),
        Dense(32, activation="relu", name="hidden3"),
        Dense(10, activation="softmax", name="output"),
    ]
)

model_summary.compile(optimizer="sgd", loss="mse")

print("\n   Detailed Model Summary:")
model_summary.summary()

print(f"\n   Total Parameters: {model_summary.count_params():,}")

# Layer details
print("\n   Layer Details:")
for layer in model_summary.layers:
    params = layer.count_params()
    print(f"   - {layer.name}: {layer.units} units, {params:,} params")


# ============================================
# 10. Saving and Loading Model State
# ============================================
print("\n" + "=" * 70)
print("10. Model State Management")
print("=" * 70)

# Create and train a model
model_save = Sequential([Dense(16, activation="relu", input_shape=(5,)), Dense(1)])

model_save.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss="mse")

X_save = np.random.randn(100, 5)
y_save = np.sum(X_save, axis=1, keepdims=True)

model_save.train(X_save, y_save, epochs=50, verbose=0)

# Get weights
weights = model_save.get_weights()
print(f"   Number of layers with weights: {len(weights)}")

# Create new model and load weights
model_load = Sequential([Dense(16, activation="relu", input_shape=(5,)), Dense(1)])
model_load.compile(optimizer="sgd", loss="mse")
model_load.set_weights(weights)

# Verify predictions match
X_test_save = np.random.randn(10, 5)
pred_original = model_save.predict(X_test_save)
pred_loaded = model_load.predict(X_test_save)

print(
    f"   Predictions match after loading weights: {np.allclose(pred_original, pred_loaded)}"
)


# ============================================
# Summary
# ============================================
print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)

print("""
Key Takeaways:
1. Neural networks can learn both linear and non-linear relationships
2. Hidden layers with activations enable learning complex patterns
3. Momentum significantly improves convergence speed
4. Different loss functions are suited for different problems
5. Proper weight initialization helps training deep networks
6. Validation helps detect overfitting
7. Softmax activation is used for multi-class classification
8. Sigmoid activation is used for binary classification
""")

print("For more information, visit: https://github.com/notlousybook/LynxLearn")
