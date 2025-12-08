#!/usr/bin/env python3
"""
Debug script to validate MLP inference with the same weights and inputs as mlp_iris
"""

import numpy as np

# Weights from mlp_iris/main.cc (quantization bit = 5)
mlp_fc1_weight = np.array([
    [-1, 7, -8, -5], [-4, 5, -9, -8], [9, -7, 7, 2], [-3, 1, -4, 5],
    [3, 8, -6, -2], [6, -11, 7, -6], [10, -9, 7, 5], [-2, -1, 4, 4],
    [1, -1, -8, -16], [2, 3, -2, 5]
], dtype=np.int16)

mlp_fc2_weight = np.array([
    [10, 8, -8, 3, 3, -8, -9, -2, 11, 0],
    [-8, -7, -1, 2, -5, 12, -5, -3, 11, -2],
    [-2, -2, 6, 0, -6, 6, 10, -4, -16, -1]
], dtype=np.int16)

# Test inputs (first 5 samples for quick check)
iris_test_X = np.array([
    [16, -2, 10, -4], [10, -1, 9, -4], [5, -2, -8, -12],
    [10, -5, 4, -7], [8, -2, 3, -8]
], dtype=np.int16)

iris_test_Y = np.array([2, 2, 0, 1, 1], dtype=np.int16)

# Forward pass
print("="*60)
print("MLP Iris Debug - Python Reference Implementation")
print("="*60)

# FC1: (5, 4) @ (4, 10).T -> (5, 10)
y_fc1 = iris_test_X @ mlp_fc1_weight.T
print(f"\nFC1 output (first sample):\n{y_fc1[0]}")

# ReLU
y_relu = np.maximum(0, y_fc1)
print(f"\nReLU output (first sample):\n{y_relu[0]}")

# FC2: (5, 10) @ (10, 3).T -> (5, 3)
y_fc2 = y_relu @ mlp_fc2_weight.T
print(f"\nFC2 output (logits, first 5 samples):")
for i in range(5):
    print(f"  Sample {i}: {y_fc2[i]} -> argmax = {np.argmax(y_fc2[i])}, true = {iris_test_Y[i]}")

# Predictions
y_pred = np.argmax(y_fc2, axis=1)
accuracy = np.mean(y_pred == iris_test_Y)

print(f"\nPredictions: {y_pred}")
print(f"Ground truth: {iris_test_Y}")
print(f"Accuracy: {accuracy*100:.2f}%")

# Check if all predictions are class 2
if np.all(y_pred == 2):
    print("\n⚠️  WARNING: All predictions are class 2!")
    print("This matches the FIONA behavior - investigating further...")

    # Analyze FC2 output distribution
    print("\nFC2 output statistics:")
    print(f"  Class 0 logits (mean): {np.mean(y_fc2[:, 0]):.2f}")
    print(f"  Class 1 logits (mean): {np.mean(y_fc2[:, 1]):.2f}")
    print(f"  Class 2 logits (mean): {np.mean(y_fc2[:, 2]):.2f}")

    print("\nFC2 weight norms:")
    print(f"  Class 0 weight norm: {np.linalg.norm(mlp_fc2_weight[0]):.2f}")
    print(f"  Class 1 weight norm: {np.linalg.norm(mlp_fc2_weight[1]):.2f}")
    print(f"  Class 2 weight norm: {np.linalg.norm(mlp_fc2_weight[2]):.2f}")
