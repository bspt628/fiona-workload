/**
 * @file main.cc
 * @brief MLP Inference on MNIST Dataset with FP32 (Floating Point 32-bit)
 *
 * This version uses 32-bit floating point arithmetic to avoid the quantization
 * overflow issues that occur with int16 in the standard MNIST workload.
 *
 * Pre-trained MLP model for MNIST handwritten digit classification.
 * Weights are trained with PyTorch and converted to FP32.
 *
 * Architecture: 784 -> 256 -> 128 -> 64 -> 10
 * - Input:   784 features (28x28 grayscale image)
 * - Hidden1: 256 neurons + ReLU
 * - Hidden2: 128 neurons + ReLU
 * - Hidden3:  64 neurons + ReLU
 * - Output:   10 classes (digits 0-9)
 *
 * PyTorch Test Accuracy: 97.65%
 * Expected FP32 Accuracy: ~97% (matching PyTorch)
 *
 * @author FIONA Project
 * @date 2025-12-18
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fiona.h"

// Include pre-trained weights (int16 quantized)
#include "mnist_weights.h"

// Architecture parameters
#define BATCH_SIZE    MNIST_MLP_NUM_TEST  // 64
#define INPUT_SIZE    MNIST_MLP_INPUT_SIZE  // 784
#define HIDDEN1_SIZE  256
#define HIDDEN2_SIZE  128
#define HIDDEN3_SIZE  64
#define NUM_CLASSES   10

// ============================================================
// FP32 Weight and activation buffers
// ============================================================

// Dequantized weights (converted from int16 to float)
static float w1_fp32[HIDDEN1_SIZE][INPUT_SIZE];
static float b1_fp32[HIDDEN1_SIZE];
static float w2_fp32[HIDDEN2_SIZE][HIDDEN1_SIZE];
static float b2_fp32[HIDDEN2_SIZE];
static float w3_fp32[HIDDEN3_SIZE][HIDDEN2_SIZE];
static float b3_fp32[HIDDEN3_SIZE];
static float w4_fp32[NUM_CLASSES][HIDDEN3_SIZE];
static float b4_fp32[NUM_CLASSES];

// Input data (converted from int16 to float)
static float test_X_fp32[BATCH_SIZE][INPUT_SIZE];

// Intermediate activation buffers (FP32)
static float y_fc1[BATCH_SIZE][HIDDEN1_SIZE];
static float y_relu1[BATCH_SIZE][HIDDEN1_SIZE];
static float y_fc2[BATCH_SIZE][HIDDEN2_SIZE];
static float y_relu2[BATCH_SIZE][HIDDEN2_SIZE];
static float y_fc3[BATCH_SIZE][HIDDEN3_SIZE];
static float y_relu3[BATCH_SIZE][HIDDEN3_SIZE];
static float y_fc4[BATCH_SIZE][NUM_CLASSES];

// ============================================================
// FP32 Helper Functions
// ============================================================

/**
 * @brief Convert int16 quantized weights to FP32
 * @note Uses the scale factors from the quantization process
 *       Weights use W_SCALE, biases use B_SCALE (different bit depths)
 */
void dequantize_weights() {
    printf("Dequantizing weights from int16 to FP32...\n");

    // Dequantize FC1 weights and biases
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            w1_fp32[i][j] = (float)mnist_mlp_w1[i][j] * MNIST_MLP_W1_SCALE;
        }
        b1_fp32[i] = (float)mnist_mlp_b1[i] * MNIST_MLP_B1_SCALE;  // Use B1_SCALE
    }

    // Dequantize FC2 weights and biases
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            w2_fp32[i][j] = (float)mnist_mlp_w2[i][j] * MNIST_MLP_W2_SCALE;
        }
        b2_fp32[i] = (float)mnist_mlp_b2[i] * MNIST_MLP_B2_SCALE;  // Use B2_SCALE
    }

    // Dequantize FC3 weights and biases
    for (int i = 0; i < HIDDEN3_SIZE; i++) {
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            w3_fp32[i][j] = (float)mnist_mlp_w3[i][j] * MNIST_MLP_W3_SCALE;
        }
        b3_fp32[i] = (float)mnist_mlp_b3[i] * MNIST_MLP_B3_SCALE;  // Use B3_SCALE
    }

    // Dequantize FC4 weights and biases
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < HIDDEN3_SIZE; j++) {
            w4_fp32[i][j] = (float)mnist_mlp_w4[i][j] * MNIST_MLP_W4_SCALE;
        }
        b4_fp32[i] = (float)mnist_mlp_b4[i] * MNIST_MLP_B4_SCALE;  // Use B4_SCALE
    }

    // Dequantize input data
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            test_X_fp32[i][j] = (float)mnist_mlp_test_X[i][j] * MNIST_MLP_INPUT_SCALE;
        }
    }

    printf("  Dequantization complete.\n");
    printf("  Scale factors: W1=%.6f, B1=%.6f\n", MNIST_MLP_W1_SCALE, MNIST_MLP_B1_SCALE);
}

/**
 * @brief FP32 Linear layer: Y = X @ W^T + b
 */
void nn_linear_fp32(float *y, const float *w, const float *x, const float *b,
                    size_t in_features, size_t out_features, size_t batch_size) {
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t j = 0; j < out_features; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < in_features; k++) {
                sum += x[n * in_features + k] * w[j * in_features + k];
            }
            if (b != nullptr) {
                sum += b[j];
            }
            y[n * out_features + j] = sum;
        }
    }
}

/**
 * @brief FP32 ReLU activation: Y = max(0, X)
 */
void relu_fp32(float *y, const float *x, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows * cols; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

/**
 * @brief FP32 Argmax: find index of maximum value per row
 */
void argmax_fp32(int *indices, const float *data, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        int max_idx = 0;
        float max_val = data[i * cols];
        for (size_t j = 1; j < cols; j++) {
            if (data[i * cols + j] > max_val) {
                max_val = data[i * cols + j];
                max_idx = j;
            }
        }
        indices[i] = max_idx;
    }
}

/**
 * @brief Print layer statistics for debugging
 */
void print_layer_stats_fp32(const char* name, const float* data, int rows, int cols) {
    float min_val = data[0];
    float max_val = data[0];
    double sum = 0.0;

    for (int i = 0; i < rows * cols; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        sum += data[i];
    }

    printf("  %s: min=%.4f, max=%.4f, mean=%.4f\n",
           name, min_val, max_val, (float)(sum / (rows * cols)));
}

// ============================================================
// Main inference function
// ============================================================
int main() {
    printf("================================================================\n");
    printf("  FIONA MNIST MLP Inference (FP32 Version)\n");
    printf("================================================================\n");
    printf("  Architecture: %d -> %d -> %d -> %d -> %d\n",
           INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, HIDDEN3_SIZE, NUM_CLASSES);
    printf("  Batch size: %d\n", BATCH_SIZE);
    printf("  Data type: FP32 (32-bit floating point)\n");
    printf("  PyTorch Test Accuracy: 97.65%%\n");
    printf("================================================================\n\n");

    // Check photonic model environment variable
    const char* model = getenv("FIONA_PHOTONIC_MODEL");
    if (model) {
        printf("  Photonic Model: %s (Note: FP32 uses software computation)\n\n", model);
    } else {
        printf("  Photonic Model: N/A (FP32 software computation)\n\n");
    }

    // ============================================================
    // Dequantize weights and input data
    // ============================================================
    printf("=== Initialization ===\n");
    dequantize_weights();
    printf("\n");

    // ============================================================
    // Model Statistics
    // ============================================================
    printf("=== Model Statistics ===\n");
    int total_params = HIDDEN1_SIZE * INPUT_SIZE + HIDDEN1_SIZE +
                       HIDDEN2_SIZE * HIDDEN1_SIZE + HIDDEN2_SIZE +
                       HIDDEN3_SIZE * HIDDEN2_SIZE + HIDDEN3_SIZE +
                       NUM_CLASSES * HIDDEN3_SIZE + NUM_CLASSES;
    printf("  Total parameters: %d\n", total_params);
    printf("  FC1: %d x %d + %d bias\n", HIDDEN1_SIZE, INPUT_SIZE, HIDDEN1_SIZE);
    printf("  FC2: %d x %d + %d bias\n", HIDDEN2_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE);
    printf("  FC3: %d x %d + %d bias\n", HIDDEN3_SIZE, HIDDEN2_SIZE, HIDDEN3_SIZE);
    printf("  FC4: %d x %d + %d bias\n", NUM_CLASSES, HIDDEN3_SIZE, NUM_CLASSES);
    printf("\n");

    // ============================================================
    // Forward Pass using FP32 Software Computation
    // ============================================================
    printf("=== Forward Pass (FP32 Software) ===\n");

    // Layer 1: FC1 + ReLU
    printf("Layer 1: FC1 (%d x %d) + bias + ReLU\n", INPUT_SIZE, HIDDEN1_SIZE);
    nn_linear_fp32(&y_fc1[0][0], &w1_fp32[0][0], &test_X_fp32[0][0],
                   b1_fp32, INPUT_SIZE, HIDDEN1_SIZE, BATCH_SIZE);
    print_layer_stats_fp32("FC1 output", &y_fc1[0][0], BATCH_SIZE, HIDDEN1_SIZE);

    relu_fp32(&y_relu1[0][0], &y_fc1[0][0], BATCH_SIZE, HIDDEN1_SIZE);
    print_layer_stats_fp32("ReLU1 output", &y_relu1[0][0], BATCH_SIZE, HIDDEN1_SIZE);

    // Layer 2: FC2 + ReLU
    printf("Layer 2: FC2 (%d x %d) + bias + ReLU\n", HIDDEN1_SIZE, HIDDEN2_SIZE);
    nn_linear_fp32(&y_fc2[0][0], &w2_fp32[0][0], &y_relu1[0][0],
                   b2_fp32, HIDDEN1_SIZE, HIDDEN2_SIZE, BATCH_SIZE);
    print_layer_stats_fp32("FC2 output", &y_fc2[0][0], BATCH_SIZE, HIDDEN2_SIZE);

    relu_fp32(&y_relu2[0][0], &y_fc2[0][0], BATCH_SIZE, HIDDEN2_SIZE);
    print_layer_stats_fp32("ReLU2 output", &y_relu2[0][0], BATCH_SIZE, HIDDEN2_SIZE);

    // Layer 3: FC3 + ReLU
    printf("Layer 3: FC3 (%d x %d) + bias + ReLU\n", HIDDEN2_SIZE, HIDDEN3_SIZE);
    nn_linear_fp32(&y_fc3[0][0], &w3_fp32[0][0], &y_relu2[0][0],
                   b3_fp32, HIDDEN2_SIZE, HIDDEN3_SIZE, BATCH_SIZE);
    print_layer_stats_fp32("FC3 output", &y_fc3[0][0], BATCH_SIZE, HIDDEN3_SIZE);

    relu_fp32(&y_relu3[0][0], &y_fc3[0][0], BATCH_SIZE, HIDDEN3_SIZE);
    print_layer_stats_fp32("ReLU3 output", &y_relu3[0][0], BATCH_SIZE, HIDDEN3_SIZE);

    // Layer 4: FC4 (output layer)
    printf("Layer 4: FC4 (%d x %d) + bias - Output\n", HIDDEN3_SIZE, NUM_CLASSES);
    nn_linear_fp32(&y_fc4[0][0], &w4_fp32[0][0], &y_relu3[0][0],
                   b4_fp32, HIDDEN3_SIZE, NUM_CLASSES, BATCH_SIZE);
    print_layer_stats_fp32("FC4 output", &y_fc4[0][0], BATCH_SIZE, NUM_CLASSES);

    printf("\n");

    // ============================================================
    // Argmax and Results
    // ============================================================
    printf("=== Results ===\n");
    int y_pred[BATCH_SIZE];
    argmax_fp32(y_pred, &y_fc4[0][0], BATCH_SIZE, NUM_CLASSES);

    // Print first 20 predictions and labels
    printf("Predictions (first 20): [");
    for (int i = 0; i < 20 && i < BATCH_SIZE; i++) {
        printf("%d", y_pred[i]);
        if (i < 19) printf(", ");
    }
    printf("]\n");

    printf("True labels (first 20): [");
    for (int i = 0; i < 20 && i < BATCH_SIZE; i++) {
        printf("%d", mnist_mlp_test_Y[i]);
        if (i < 19) printf(", ");
    }
    printf("]\n");

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < BATCH_SIZE; i++) {
        if (y_pred[i] == mnist_mlp_test_Y[i]) {
            correct++;
        }
    }

    printf("\n[Result] FP32 Accuracy: %d / %d = %.2f%%\n",
           correct, BATCH_SIZE, (float)correct / BATCH_SIZE * 100.0f);
    printf("(PyTorch Reference: 97.65%%)\n");

    // ============================================================
    // Comparison with INT16 version
    // ============================================================
    printf("\n=== Comparison ===\n");
    printf("INT16 version: ~10.94%% (due to quantization overflow)\n");
    printf("FP32 version:  %.2f%% (expected to match PyTorch)\n",
           (float)correct / BATCH_SIZE * 100.0f);
    printf("\n");
    printf("Note: The INT16 version has overflow issues in FC1 layer\n");
    printf("because 784 inputs x 8-bit weights can exceed int16 range.\n");
    printf("FP32 avoids this by using 32-bit floating point arithmetic.\n");

    // ============================================================
    // Performance Summary
    // ============================================================
    printf("\n=== Performance Summary ===\n");

    // Calculate total MAC operations
    long long total_macs = 0;
    total_macs += (long long)BATCH_SIZE * INPUT_SIZE * HIDDEN1_SIZE;     // FC1
    total_macs += (long long)BATCH_SIZE * HIDDEN1_SIZE * HIDDEN2_SIZE;   // FC2
    total_macs += (long long)BATCH_SIZE * HIDDEN2_SIZE * HIDDEN3_SIZE;   // FC3
    total_macs += (long long)BATCH_SIZE * HIDDEN3_SIZE * NUM_CLASSES;    // FC4

    printf("Total MAC operations: %lld\n", total_macs);
    printf("  FC1: %d x %d x %d = %lld MACs\n",
           BATCH_SIZE, INPUT_SIZE, HIDDEN1_SIZE,
           (long long)BATCH_SIZE * INPUT_SIZE * HIDDEN1_SIZE);
    printf("  FC2: %d x %d x %d = %lld MACs\n",
           BATCH_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE,
           (long long)BATCH_SIZE * HIDDEN1_SIZE * HIDDEN2_SIZE);
    printf("  FC3: %d x %d x %d = %lld MACs\n",
           BATCH_SIZE, HIDDEN2_SIZE, HIDDEN3_SIZE,
           (long long)BATCH_SIZE * HIDDEN2_SIZE * HIDDEN3_SIZE);
    printf("  FC4: %d x %d x %d = %lld MACs\n",
           BATCH_SIZE, HIDDEN3_SIZE, NUM_CLASSES,
           (long long)BATCH_SIZE * HIDDEN3_SIZE * NUM_CLASSES);

    printf("\n================================================================\n");
    printf("  MNIST MLP Inference Complete! (FP32 Version)\n");
    printf("================================================================\n");

    DUMP_STAT;

    return 0;
}
