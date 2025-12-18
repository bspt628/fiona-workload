/**
 * @file main.cc
 * @brief MLP Inference on MNIST Dataset with Photonic Backend
 *
 * Pre-trained MLP model for MNIST handwritten digit classification.
 * Weights are trained with PyTorch and quantized for FIONA photonic accelerator.
 *
 * Architecture: 784 -> 256 -> 128 -> 64 -> 10
 * - Input:   784 features (28x28 grayscale image)
 * - Hidden1: 256 neurons + ReLU
 * - Hidden2: 128 neurons + ReLU
 * - Hidden3:  64 neurons + ReLU
 * - Output:   10 classes (digits 0-9)
 *
 * PyTorch Test Accuracy: 97.65%
 * Quantization: 8-bit weights
 *
 * Supports different photonic models via FIONA_PHOTONIC_MODEL environment variable.
 * Available models: ideal, noisy, quantized, mzi_nonlinear, mzi_realistic, all_effects
 *
 * @author FIONA Project
 * @date 2025-12-18
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fiona.h"
#include "utils/pprint.h"

// Include pre-trained weights
#include "mnist_weights.h"

// Architecture parameters (from mnist_weights.h)
#define BATCH_SIZE    MNIST_MLP_NUM_TEST  // 64
#define INPUT_SIZE    MNIST_MLP_INPUT_SIZE  // 784
#define HIDDEN1_SIZE  256
#define HIDDEN2_SIZE  128
#define HIDDEN3_SIZE  64
#define NUM_CLASSES   10

// ============================================================
// Intermediate activation buffers (static allocation)
// ============================================================
static elem_t y_fc1[BATCH_SIZE][HIDDEN1_SIZE];
static elem_t y_relu1[BATCH_SIZE][HIDDEN1_SIZE];
static elem_t y_fc2[BATCH_SIZE][HIDDEN2_SIZE];
static elem_t y_relu2[BATCH_SIZE][HIDDEN2_SIZE];
static elem_t y_fc3[BATCH_SIZE][HIDDEN3_SIZE];
static elem_t y_relu3[BATCH_SIZE][HIDDEN3_SIZE];
static elem_t y_fc4[BATCH_SIZE][NUM_CLASSES];

// ============================================================
// Print layer statistics
// ============================================================
void print_layer_stats(const char* name, const elem_t* data, int rows, int cols) {
    elem_t min_val = data[0];
    elem_t max_val = data[0];
    int32_t sum = 0;

    for (int i = 0; i < rows * cols; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        sum += data[i];
    }

    printf("  %s: min=%d, max=%d, mean=%.2f\n",
           name, min_val, max_val, (float)sum / (rows * cols));
}

// ============================================================
// Main inference function
// ============================================================
int main() {
    printf("================================================================\n");
    printf("  FIONA MNIST MLP Inference (Pre-trained Model)\n");
    printf("================================================================\n");
    printf("  Architecture: %d -> %d -> %d -> %d -> %d\n",
           INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, HIDDEN3_SIZE, NUM_CLASSES);
    printf("  Batch size: %d\n", BATCH_SIZE);
    printf("  Quantization: %d-bit\n", MNIST_MLP_QUANT_BITS);
    printf("  PyTorch Test Accuracy: 97.65%%\n");
    printf("================================================================\n\n");

    // Check photonic model environment variable
    const char* model = getenv("FIONA_PHOTONIC_MODEL");
    if (model) {
        printf("  Photonic Model: %s\n\n", model);
    } else {
        printf("  Photonic Model: ideal (default)\n\n");
    }

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
    // Forward Pass using FIONA Photonic Hardware
    // ============================================================
    printf("=== Forward Pass (Photonic MVM) ===\n");

    // Layer 1: FC1 + ReLU
    printf("Layer 1: FC1 (%d x %d) + bias + ReLU\n", INPUT_SIZE, HIDDEN1_SIZE);
    nn_linear(&y_fc1[0][0], &mnist_mlp_w1[0][0], &mnist_mlp_test_X[0][0],
              mnist_mlp_b1, INPUT_SIZE, HIDDEN1_SIZE, BATCH_SIZE);
    print_layer_stats("FC1 output", &y_fc1[0][0], BATCH_SIZE, HIDDEN1_SIZE);

    tiled_matrix_relu(&y_relu1[0][0], &y_fc1[0][0], BATCH_SIZE, HIDDEN1_SIZE);
    print_layer_stats("ReLU1 output", &y_relu1[0][0], BATCH_SIZE, HIDDEN1_SIZE);

    // Layer 2: FC2 + ReLU
    printf("Layer 2: FC2 (%d x %d) + bias + ReLU\n", HIDDEN1_SIZE, HIDDEN2_SIZE);
    nn_linear(&y_fc2[0][0], &mnist_mlp_w2[0][0], &y_relu1[0][0],
              mnist_mlp_b2, HIDDEN1_SIZE, HIDDEN2_SIZE, BATCH_SIZE);
    print_layer_stats("FC2 output", &y_fc2[0][0], BATCH_SIZE, HIDDEN2_SIZE);

    tiled_matrix_relu(&y_relu2[0][0], &y_fc2[0][0], BATCH_SIZE, HIDDEN2_SIZE);
    print_layer_stats("ReLU2 output", &y_relu2[0][0], BATCH_SIZE, HIDDEN2_SIZE);

    // Layer 3: FC3 + ReLU
    printf("Layer 3: FC3 (%d x %d) + bias + ReLU\n", HIDDEN2_SIZE, HIDDEN3_SIZE);
    nn_linear(&y_fc3[0][0], &mnist_mlp_w3[0][0], &y_relu2[0][0],
              mnist_mlp_b3, HIDDEN2_SIZE, HIDDEN3_SIZE, BATCH_SIZE);
    print_layer_stats("FC3 output", &y_fc3[0][0], BATCH_SIZE, HIDDEN3_SIZE);

    tiled_matrix_relu(&y_relu3[0][0], &y_fc3[0][0], BATCH_SIZE, HIDDEN3_SIZE);
    print_layer_stats("ReLU3 output", &y_relu3[0][0], BATCH_SIZE, HIDDEN3_SIZE);

    // Layer 4: FC4 (output layer)
    printf("Layer 4: FC4 (%d x %d) + bias - Output\n", HIDDEN3_SIZE, NUM_CLASSES);
    nn_linear(&y_fc4[0][0], &mnist_mlp_w4[0][0], &y_relu3[0][0],
              mnist_mlp_b4, HIDDEN3_SIZE, NUM_CLASSES, BATCH_SIZE);
    print_layer_stats("FC4 output", &y_fc4[0][0], BATCH_SIZE, NUM_CLASSES);

    printf("\n");

    // ============================================================
    // Argmax and Results
    // ============================================================
    printf("=== Results ===\n");
    elem_t y_pred[BATCH_SIZE];
    matrix_vector_argmax(y_pred, &y_fc4[0][0], BATCH_SIZE, NUM_CLASSES);

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
    elem_t bool_equal[BATCH_SIZE];
    vector_equal(bool_equal, y_pred, mnist_mlp_test_Y, BATCH_SIZE);
    elem_t correct;
    vector_sum(&correct, bool_equal, BATCH_SIZE);

    printf("\n[Result] FIONA Accuracy: %d / %d = %.2f%%\n",
           correct, BATCH_SIZE, (float)correct / BATCH_SIZE * 100.0f);
    printf("(PyTorch Reference: 97.65%%)\n");

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
    printf("  MNIST MLP Inference Complete!\n");
    printf("================================================================\n");

    DUMP_STAT;

    return 0;
}
