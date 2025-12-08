/**
 * @file main.cc
 * @brief MLP Inference on Iris Dataset with Photonic Model Selection
 *
 * Uses the same weights as original mlp_iris to verify photonic model
 * switching mechanism. Supports different photonic models via the
 * FIONA_PHOTONIC_MODEL environment variable.
 *
 * Available models: ideal, noisy, quantized, mzi_nonlinear, all_effects
 *
 * @author FIONA Project
 * @date 2025-12-05
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fiona.h"
#include "utils/pprint.h"

// Architecture: 4 -> 10 -> 3 (same as original mlp_iris)
#define NUM_TEST     30
#define NUM_FEATURES 4
#define HIDDEN_SIZE  10
#define NUM_CLASSES  3

// Weights from original mlp_iris (quantization bit = 5)
static const elem_t mlp_fc1_weight[HIDDEN_SIZE][NUM_FEATURES] = {
    {-1, 7, -8, -5}, {-4, 5, -9, -8}, {9, -7, 7, 2}, {-3, 1, -4, 5},
    {3, 8, -6, -2}, {6, -11, 7, -6}, {10, -9, 7, 5}, {-2, -1, 4, 4},
    {1, -1, -8, -16}, {2, 3, -2, 5}
};

static const elem_t mlp_fc2_weight[NUM_CLASSES][HIDDEN_SIZE] = {
    {10, 8, -8, 3, 3, -8, -9, -2, 11, 0},
    {-8, -7, -1, 2, -5, 12, -5, -3, 11, -2},
    {-2, -2, 6, 0, -6, 6, 10, -4, -16, -1}
};

// Test data from original mlp_iris
static const elem_t iris_test_X[NUM_TEST][NUM_FEATURES] = {
    {16, -2, 10, -4}, {10, -1, 9, -4}, {5, -2, -8, -12}, {10, -5, 4, -7},
    {8, -2, 3, -8}, {14, -2, 10, -6}, {11, -2, 8, -5}, {10, -3, 5, -6},
    {10, -2, 5, -8}, {11, -1, 7, -4}, {5, -1, -7, -12}, {8, 1, -6, -12},
    {12, -2, 7, -5}, {16, -3, 13, -4}, {9, -3, 2, -9}, {8, 3, -7, -11},
    {8, -2, 4, -7}, {5, 0, -7, -12}, {7, 0, -7, -11}, {6, 1, -6, -11},
    {7, 1, -7, -12}, {5, -1, -7, -12}, {6, -1, -8, -12}, {13, -2, 9, -5},
    {8, -2, 2, -8}, {10, 0, 7, -4}, {6, -1, -6, -11}, {6, 2, -7, -12},
    {12, -4, 9, -6}, {11, -2, 9, -5}
};

static const elem_t iris_test_Y[NUM_TEST] = {
    2, 2, 0, 1, 1, 2, 2, 2, 1, 2, 0, 0, 2, 2, 1, 0, 1, 0, 0, 0,
    0, 0, 0, 2, 1, 2, 0, 0, 2, 2
};

int main() {
    printf("============================================\n");
    printf("  FIONA MLP Inference (Photonic Model Test)\n");
    printf("============================================\n");
    printf("  Architecture: %d -> %d -> %d\n", NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES);
    printf("  Quantization: 5-bit\n");
    printf("============================================\n\n");

    // Check photonic model environment variable
    const char* model = getenv("FIONA_PHOTONIC_MODEL");
    if (model) {
        printf("  Photonic Model: %s\n", model);
    } else {
        printf("  Photonic Model: ideal (default)\n");
    }
    printf("\n");

    // ============================================================
    // Run Inference using FIONA Photonic Hardware
    // ============================================================
    printf("=== Running Photonic Inference ===\n");

    // Layer 1: FC1 (photonic MVM)
    elem_t y_fc1[NUM_TEST][HIDDEN_SIZE];
    printf("Computing FC1 (photonic MVM)...\n");
    nn_linear(&y_fc1[0][0], &mlp_fc1_weight[0][0], &iris_test_X[0][0],
              NUM_FEATURES, HIDDEN_SIZE, NUM_TEST);

    // Debug: Print FC1 output
    printf("[debug] FC1 output (first 2 samples):\n");
    for (int i = 0; i < 2; i++) {
        printf("  Sample %d: [", i);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            printf("%d", y_fc1[i][j]);
            if (j < HIDDEN_SIZE - 1) printf(", ");
        }
        printf("]\n");
    }

    // ReLU activation
    elem_t y_relu[NUM_TEST][HIDDEN_SIZE];
    tiled_matrix_relu(&y_relu[0][0], &y_fc1[0][0], NUM_TEST, HIDDEN_SIZE);

    // Debug: Print ReLU output
    printf("[debug] ReLU output (first 2 samples):\n");
    for (int i = 0; i < 2; i++) {
        printf("  Sample %d: [", i);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            printf("%d", y_relu[i][j]);
            if (j < HIDDEN_SIZE - 1) printf(", ");
        }
        printf("]\n");
    }

    // Layer 2: FC2 (photonic MVM)
    elem_t y_fc2[NUM_TEST][NUM_CLASSES];
    printf("Computing FC2 (photonic MVM)...\n");
    nn_linear(&y_fc2[0][0], &mlp_fc2_weight[0][0], &y_relu[0][0],
              HIDDEN_SIZE, NUM_CLASSES, NUM_TEST);

    // Debug: Print FC2 output
    printf("[debug] FC2 output (first 5 samples):\n");
    for (int i = 0; i < 5; i++) {
        printf("  Sample %d: [%d, %d, %d]\n", i,
               y_fc2[i][0], y_fc2[i][1], y_fc2[i][2]);
    }

    // ============================================================
    // Argmax and Accuracy
    // ============================================================
    printf("\n=== Results ===\n");
    elem_t y_pred[NUM_TEST];
    matrix_vector_argmax(y_pred, &y_fc2[0][0], NUM_TEST, NUM_CLASSES);

    printf("Predictions: ");
    print_vec(y_pred, NUM_TEST);

    printf("True labels: ");
    print_vec(iris_test_Y, NUM_TEST);

    // Calculate accuracy
    elem_t bool_equal[NUM_TEST];
    vector_equal(bool_equal, y_pred, iris_test_Y, NUM_TEST);
    elem_t correct;
    vector_sum(&correct, bool_equal, NUM_TEST);

    printf("\n[Result] Test Accuracy: %d / %d = %.2f%%\n",
           correct, NUM_TEST, (float)correct / NUM_TEST * 100.0f);

    printf("\n============================================\n");
    printf("  Inference Complete!\n");
    printf("============================================\n");

    DUMP_STAT;

    return 0;
}
