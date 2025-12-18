/**
 * @file main.cc
 * @brief Large-scale MLP Inference Benchmark for FIONA
 *
 * A significantly larger MLP model for benchmarking photonic accelerator
 * performance. Uses deterministic pseudo-random weights for reproducibility.
 *
 * Architecture: 128 -> 256 -> 128 -> 64 -> 10
 * - Input:   128 features
 * - Hidden1: 256 neurons (ReLU)
 * - Hidden2: 128 neurons (ReLU)
 * - Hidden3:  64 neurons (ReLU)
 * - Output:   10 classes
 *
 * Batch size: 64 samples
 *
 * Supports different photonic models via FIONA_PHOTONIC_MODEL environment variable.
 * Available models: ideal, noisy, quantized, mzi_nonlinear, all_effects
 *
 * @author FIONA Project
 * @date 2025-12-18
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fiona.h"
#include "utils/pprint.h"

// ============================================================
// Model Architecture Parameters
// ============================================================
#define BATCH_SIZE    64
#define NUM_FEATURES  128
#define HIDDEN1_SIZE  256
#define HIDDEN2_SIZE  128
#define HIDDEN3_SIZE  64
#define NUM_CLASSES   10

// ============================================================
// Pseudo-random number generator for reproducible weights
// Uses Linear Congruential Generator (LCG)
// ============================================================
static uint32_t lcg_state = 12345;

static inline int16_t lcg_rand_weight() {
    // LCG parameters (same as glibc)
    lcg_state = lcg_state * 1103515245 + 12345;
    // Map to [-16, 15] range (5-bit quantized weights)
    return (int16_t)((lcg_state >> 16) % 32) - 16;
}

static inline int16_t lcg_rand_input() {
    lcg_state = lcg_state * 1103515245 + 12345;
    // Map to [-32, 31] range for inputs
    return (int16_t)((lcg_state >> 16) % 64) - 32;
}

// Reset RNG to specific seed for reproducibility
static inline void lcg_seed(uint32_t seed) {
    lcg_state = seed;
}

// ============================================================
// Weight matrices (static allocation)
// ============================================================
static elem_t fc1_weight[HIDDEN1_SIZE][NUM_FEATURES];   // 256 x 128 = 32,768 params
static elem_t fc2_weight[HIDDEN2_SIZE][HIDDEN1_SIZE];   // 128 x 256 = 32,768 params
static elem_t fc3_weight[HIDDEN3_SIZE][HIDDEN2_SIZE];   //  64 x 128 =  8,192 params
static elem_t fc4_weight[NUM_CLASSES][HIDDEN3_SIZE];    //  10 x  64 =    640 params
                                                        // Total: 74,368 parameters

// ============================================================
// Input/Output buffers
// ============================================================
static elem_t input_data[BATCH_SIZE][NUM_FEATURES];
static elem_t true_labels[BATCH_SIZE];

// Intermediate activations
static elem_t y_fc1[BATCH_SIZE][HIDDEN1_SIZE];
static elem_t y_relu1[BATCH_SIZE][HIDDEN1_SIZE];
static elem_t y_fc2[BATCH_SIZE][HIDDEN2_SIZE];
static elem_t y_relu2[BATCH_SIZE][HIDDEN2_SIZE];
static elem_t y_fc3[BATCH_SIZE][HIDDEN3_SIZE];
static elem_t y_relu3[BATCH_SIZE][HIDDEN3_SIZE];
static elem_t y_fc4[BATCH_SIZE][NUM_CLASSES];

// ============================================================
// Initialize weights with deterministic pseudo-random values
// ============================================================
void init_weights() {
    printf("Initializing weights...\n");

    // FC1: 256 x 128
    lcg_seed(42);
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            fc1_weight[i][j] = lcg_rand_weight();
        }
    }
    printf("  FC1: %d x %d = %d parameters\n",
           HIDDEN1_SIZE, NUM_FEATURES, HIDDEN1_SIZE * NUM_FEATURES);

    // FC2: 128 x 256
    lcg_seed(123);
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            fc2_weight[i][j] = lcg_rand_weight();
        }
    }
    printf("  FC2: %d x %d = %d parameters\n",
           HIDDEN2_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE * HIDDEN1_SIZE);

    // FC3: 64 x 128
    lcg_seed(456);
    for (int i = 0; i < HIDDEN3_SIZE; i++) {
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            fc3_weight[i][j] = lcg_rand_weight();
        }
    }
    printf("  FC3: %d x %d = %d parameters\n",
           HIDDEN3_SIZE, HIDDEN2_SIZE, HIDDEN3_SIZE * HIDDEN2_SIZE);

    // FC4: 10 x 64
    lcg_seed(789);
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < HIDDEN3_SIZE; j++) {
            fc4_weight[i][j] = lcg_rand_weight();
        }
    }
    printf("  FC4: %d x %d = %d parameters\n",
           NUM_CLASSES, HIDDEN3_SIZE, NUM_CLASSES * HIDDEN3_SIZE);

    int total_params = HIDDEN1_SIZE * NUM_FEATURES +
                       HIDDEN2_SIZE * HIDDEN1_SIZE +
                       HIDDEN3_SIZE * HIDDEN2_SIZE +
                       NUM_CLASSES * HIDDEN3_SIZE;
    printf("  Total: %d parameters\n", total_params);
}

// ============================================================
// Initialize input data with deterministic pseudo-random values
// ============================================================
void init_input_data() {
    printf("Initializing input data (%d samples, %d features)...\n",
           BATCH_SIZE, NUM_FEATURES);

    lcg_seed(999);
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            input_data[i][j] = lcg_rand_input();
        }
        // Generate pseudo "true" labels for accuracy calculation
        true_labels[i] = (elem_t)(i % NUM_CLASSES);
    }
}

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
    printf("  FIONA Large-scale MLP Inference Benchmark\n");
    printf("================================================================\n");
    printf("  Architecture: %d -> %d -> %d -> %d -> %d\n",
           NUM_FEATURES, HIDDEN1_SIZE, HIDDEN2_SIZE, HIDDEN3_SIZE, NUM_CLASSES);
    printf("  Batch size: %d\n", BATCH_SIZE);
    printf("  Quantization: 5-bit weights, 6-bit inputs\n");
    printf("================================================================\n\n");

    // Check photonic model environment variable
    const char* model = getenv("FIONA_PHOTONIC_MODEL");
    if (model) {
        printf("  Photonic Model: %s\n\n", model);
    } else {
        printf("  Photonic Model: ideal (default)\n\n");
    }

    // ============================================================
    // Initialize weights and input data
    // ============================================================
    printf("=== Initialization ===\n");
    init_weights();
    init_input_data();
    printf("\n");

    // ============================================================
    // Forward Pass
    // ============================================================
    printf("=== Forward Pass (Photonic MVM) ===\n");

    // Layer 1: FC1 + ReLU
    printf("Layer 1: FC1 (%d x %d) + ReLU\n", NUM_FEATURES, HIDDEN1_SIZE);
    nn_linear(&y_fc1[0][0], &fc1_weight[0][0], &input_data[0][0],
              NUM_FEATURES, HIDDEN1_SIZE, BATCH_SIZE);
    print_layer_stats("FC1 output", &y_fc1[0][0], BATCH_SIZE, HIDDEN1_SIZE);

    tiled_matrix_relu(&y_relu1[0][0], &y_fc1[0][0], BATCH_SIZE, HIDDEN1_SIZE);
    print_layer_stats("ReLU1 output", &y_relu1[0][0], BATCH_SIZE, HIDDEN1_SIZE);

    // Layer 2: FC2 + ReLU
    printf("Layer 2: FC2 (%d x %d) + ReLU\n", HIDDEN1_SIZE, HIDDEN2_SIZE);
    nn_linear(&y_fc2[0][0], &fc2_weight[0][0], &y_relu1[0][0],
              HIDDEN1_SIZE, HIDDEN2_SIZE, BATCH_SIZE);
    print_layer_stats("FC2 output", &y_fc2[0][0], BATCH_SIZE, HIDDEN2_SIZE);

    tiled_matrix_relu(&y_relu2[0][0], &y_fc2[0][0], BATCH_SIZE, HIDDEN2_SIZE);
    print_layer_stats("ReLU2 output", &y_relu2[0][0], BATCH_SIZE, HIDDEN2_SIZE);

    // Layer 3: FC3 + ReLU
    printf("Layer 3: FC3 (%d x %d) + ReLU\n", HIDDEN2_SIZE, HIDDEN3_SIZE);
    nn_linear(&y_fc3[0][0], &fc3_weight[0][0], &y_relu2[0][0],
              HIDDEN2_SIZE, HIDDEN3_SIZE, BATCH_SIZE);
    print_layer_stats("FC3 output", &y_fc3[0][0], BATCH_SIZE, HIDDEN3_SIZE);

    tiled_matrix_relu(&y_relu3[0][0], &y_fc3[0][0], BATCH_SIZE, HIDDEN3_SIZE);
    print_layer_stats("ReLU3 output", &y_relu3[0][0], BATCH_SIZE, HIDDEN3_SIZE);

    // Layer 4: FC4 (output layer)
    printf("Layer 4: FC4 (%d x %d) - Output\n", HIDDEN3_SIZE, NUM_CLASSES);
    nn_linear(&y_fc4[0][0], &fc4_weight[0][0], &y_relu3[0][0],
              HIDDEN3_SIZE, NUM_CLASSES, BATCH_SIZE);
    print_layer_stats("FC4 output", &y_fc4[0][0], BATCH_SIZE, NUM_CLASSES);

    printf("\n");

    // ============================================================
    // Argmax and Results
    // ============================================================
    printf("=== Results ===\n");
    elem_t y_pred[BATCH_SIZE];
    matrix_vector_argmax(y_pred, &y_fc4[0][0], BATCH_SIZE, NUM_CLASSES);

    // Print first 10 predictions
    printf("Predictions (first 10): [");
    for (int i = 0; i < 10 && i < BATCH_SIZE; i++) {
        printf("%d", y_pred[i]);
        if (i < 9) printf(", ");
    }
    printf("]\n");

    printf("True labels (first 10): [");
    for (int i = 0; i < 10 && i < BATCH_SIZE; i++) {
        printf("%d", true_labels[i]);
        if (i < 9) printf(", ");
    }
    printf("]\n");

    // Calculate accuracy (note: random weights, so accuracy will be ~10%)
    elem_t bool_equal[BATCH_SIZE];
    vector_equal(bool_equal, y_pred, true_labels, BATCH_SIZE);
    elem_t correct;
    vector_sum(&correct, bool_equal, BATCH_SIZE);

    printf("\n[Result] Test Accuracy: %d / %d = %.2f%%\n",
           correct, BATCH_SIZE, (float)correct / BATCH_SIZE * 100.0f);
    printf("(Note: Random weights - accuracy ~10%% expected)\n");

    // ============================================================
    // Performance Summary
    // ============================================================
    printf("\n=== Performance Summary ===\n");

    // Calculate total MAC operations
    long long total_macs = 0;
    total_macs += (long long)BATCH_SIZE * NUM_FEATURES * HIDDEN1_SIZE;   // FC1
    total_macs += (long long)BATCH_SIZE * HIDDEN1_SIZE * HIDDEN2_SIZE;   // FC2
    total_macs += (long long)BATCH_SIZE * HIDDEN2_SIZE * HIDDEN3_SIZE;   // FC3
    total_macs += (long long)BATCH_SIZE * HIDDEN3_SIZE * NUM_CLASSES;    // FC4

    printf("Total MAC operations: %lld\n", total_macs);
    printf("  FC1: %d x %d x %d = %lld MACs\n",
           BATCH_SIZE, NUM_FEATURES, HIDDEN1_SIZE,
           (long long)BATCH_SIZE * NUM_FEATURES * HIDDEN1_SIZE);
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
    printf("  Large-scale MLP Inference Complete!\n");
    printf("================================================================\n");

    DUMP_STAT;

    return 0;
}
