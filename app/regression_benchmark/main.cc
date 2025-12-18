/**
 * @file main.cc
 * @brief Regression Benchmark for Photonic Model Evaluation
 *
 * This workload is designed to show clear differences between photonic models
 * by using:
 * 1. FP32 photonic MVM operations (not software emulation)
 * 2. Regression output (MSE/MAE metrics instead of classification accuracy)
 * 3. Multiple layers for noise accumulation
 * 4. Direct comparison between expected and actual outputs
 *
 * Architecture: 128 -> 256 -> 256 -> 128 -> 32
 * Total parameters: ~150,000
 *
 * @author FIONA Project
 * @date 2025-12-18
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fiona.h"

// Architecture parameters
#define BATCH_SIZE    32
#define INPUT_SIZE    128
#define HIDDEN1_SIZE  256
#define HIDDEN2_SIZE  256
#define HIDDEN3_SIZE  128
#define OUTPUT_SIZE   32

// Vector register size limit
#define VREG_SIZE     32

// ============================================================
// Weight matrices (initialized with deterministic pseudo-random values)
// ============================================================

static float w1[HIDDEN1_SIZE][INPUT_SIZE];
static float w2[HIDDEN2_SIZE][HIDDEN1_SIZE];
static float w3[HIDDEN3_SIZE][HIDDEN2_SIZE];
static float w4[OUTPUT_SIZE][HIDDEN3_SIZE];

// Input and output buffers
static float input_data[BATCH_SIZE][INPUT_SIZE];
static float expected_output[BATCH_SIZE][OUTPUT_SIZE];

// Intermediate activation buffers
static float act1[BATCH_SIZE][HIDDEN1_SIZE];
static float act2[BATCH_SIZE][HIDDEN2_SIZE];
static float act3[BATCH_SIZE][HIDDEN3_SIZE];
static float act4[BATCH_SIZE][OUTPUT_SIZE];

// Temporary buffers for tiled MVM
static float temp_vec[VREG_SIZE];
static float temp_mat[VREG_SIZE][VREG_SIZE];
static float temp_out[VREG_SIZE];

// ============================================================
// Deterministic pseudo-random number generator
// ============================================================

static unsigned int rand_seed = 12345;

float pseudo_random() {
    rand_seed = rand_seed * 1103515245 + 12345;
    return ((float)(rand_seed % 10000) / 10000.0f) - 0.5f;  // [-0.5, 0.5]
}

void init_weights() {
    printf("Initializing weights with deterministic pseudo-random values...\n");

    // Initialize W1: HIDDEN1_SIZE x INPUT_SIZE
    rand_seed = 11111;
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            w1[i][j] = pseudo_random() * 0.1f;  // Small values for stability
        }
    }

    // Initialize W2: HIDDEN2_SIZE x HIDDEN1_SIZE
    rand_seed = 22222;
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            w2[i][j] = pseudo_random() * 0.1f;
        }
    }

    // Initialize W3: HIDDEN3_SIZE x HIDDEN2_SIZE
    rand_seed = 33333;
    for (int i = 0; i < HIDDEN3_SIZE; i++) {
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            w3[i][j] = pseudo_random() * 0.1f;
        }
    }

    // Initialize W4: OUTPUT_SIZE x HIDDEN3_SIZE
    rand_seed = 44444;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN3_SIZE; j++) {
            w4[i][j] = pseudo_random() * 0.1f;
        }
    }

    int total_params = HIDDEN1_SIZE * INPUT_SIZE +
                       HIDDEN2_SIZE * HIDDEN1_SIZE +
                       HIDDEN3_SIZE * HIDDEN2_SIZE +
                       OUTPUT_SIZE * HIDDEN3_SIZE;
    printf("  Total parameters: %d\n", total_params);
}

void init_input_data() {
    printf("Initializing input data...\n");

    // Initialize input with deterministic values
    rand_seed = 55555;
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            input_data[b][i] = pseudo_random();
        }
    }
}

// ============================================================
// Software reference computation (for expected output)
// ============================================================

void compute_expected_output_software() {
    printf("Computing expected output (software reference)...\n");

    // Temporary buffers for software computation
    static float sw_y1[BATCH_SIZE][HIDDEN1_SIZE];
    static float sw_y2[BATCH_SIZE][HIDDEN2_SIZE];
    static float sw_y3[BATCH_SIZE][HIDDEN3_SIZE];

    // Layer 1: y1 = W1 @ x
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            float sum = 0.0f;
            for (int j = 0; j < INPUT_SIZE; j++) {
                sum += w1[i][j] * input_data[b][j];
            }
            sw_y1[b][i] = sum > 0 ? sum : 0;  // ReLU
        }
    }

    // Layer 2: y2 = W2 @ y1
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            float sum = 0.0f;
            for (int j = 0; j < HIDDEN1_SIZE; j++) {
                sum += w2[i][j] * sw_y1[b][j];
            }
            sw_y2[b][i] = sum > 0 ? sum : 0;  // ReLU
        }
    }

    // Layer 3: y3 = W3 @ y2
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < HIDDEN3_SIZE; i++) {
            float sum = 0.0f;
            for (int j = 0; j < HIDDEN2_SIZE; j++) {
                sum += w3[i][j] * sw_y2[b][j];
            }
            sw_y3[b][i] = sum > 0 ? sum : 0;  // ReLU
        }
    }

    // Layer 4: output = W4 @ y3 (no activation)
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float sum = 0.0f;
            for (int j = 0; j < HIDDEN3_SIZE; j++) {
                sum += w4[i][j] * sw_y3[b][j];
            }
            expected_output[b][i] = sum;
        }
    }
}

// ============================================================
// FP32 Photonic MVM (tiled)
// ============================================================

/**
 * @brief Single tile MVM using FP32 photonic instruction
 * @note Uses MVM_FP32 instruction which calls Python photonic model
 */
void photonic_mvm_tile(float *out, const float *mat, const float *vec,
                       int out_size, int in_size) {
    // Prepare input vector (pad to VREG_SIZE)
    for (int i = 0; i < VREG_SIZE; i++) {
        temp_vec[i] = (i < in_size) ? vec[i] : 0.0f;
    }

    // Prepare weight matrix (pad to VREG_SIZE x VREG_SIZE)
    for (int i = 0; i < VREG_SIZE; i++) {
        for (int j = 0; j < VREG_SIZE; j++) {
            if (i < out_size && j < in_size) {
                temp_mat[i][j] = mat[i * in_size + j];
            } else {
                temp_mat[i][j] = 0.0f;
            }
        }
    }

    // Set vector length
    size_t vlen = VREG_SIZE;
    SET_VLEN_FP32(vlen);

    // Load input vector to FP32 vector register 1
    LOAD_V_FP32(1, temp_vec);

    // Set weight matrix
    SET_MAT_FP32(&temp_mat[0][0]);

    // Execute MVM (result in vector register 0)
    MVM_FP32(0, 1);

    // Store result
    STORE_V_FP32(0, temp_out);

    // Copy valid outputs
    for (int i = 0; i < out_size; i++) {
        out[i] = temp_out[i];
    }
}

/**
 * @brief Tiled MVM for arbitrary sizes using FP32 photonic
 */
void photonic_mvm_tiled(float *out, const float *mat, const float *vec,
                        int out_size, int in_size) {
    // Initialize output to zero
    for (int i = 0; i < out_size; i++) {
        out[i] = 0.0f;
    }

    // Tile over output dimension
    for (int out_tile = 0; out_tile < out_size; out_tile += VREG_SIZE) {
        int out_tile_size = (out_tile + VREG_SIZE <= out_size) ? VREG_SIZE : (out_size - out_tile);

        // Tile over input dimension (accumulate partial results)
        for (int in_tile = 0; in_tile < in_size; in_tile += VREG_SIZE) {
            int in_tile_size = (in_tile + VREG_SIZE <= in_size) ? VREG_SIZE : (in_size - in_tile);

            // Prepare tile inputs
            float tile_vec[VREG_SIZE] = {0};
            float tile_mat[VREG_SIZE * VREG_SIZE] = {0};
            float tile_out[VREG_SIZE] = {0};

            // Copy input vector tile
            for (int i = 0; i < in_tile_size; i++) {
                tile_vec[i] = vec[in_tile + i];
            }

            // Copy weight matrix tile
            for (int i = 0; i < out_tile_size; i++) {
                for (int j = 0; j < in_tile_size; j++) {
                    tile_mat[i * VREG_SIZE + j] = mat[(out_tile + i) * in_size + (in_tile + j)];
                }
            }

            // Execute photonic MVM on tile
            photonic_mvm_tile(tile_out, tile_mat, tile_vec, out_tile_size, in_tile_size);

            // Accumulate results
            for (int i = 0; i < out_tile_size; i++) {
                out[out_tile + i] += tile_out[i];
            }
        }
    }
}

/**
 * @brief ReLU activation
 */
void relu_fp32(float *out, const float *in, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = (in[i] > 0) ? in[i] : 0.0f;
    }
}

// ============================================================
// Forward pass using photonic MVM
// ============================================================

void forward_pass_photonic() {
    printf("\n=== Forward Pass (Photonic FP32 MVM) ===\n");

    // Layer 1: act1 = ReLU(W1 @ x)
    printf("Layer 1: %d x %d -> %d\n", INPUT_SIZE, HIDDEN1_SIZE, HIDDEN1_SIZE);
    for (int b = 0; b < BATCH_SIZE; b++) {
        float temp_act1[HIDDEN1_SIZE];
        photonic_mvm_tiled(temp_act1, &w1[0][0], input_data[b], HIDDEN1_SIZE, INPUT_SIZE);
        relu_fp32(act1[b], temp_act1, HIDDEN1_SIZE);
    }

    // Layer 2: act2 = ReLU(W2 @ act1)
    printf("Layer 2: %d x %d -> %d\n", HIDDEN1_SIZE, HIDDEN2_SIZE, HIDDEN2_SIZE);
    for (int b = 0; b < BATCH_SIZE; b++) {
        float temp_act2[HIDDEN2_SIZE];
        photonic_mvm_tiled(temp_act2, &w2[0][0], act1[b], HIDDEN2_SIZE, HIDDEN1_SIZE);
        relu_fp32(act2[b], temp_act2, HIDDEN2_SIZE);
    }

    // Layer 3: act3 = ReLU(W3 @ act2)
    printf("Layer 3: %d x %d -> %d\n", HIDDEN2_SIZE, HIDDEN3_SIZE, HIDDEN3_SIZE);
    for (int b = 0; b < BATCH_SIZE; b++) {
        float temp_act3[HIDDEN3_SIZE];
        photonic_mvm_tiled(temp_act3, &w3[0][0], act2[b], HIDDEN3_SIZE, HIDDEN2_SIZE);
        relu_fp32(act3[b], temp_act3, HIDDEN3_SIZE);
    }

    // Layer 4: output = W4 @ act3 (no activation - regression output)
    printf("Layer 4: %d x %d -> %d\n", HIDDEN3_SIZE, OUTPUT_SIZE, OUTPUT_SIZE);
    for (int b = 0; b < BATCH_SIZE; b++) {
        photonic_mvm_tiled(act4[b], &w4[0][0], act3[b], OUTPUT_SIZE, HIDDEN3_SIZE);
    }
}

// ============================================================
// Metrics computation
// ============================================================

void compute_metrics() {
    printf("\n=== Regression Metrics ===\n");

    double mse = 0.0;
    double mae = 0.0;
    double max_error = 0.0;
    double sum_expected = 0.0;
    double sum_actual = 0.0;
    int total_outputs = BATCH_SIZE * OUTPUT_SIZE;

    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float expected = expected_output[b][i];
            float actual = act4[b][i];
            float error = actual - expected;

            mse += error * error;
            mae += fabs(error);
            if (fabs(error) > max_error) {
                max_error = fabs(error);
            }
            sum_expected += expected;
            sum_actual += actual;
        }
    }

    mse /= total_outputs;
    mae /= total_outputs;
    double rmse = sqrt(mse);

    printf("  Total output values: %d\n", total_outputs);
    printf("  Mean Squared Error (MSE): %.6f\n", mse);
    printf("  Root Mean Squared Error (RMSE): %.6f\n", rmse);
    printf("  Mean Absolute Error (MAE): %.6f\n", mae);
    printf("  Maximum Absolute Error: %.6f\n", max_error);
    printf("  Mean expected value: %.6f\n", sum_expected / total_outputs);
    printf("  Mean actual value: %.6f\n", sum_actual / total_outputs);

    // Relative error (percentage)
    double mean_abs_expected = 0.0;
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            mean_abs_expected += fabs(expected_output[b][i]);
        }
    }
    mean_abs_expected /= total_outputs;

    double relative_error = (mean_abs_expected > 0) ? (mae / mean_abs_expected * 100.0) : 0.0;
    printf("  Relative Error: %.2f%%\n", relative_error);

    // Print first few outputs for comparison
    printf("\nSample outputs (first 5 samples, first 4 outputs each):\n");
    for (int b = 0; b < 5 && b < BATCH_SIZE; b++) {
        printf("  Sample %d:\n", b);
        printf("    Expected: [");
        for (int i = 0; i < 4 && i < OUTPUT_SIZE; i++) {
            printf("%.4f", expected_output[b][i]);
            if (i < 3) printf(", ");
        }
        printf("]\n");
        printf("    Actual:   [");
        for (int i = 0; i < 4 && i < OUTPUT_SIZE; i++) {
            printf("%.4f", act4[b][i]);
            if (i < 3) printf(", ");
        }
        printf("]\n");
    }

    // Summary line for easy parsing
    printf("\n[Result] MSE=%.6f RMSE=%.6f MAE=%.6f MaxError=%.6f RelError=%.2f%%\n",
           mse, rmse, mae, max_error, relative_error);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("================================================================\n");
    printf("  FIONA Regression Benchmark (Photonic Model Evaluation)\n");
    printf("================================================================\n");
    printf("  Architecture: %d -> %d -> %d -> %d -> %d\n",
           INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, HIDDEN3_SIZE, OUTPUT_SIZE);
    printf("  Batch size: %d\n", BATCH_SIZE);
    printf("  Data type: FP32 with Photonic MVM\n");
    printf("================================================================\n\n");

    // Print photonic model info
    const char* model = getenv("FIONA_PHOTONIC_MODEL");
    if (model) {
        printf("  Photonic Model: %s\n", model);
    } else {
        printf("  Photonic Model: ideal (default)\n");
    }

    const char* noise_sigma = getenv("FIONA_NOISE_SIGMA");
    if (noise_sigma) {
        printf("  Noise Sigma: %s\n", noise_sigma);
    }
    printf("\n");

    // Initialize
    printf("=== Initialization ===\n");
    init_weights();
    init_input_data();

    // Compute expected output (software reference with ideal computation)
    compute_expected_output_software();

    // Forward pass using photonic MVM
    forward_pass_photonic();

    // Compute and display metrics
    compute_metrics();

    printf("\n================================================================\n");
    printf("  Regression Benchmark Complete\n");
    printf("================================================================\n");

    DUMP_STAT;

    return 0;
}
