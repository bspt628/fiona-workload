/**
 * @file main.cc
 * @brief Large-scale MNIST inference with FP32 Photonic MVM
 *
 * This workload is designed for paper-level validation with:
 * 1. 1000+ test samples (vs 64 in original)
 * 2. FP32 photonic MVM operations (DPI-C -> Python)
 * 3. Per-class accuracy statistics
 * 4. Confidence interval estimation
 *
 * Architecture: 784 -> 256 -> 128 -> 64 -> 10
 * PyTorch Test Accuracy: ~97%
 *
 * Usage:
 *   export FIONA_PHOTONIC_MODEL=ideal    # or noisy, mzi_realistic, all_effects
 *   spike --extension=fiona pk mlp_mnist_photonic.elf
 *
 * @author FIONA Project
 * @date 2025-12-18
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fiona.h"

// Include FP32 weights and test data
#include "mnist_fp32_weights.h"
#include "mnist_fp32_testdata.h"

// Architecture parameters
#define INPUT_SIZE    MNIST_INPUT_SIZE    // 784
#define HIDDEN1_SIZE  MNIST_HIDDEN1_SIZE  // 256
#define HIDDEN2_SIZE  MNIST_HIDDEN2_SIZE  // 128
#define HIDDEN3_SIZE  MNIST_HIDDEN3_SIZE  // 64
#define NUM_CLASSES   MNIST_OUTPUT_SIZE   // 10
#define NUM_SAMPLES   MNIST_NUM_TEST_SAMPLES  // 1000

// Vector register size limit for photonic MVM
#define VREG_SIZE     32

// ============================================================
// Temporary buffers for tiled MVM
// ============================================================
static float temp_vec[VREG_SIZE];
static float temp_mat[VREG_SIZE][VREG_SIZE];
static float temp_out[VREG_SIZE];

// Intermediate activation buffers (single sample at a time to save memory)
static float act_fc1[HIDDEN1_SIZE];
static float act_relu1[HIDDEN1_SIZE];
static float act_fc2[HIDDEN2_SIZE];
static float act_relu2[HIDDEN2_SIZE];
static float act_fc3[HIDDEN3_SIZE];
static float act_relu3[HIDDEN3_SIZE];
static float act_fc4[NUM_CLASSES];

// ============================================================
// FP32 Photonic MVM (tiled)
// ============================================================

/**
 * @brief Single tile MVM using FP32 photonic instruction
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

            // Copy weight matrix tile (row-major)
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
 * @brief FP32 Linear layer with bias: Y = W @ X + b
 */
void photonic_linear(float *out, const float *weight, const float *input,
                     const float *bias, int out_size, int in_size) {
    photonic_mvm_tiled(out, weight, input, out_size, in_size);

    // Add bias
    if (bias != NULL) {
        for (int i = 0; i < out_size; i++) {
            out[i] += bias[i];
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

/**
 * @brief Argmax: find index of maximum value
 */
int argmax_fp32(const float *data, int size) {
    int max_idx = 0;
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/**
 * @brief Softmax confidence (max logit vs second max)
 */
float compute_confidence(const float *logits, int size) {
    float max1 = -1e10f, max2 = -1e10f;
    for (int i = 0; i < size; i++) {
        if (logits[i] > max1) {
            max2 = max1;
            max1 = logits[i];
        } else if (logits[i] > max2) {
            max2 = logits[i];
        }
    }
    return max1 - max2;  // Margin between top 2 logits
}

// ============================================================
// Forward pass for single sample
// ============================================================

int forward_sample(const float *input) {
    // Layer 1: FC1 + ReLU
    photonic_linear(act_fc1, &mnist_w1[0][0], input, mnist_b1, HIDDEN1_SIZE, INPUT_SIZE);
    relu_fp32(act_relu1, act_fc1, HIDDEN1_SIZE);

    // Layer 2: FC2 + ReLU
    photonic_linear(act_fc2, &mnist_w2[0][0], act_relu1, mnist_b2, HIDDEN2_SIZE, HIDDEN1_SIZE);
    relu_fp32(act_relu2, act_fc2, HIDDEN2_SIZE);

    // Layer 3: FC3 + ReLU
    photonic_linear(act_fc3, &mnist_w3[0][0], act_relu2, mnist_b3, HIDDEN3_SIZE, HIDDEN2_SIZE);
    relu_fp32(act_relu3, act_fc3, HIDDEN3_SIZE);

    // Layer 4: FC4 (output logits)
    photonic_linear(act_fc4, &mnist_w4[0][0], act_relu3, mnist_b4, NUM_CLASSES, HIDDEN3_SIZE);

    return argmax_fp32(act_fc4, NUM_CLASSES);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("================================================================\n");
    printf("  FIONA MNIST Large-Scale Validation (Photonic FP32 MVM)\n");
    printf("================================================================\n");
    printf("  Architecture: %d -> %d -> %d -> %d -> %d\n",
           INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, HIDDEN3_SIZE, NUM_CLASSES);
    printf("  Test samples: %d\n", NUM_SAMPLES);
    printf("  Total parameters: %d\n", MNIST_TOTAL_PARAMS);
    printf("================================================================\n\n");

    // Print photonic model info
    const char* model = getenv("FIONA_PHOTONIC_MODEL");
    if (model) {
        printf("Photonic Model: %s\n", model);
    } else {
        printf("Photonic Model: ideal (default)\n");
    }

    const char* noise_sigma = getenv("FIONA_NOISE_SIGMA");
    if (noise_sigma) {
        printf("Noise Sigma: %s\n", noise_sigma);
    }
    printf("\n");

    // Per-class statistics
    int class_correct[NUM_CLASSES] = {0};
    int class_total[NUM_CLASSES] = {0};
    int total_correct = 0;

    // Confusion matrix (predicted x actual)
    int confusion[NUM_CLASSES][NUM_CLASSES] = {{0}};

    // Process each test sample
    printf("=== Processing %d test samples ===\n", NUM_SAMPLES);

    int progress_interval = NUM_SAMPLES / 10;
    if (progress_interval == 0) progress_interval = 1;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        // Progress indicator
        if ((i + 1) % progress_interval == 0 || i == NUM_SAMPLES - 1) {
            printf("  Progress: %d / %d (%.1f%%)\n",
                   i + 1, NUM_SAMPLES, 100.0f * (i + 1) / NUM_SAMPLES);
        }

        // Get ground truth
        int true_label = mnist_test_y[i];
        class_total[true_label]++;

        // Run inference
        int pred = forward_sample(mnist_test_x[i]);

        // Update statistics
        confusion[pred][true_label]++;
        if (pred == true_label) {
            class_correct[true_label]++;
            total_correct++;
        }
    }

    // ============================================================
    // Results
    // ============================================================
    printf("\n=== Results ===\n");

    // Overall accuracy
    float accuracy = 100.0f * total_correct / NUM_SAMPLES;
    printf("\nOverall Accuracy: %d / %d = %.2f%%\n", total_correct, NUM_SAMPLES, accuracy);

    // Per-class accuracy
    printf("\nPer-class Accuracy:\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        float class_acc = (class_total[i] > 0) ?
                          100.0f * class_correct[i] / class_total[i] : 0.0f;
        printf("  Class %d: %3d / %3d = %5.2f%%\n",
               i, class_correct[i], class_total[i], class_acc);
    }

    // Confusion matrix (compact)
    printf("\nConfusion Matrix (rows=predicted, cols=actual):\n");
    printf("     ");
    for (int j = 0; j < NUM_CLASSES; j++) {
        printf("%4d", j);
    }
    printf("\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("  %d: ", i);
        for (int j = 0; j < NUM_CLASSES; j++) {
            printf("%4d", confusion[i][j]);
        }
        printf("\n");
    }

    // 95% confidence interval (using normal approximation)
    // CI = p +/- 1.96 * sqrt(p * (1-p) / n)
    float p = (float)total_correct / NUM_SAMPLES;
    float ci_margin = 1.96f * sqrtf(p * (1.0f - p) / NUM_SAMPLES);
    printf("\n95%% Confidence Interval: %.2f%% +/- %.2f%% [%.2f%% - %.2f%%]\n",
           accuracy, ci_margin * 100.0f,
           (p - ci_margin) * 100.0f, (p + ci_margin) * 100.0f);

    // Summary line for easy parsing
    printf("\n[Result] Accuracy=%.2f%% (%d/%d) CI=[%.2f%%-%.2f%%]\n",
           accuracy, total_correct, NUM_SAMPLES,
           (p - ci_margin) * 100.0f, (p + ci_margin) * 100.0f);

    // Calculate MAC operations
    long long total_macs = 0;
    total_macs += (long long)NUM_SAMPLES * INPUT_SIZE * HIDDEN1_SIZE;
    total_macs += (long long)NUM_SAMPLES * HIDDEN1_SIZE * HIDDEN2_SIZE;
    total_macs += (long long)NUM_SAMPLES * HIDDEN2_SIZE * HIDDEN3_SIZE;
    total_macs += (long long)NUM_SAMPLES * HIDDEN3_SIZE * NUM_CLASSES;

    printf("\n=== Performance Summary ===\n");
    printf("Total samples: %d\n", NUM_SAMPLES);
    printf("Total MAC operations: %lld\n", total_macs);
    printf("MACs per sample: %lld\n", total_macs / NUM_SAMPLES);

    printf("\n================================================================\n");
    printf("  MNIST Large-Scale Validation Complete!\n");
    printf("================================================================\n");

    DUMP_STAT;

    return 0;
}
