/**
 * @file main.cc
 * @brief INT16 Transformer test for FIONA RTL (Verilator)
 *
 * Tests INT16 Transformer operations that work with FIONA-V RTL.
 * Uses INT16 photonic instructions (DOTP, MVM) instead of FP32.
 *
 * IMPORTANT: All dimensions must be multiples of 32 for 64-byte alignment
 * (32 x int16_t = 64 bytes)
 *
 * Dimensions:
 * - seq_len = 4 tokens
 * - d_model = 32 (model dimension, must be multiple of 32)
 * - d_k = 32 (key/query dimension, must be multiple of 32)
 * - d_ff = 64 (feed-forward dimension)
 *
 * @author FIONA Project
 * @date 2025-12-22
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fiona.h"
#include "nn/transformer_int16.h"

// Test dimensions - ALL MUST BE MULTIPLES OF 32 for VLSU alignment
#define SEQ_LEN   4
#define D_MODEL   32
#define D_K       32   // Changed from 16 to 32 for alignment
#define D_FF      64
#define SCALE     128.0f

// Row stride for matrices (must be multiple of 32)
#define STRIDE_32  32
#define STRIDE_64  64

// ============================================================
// 64-byte aligned static arrays
// ============================================================

// Input/Output buffers
static int16_t input_buf[SEQ_LEN * D_MODEL] __attribute__((aligned(64)));
static int16_t output_buf[SEQ_LEN * D_MODEL] __attribute__((aligned(64)));

// Attention weights (row-major, rows padded to 32)
static int16_t Wq_buf[D_MODEL * D_K] __attribute__((aligned(64)));
static int16_t Wk_buf[D_MODEL * D_K] __attribute__((aligned(64)));
static int16_t Wv_buf[D_MODEL * D_K] __attribute__((aligned(64)));
static int16_t Wo_buf[D_K * D_MODEL] __attribute__((aligned(64)));

// FFN weights
static int16_t W1_buf[D_MODEL * D_FF] __attribute__((aligned(64)));
static int16_t W2_buf[D_FF * D_MODEL] __attribute__((aligned(64)));

// Temporary buffers
static int16_t temp1[SEQ_LEN * D_MODEL] __attribute__((aligned(64)));
static int16_t temp2[SEQ_LEN * D_FF] __attribute__((aligned(64)));

// ============================================================
// Test utilities
// ============================================================

static int test_count = 0;
static int pass_count = 0;

// Simple pseudo-random for reproducibility
static unsigned int rand_seed = 42;

int16_t pseudo_rand_int16() {
    rand_seed = rand_seed * 1103515245 + 12345;
    return (int16_t)((rand_seed >> 8) % 256 - 128);  // [-128, 127]
}

void init_random_int16(int16_t *arr, size_t len) {
    for (size_t i = 0; i < len; i++) {
        arr[i] = pseudo_rand_int16();
    }
}

void reset_seed(unsigned int seed) {
    rand_seed = seed;
}

void print_result(const char *test_name, bool passed) {
    test_count++;
    if (passed) {
        pass_count++;
        printf("[PASS] %s\n", test_name);
    } else {
        printf("[FAIL] %s\n", test_name);
    }
}

void print_vector_int16(const char *name, const int16_t *v, size_t len) {
    printf("%s: [", name);
    size_t show = (len < 8) ? len : 8;
    for (size_t i = 0; i < show; i++) {
        printf("%d", v[i]);
        if (i < show - 1) printf(", ");
    }
    if (len > 8) printf(", ...");
    printf("]\n");
}

// ============================================================
// Test: Quantization (no photonic ops)
// ============================================================

bool test_quantization() {
    printf("\n=== Test: Quantization ===\n");

    float in_f[] = {0.5f, -0.5f, 1.0f, -1.0f, 0.0f};
    int16_t out_i[5];
    float out_f[5];

    quantize_array(out_i, in_f, 5, SCALE);
    dequantize_array(out_f, out_i, 5, SCALE);

    printf("Original:    [0.5, -0.5, 1.0, -1.0, 0.0]\n");
    printf("Quantized:   [%d, %d, %d, %d, %d]\n",
           out_i[0], out_i[1], out_i[2], out_i[3], out_i[4]);
    printf("Dequantized: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
           out_f[0], out_f[1], out_f[2], out_f[3], out_f[4]);

    float max_error = 0.0f;
    for (int i = 0; i < 5; i++) {
        float err = fabsf(in_f[i] - out_f[i]);
        if (err > max_error) max_error = err;
    }
    printf("Max roundtrip error: %.6f\n", max_error);

    return max_error < 0.01f;
}

// ============================================================
// Test: ReLU INT16 (no photonic ops)
// ============================================================

bool test_relu_int16() {
    printf("\n=== Test: ReLU INT16 ===\n");

    int16_t in[] = {100, -50, 0, 200, -200, 50};
    int16_t out[6];

    relu_int16(out, in, 6);

    printf("Input:  [%d, %d, %d, %d, %d, %d]\n",
           in[0], in[1], in[2], in[3], in[4], in[5]);
    printf("Output: [%d, %d, %d, %d, %d, %d]\n",
           out[0], out[1], out[2], out[3], out[4], out[5]);

    bool pass = (out[0] == 100) && (out[1] == 0) && (out[2] == 0) &&
                (out[3] == 200) && (out[4] == 0) && (out[5] == 50);

    return pass;
}

// ============================================================
// Test: Softmax INT16 (no photonic ops)
// ============================================================

bool test_softmax_int16() {
    printf("\n=== Test: Softmax INT16 ===\n");

    int16_t in[4];
    int16_t out[4];

    in[0] = quantize(1.0f, SCALE);
    in[1] = quantize(2.0f, SCALE);
    in[2] = quantize(3.0f, SCALE);
    in[3] = quantize(4.0f, SCALE);

    softmax_int16(out, in, 4, SCALE, SCALE);

    printf("Input (dequant):  [%.2f, %.2f, %.2f, %.2f]\n",
           dequantize(in[0], SCALE), dequantize(in[1], SCALE),
           dequantize(in[2], SCALE), dequantize(in[3], SCALE));
    printf("Output (dequant): [%.3f, %.3f, %.3f, %.3f]\n",
           dequantize(out[0], SCALE), dequantize(out[1], SCALE),
           dequantize(out[2], SCALE), dequantize(out[3], SCALE));

    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += dequantize(out[i], SCALE);
    }
    printf("Sum: %.3f (should be ~1.0)\n", sum);

    return fabsf(sum - 1.0f) < 0.1f;
}

// ============================================================
// Test: Simple MVM with aligned static arrays
// ============================================================

// Static arrays for MVM test (32x32 matrix)
static int16_t mvm_weight[32 * 32] __attribute__((aligned(64)));
static int16_t mvm_input[32] __attribute__((aligned(64)));
static int16_t mvm_output[32] __attribute__((aligned(64)));

bool test_photonic_mvm_int16() {
    printf("\n=== Test: Photonic MVM INT16 (32x32) ===\n");

    // Initialize: identity-ish matrix
    memset(mvm_weight, 0, sizeof(mvm_weight));
    for (int i = 0; i < 32; i++) {
        mvm_weight[i * 32 + i] = 64;  // Diagonal
    }

    // Input: [32, 64, 96, ..., 32*32]
    for (int i = 0; i < 32; i++) {
        mvm_input[i] = (i + 1) * 10;
    }

    printf("Weight: 32x32 diagonal (diag=64)\n");
    printf("Input[0..3]: [%d, %d, %d, %d]\n",
           mvm_input[0], mvm_input[1], mvm_input[2], mvm_input[3]);

    // Call photonic MVM using the INT16 wrapper (uses tiled_mvm_strided internally)
    photonic_mvm_int16(mvm_output, mvm_weight, mvm_input, 32, 32);

    printf("Output[0..3]: [%d, %d, %d, %d]\n",
           mvm_output[0], mvm_output[1], mvm_output[2], mvm_output[3]);

    // With diagonal weight (64), output should be input * 64
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        int expected = mvm_input[i] * 64;
        if (mvm_output[i] != expected) {
            printf("Mismatch at [%d]: got %d, expected %d\n",
                   i, mvm_output[i], expected);
            pass = false;
        }
    }

    return pass;
}

// ============================================================
// Test: Full Transformer Block with aligned arrays
// ============================================================

bool test_transformer_block() {
    printf("\n=== Test: Full Transformer Block ===\n");
    printf("Configuration:\n");
    printf("  seq_len=%d, d_model=%d, d_k=%d, d_ff=%d\n",
           SEQ_LEN, D_MODEL, D_K, D_FF);
    printf("  All dimensions are multiples of 32 (64-byte aligned)\n");

    // Initialize weights
    reset_seed(789);
    init_random_int16(input_buf, SEQ_LEN * D_MODEL);
    init_random_int16(Wq_buf, D_MODEL * D_K);
    init_random_int16(Wk_buf, D_MODEL * D_K);
    init_random_int16(Wv_buf, D_MODEL * D_K);
    init_random_int16(Wo_buf, D_K * D_MODEL);
    init_random_int16(W1_buf, D_MODEL * D_FF);
    init_random_int16(W2_buf, D_FF * D_MODEL);

    printf("\nInput[0] (first 8): ");
    print_vector_int16("", input_buf, 8);

    // Run transformer block
    transformer_block_int16(output_buf, input_buf,
                            Wq_buf, Wk_buf, Wv_buf, Wo_buf,
                            W1_buf, W2_buf,
                            SEQ_LEN, D_MODEL, D_K, D_FF,
                            true, SCALE);

    printf("Output[0] (first 8): ");
    print_vector_int16("", output_buf, 8);

    // Check that output is not all zeros
    bool has_nonzero = false;
    for (size_t i = 0; i < SEQ_LEN * D_MODEL; i++) {
        if (output_buf[i] != 0) {
            has_nonzero = true;
            break;
        }
    }

    if (!has_nonzero) {
        printf("WARNING: Output is all zeros!\n");
    }

    return has_nonzero;
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("========================================\n");
    printf("FIONA INT16 Transformer Test\n");
    printf("========================================\n");
    printf("RTL-compatible INT16 operations\n");
    printf("64-byte aligned memory (VLSU requirement)\n");
    printf("All dimensions multiples of 32\n");
    printf("========================================\n");

    // Run tests (ordered from simple to complex)
    print_result("Quantization", test_quantization());
    print_result("ReLU INT16", test_relu_int16());
    print_result("Softmax INT16", test_softmax_int16());
    print_result("Photonic MVM INT16", test_photonic_mvm_int16());
    print_result("Transformer Block", test_transformer_block());

    // Summary
    printf("\n========================================\n");
    printf("Results: %d/%d tests passed\n", pass_count, test_count);
    printf("========================================\n");

    return (pass_count == test_count) ? 0 : 1;
}
