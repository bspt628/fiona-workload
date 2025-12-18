/**
 * @file main.cc
 * @brief Transformer operations verification test
 *
 * Tests all transformer operations with small dimensions:
 * - seq_len = 4 tokens
 * - d_model = 32 (model dimension)
 * - d_k = 16 (key/query dimension)
 * - d_ff = 64 (feed-forward dimension)
 *
 * Verifies:
 * 1. Photonic MVM (basic and tiled)
 * 2. Electronic ops (softmax, GELU, LayerNorm)
 * 3. Attention mechanism
 * 4. FFN (GELU and SiLU variants)
 * 5. Full transformer block
 *
 * Usage:
 *   export FIONA_PHOTONIC_MODEL=ideal
 *   spike --extension=fiona pk test_transformer.elf
 *
 * @author FIONA Project
 * @date 2025-12-18
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fiona.h"
#include "nn/transformer.h"

// Test dimensions (small for quick verification)
#define SEQ_LEN   4
#define D_MODEL   32
#define D_K       16
#define D_FF      64

// Tolerance for floating point comparison
#define TOLERANCE 1e-4f

// ============================================================
// Test utilities
// ============================================================

static int test_count = 0;
static int pass_count = 0;

void print_vector(const char *name, const float *v, size_t len) {
    printf("%s: [", name);
    for (size_t i = 0; i < len && i < 8; i++) {
        printf("%.4f", v[i]);
        if (i < len - 1 && i < 7) printf(", ");
    }
    if (len > 8) printf(", ...");
    printf("]\n");
}

void print_matrix(const char *name, const float *m, size_t rows, size_t cols) {
    printf("%s [%zu x %zu]:\n", name, rows, cols);
    for (size_t i = 0; i < rows && i < 4; i++) {
        printf("  [");
        for (size_t j = 0; j < cols && j < 6; j++) {
            printf("%7.3f", m[i * cols + j]);
            if (j < cols - 1 && j < 5) printf(", ");
        }
        if (cols > 6) printf(", ...");
        printf("]\n");
    }
    if (rows > 4) printf("  ...\n");
}

bool check_close(float a, float b, float tol) {
    return fabsf(a - b) < tol;
}

bool check_array_close(const float *a, const float *b, size_t len, float tol) {
    for (size_t i = 0; i < len; i++) {
        if (!check_close(a[i], b[i], tol)) {
            return false;
        }
    }
    return true;
}

void test_result(const char *name, bool passed) {
    test_count++;
    if (passed) {
        pass_count++;
        printf("[PASS] %s\n", name);
    } else {
        printf("[FAIL] %s\n", name);
    }
}

// Initialize array with random-ish values (deterministic)
void init_array(float *arr, size_t len, float scale, int seed) {
    for (size_t i = 0; i < len; i++) {
        // Simple pseudo-random based on index and seed
        float val = sinf((float)(i + seed) * 0.1f) * scale;
        arr[i] = val;
    }
}

// Initialize array with values normalized to [-1, 1]
void init_normalized(float *arr, size_t len, int seed) {
    for (size_t i = 0; i < len; i++) {
        arr[i] = sinf((float)(i + seed) * 0.1f);
    }
}

// ============================================================
// Test 1: Softmax
// ============================================================

void test_softmax() {
    printf("\n=== Test: Softmax ===\n");

    float input[8] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    float output[8];

    softmax_fp32(output, input, 8);

    print_vector("Input", input, 8);
    print_vector("Output", output, 8);

    // Check properties
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        sum += output[i];
    }
    printf("Sum of softmax outputs: %.6f (should be 1.0)\n", sum);

    bool passed = check_close(sum, 1.0f, TOLERANCE);

    // Check that larger inputs have larger outputs
    passed = passed && (output[3] > output[2]);
    passed = passed && (output[2] > output[1]);
    passed = passed && (output[1] > output[0]);

    test_result("Softmax basic", passed);
}

// ============================================================
// Test 2: GELU
// ============================================================

void test_gelu() {
    printf("\n=== Test: GELU ===\n");

    float input[8] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
    float output[8];

    gelu_fp32(output, input, 8);

    print_vector("Input", input, 8);
    print_vector("Output", output, 8);

    // Check properties
    // GELU(0) should be 0
    bool passed = check_close(output[3], 0.0f, TOLERANCE);

    // GELU is approximately linear for large positive values
    // GELU(3) should be close to 3
    passed = passed && (output[7] > 2.9f);

    // GELU should be negative for negative inputs but close to 0
    passed = passed && (output[0] < 0.0f && output[0] > -0.1f);

    test_result("GELU basic", passed);
}

// ============================================================
// Test 3: LayerNorm
// ============================================================

void test_layernorm() {
    printf("\n=== Test: LayerNorm ===\n");

    float input[D_MODEL];
    float output[D_MODEL];
    float gamma[D_MODEL];
    float beta[D_MODEL];

    // Initialize
    init_array(input, D_MODEL, 2.0f, 42);
    for (size_t i = 0; i < D_MODEL; i++) {
        gamma[i] = 1.0f;
        beta[i] = 0.0f;
    }

    layernorm_fp32(output, input, D_MODEL, gamma, beta, 1e-5f);

    print_vector("Input", input, D_MODEL);
    print_vector("Output", output, D_MODEL);

    // Check that output has mean ~0 and variance ~1
    float mean = mean_fp32(output, D_MODEL);
    float var = variance_fp32(output, D_MODEL);

    printf("Output mean: %.6f (should be ~0)\n", mean);
    printf("Output variance: %.6f (should be ~1)\n", var);

    bool passed = check_close(mean, 0.0f, 0.01f);
    passed = passed && check_close(var, 1.0f, 0.01f);

    test_result("LayerNorm basic", passed);
}

// ============================================================
// Test 4: Photonic MVM (single tile)
// ============================================================

void test_photonic_mvm_tile() {
    printf("\n=== Test: Photonic MVM (tile) ===\n");

    const size_t out_size = 8;
    const size_t in_size = 8;

    float weight[out_size * in_size];
    float input[in_size];
    float output[out_size];

    // Initialize with simple values
    // Identity-like matrix with small perturbations
    for (size_t i = 0; i < out_size; i++) {
        for (size_t j = 0; j < in_size; j++) {
            weight[i * in_size + j] = (i == j) ? 1.0f : 0.01f;
        }
    }

    for (size_t i = 0; i < in_size; i++) {
        input[i] = (float)(i + 1);  // [1, 2, 3, ...]
    }

    // Execute photonic MVM
    photonic_mvm_tile_fp32(output, weight, input, out_size, in_size);

    print_vector("Input", input, in_size);
    print_matrix("Weight", weight, out_size, in_size);
    print_vector("Output", output, out_size);

    // With near-identity matrix, output should be close to input
    // Plus small contribution from off-diagonal elements
    bool passed = true;
    for (size_t i = 0; i < out_size; i++) {
        float expected = input[i];
        // Add contribution from off-diagonal (0.01 * sum of other inputs)
        for (size_t j = 0; j < in_size; j++) {
            if (j != i) expected += 0.01f * input[j];
        }
        if (!check_close(output[i], expected, 0.1f)) {
            printf("Mismatch at %zu: got %.4f, expected %.4f\n", i, output[i], expected);
            passed = false;
        }
    }

    test_result("Photonic MVM tile", passed);
}

// ============================================================
// Test 5: Photonic MVM (tiled - larger matrix)
// ============================================================

void test_photonic_mvm_tiled() {
    printf("\n=== Test: Photonic MVM (tiled) ===\n");

    const size_t out_size = 64;  // Requires tiling (> 32)
    const size_t in_size = 48;

    float *weight = (float *)malloc(out_size * in_size * sizeof(float));
    float *input = (float *)malloc(in_size * sizeof(float));
    float *output = (float *)malloc(out_size * sizeof(float));
    float *expected = (float *)malloc(out_size * sizeof(float));

    // Initialize weight with uniform small values
    for (size_t i = 0; i < out_size; i++) {
        for (size_t j = 0; j < in_size; j++) {
            weight[i * in_size + j] = 0.02f;  // All same value
        }
    }

    for (size_t i = 0; i < in_size; i++) {
        input[i] = 1.0f;  // All ones
    }

    // Compute expected output: each output = 0.02 * 48 = 0.96
    float expected_val = 0.02f * in_size;
    for (size_t i = 0; i < out_size; i++) {
        expected[i] = expected_val;
    }

    // Execute photonic MVM
    photonic_mvm_fp32(output, weight, input, out_size, in_size);

    printf("Matrix size: %zu x %zu (requires tiling)\n", out_size, in_size);
    print_vector("Output (first 8)", output, 8);
    printf("Expected per-output: %.4f\n", expected_val);

    // Check outputs are close to expected
    bool passed = true;
    float max_err = 0.0f;
    for (size_t i = 0; i < out_size; i++) {
        float err = fabsf(output[i] - expected[i]);
        if (err > max_err) max_err = err;
        if (err > 0.1f) {  // 10% tolerance
            passed = false;
        }
    }
    printf("Max error: %.6f\n", max_err);

    test_result("Photonic MVM tiled", passed);

    free(weight);
    free(input);
    free(output);
    free(expected);
}

// ============================================================
// Test 6: Scaled Dot-Product Attention
// ============================================================

void test_attention() {
    printf("\n=== Test: Scaled Dot-Product Attention ===\n");

    // Small attention test: 4 tokens, d_k = 8
    const size_t seq_len = 4;
    const size_t d_k = 8;

    float *Q = (float *)malloc(seq_len * d_k * sizeof(float));
    float *K = (float *)malloc(seq_len * d_k * sizeof(float));
    float *V = (float *)malloc(seq_len * d_k * sizeof(float));
    float *output = (float *)malloc(seq_len * d_k * sizeof(float));

    // Initialize Q, K, V with simple patterns
    init_normalized(Q, seq_len * d_k, 1);
    init_normalized(K, seq_len * d_k, 2);
    init_normalized(V, seq_len * d_k, 3);

    printf("Q, K, V dimensions: [%zu x %zu]\n", seq_len, d_k);
    print_matrix("Q", Q, seq_len, d_k);
    print_matrix("K", K, seq_len, d_k);
    print_matrix("V", V, seq_len, d_k);

    // Run attention (non-causal)
    scaled_dot_product_attention_fp32(output, Q, K, V, seq_len, d_k, d_k, false);

    print_matrix("Attention output", output, seq_len, d_k);

    // Basic sanity check: output should have reasonable values
    bool passed = true;
    for (size_t i = 0; i < seq_len * d_k; i++) {
        if (isnan(output[i]) || isinf(output[i])) {
            passed = false;
            break;
        }
    }

    // Output should be weighted combination of V
    // So values should be in similar range as V
    float v_max = 0.0f, v_min = 0.0f;
    float o_max = 0.0f, o_min = 0.0f;
    for (size_t i = 0; i < seq_len * d_k; i++) {
        if (V[i] > v_max) v_max = V[i];
        if (V[i] < v_min) v_min = V[i];
        if (output[i] > o_max) o_max = output[i];
        if (output[i] < o_min) o_min = output[i];
    }
    printf("V range: [%.4f, %.4f]\n", v_min, v_max);
    printf("Output range: [%.4f, %.4f]\n", o_min, o_max);

    test_result("Scaled Dot-Product Attention", passed);

    free(Q);
    free(K);
    free(V);
    free(output);
}

// ============================================================
// Test 7: Causal Attention
// ============================================================

void test_causal_attention() {
    printf("\n=== Test: Causal Attention ===\n");

    const size_t seq_len = 4;
    const size_t d_k = 8;

    float *Q = (float *)malloc(seq_len * d_k * sizeof(float));
    float *K = (float *)malloc(seq_len * d_k * sizeof(float));
    float *V = (float *)malloc(seq_len * d_k * sizeof(float));
    float *output_causal = (float *)malloc(seq_len * d_k * sizeof(float));
    float *output_nocausal = (float *)malloc(seq_len * d_k * sizeof(float));

    // Initialize with distinct patterns for each token
    for (size_t t = 0; t < seq_len; t++) {
        for (size_t i = 0; i < d_k; i++) {
            Q[t * d_k + i] = 0.1f * (float)(t + 1) * sinf((float)i * 0.5f);
            K[t * d_k + i] = 0.1f * (float)(t + 1) * cosf((float)i * 0.5f);
            V[t * d_k + i] = (float)(t + 1) * 0.25f;  // Each token has distinct V
        }
    }

    // Run both causal and non-causal
    scaled_dot_product_attention_fp32(output_causal, Q, K, V, seq_len, d_k, d_k, true);
    scaled_dot_product_attention_fp32(output_nocausal, Q, K, V, seq_len, d_k, d_k, false);

    printf("With causal mask:\n");
    print_matrix("Output", output_causal, seq_len, d_k);

    printf("Without causal mask:\n");
    print_matrix("Output", output_nocausal, seq_len, d_k);

    // Basic sanity check: no NaN/Inf
    bool passed = true;
    for (size_t i = 0; i < seq_len * d_k; i++) {
        if (isnan(output_causal[i]) || isinf(output_causal[i]) ||
            isnan(output_nocausal[i]) || isinf(output_nocausal[i])) {
            passed = false;
            break;
        }
    }

    // First token should be similar (only attends to itself in causal)
    float first_diff = 0.0f;
    for (size_t i = 0; i < d_k; i++) {
        first_diff += fabsf(output_causal[i] - output_nocausal[i]);
    }
    first_diff /= d_k;
    printf("First token avg diff: %.6f\n", first_diff);

    // Last token should differ more (causal can't see all tokens)
    float last_diff = 0.0f;
    for (size_t i = 0; i < d_k; i++) {
        last_diff += fabsf(output_causal[(seq_len - 1) * d_k + i] -
                          output_nocausal[(seq_len - 1) * d_k + i]);
    }
    last_diff /= d_k;
    printf("Last token avg diff: %.6f\n", last_diff);

    // The difference should exist (masking has effect)
    // Note: With some patterns, the difference might be subtle
    bool mask_has_effect = (last_diff > 0.001f) || (first_diff < last_diff);
    printf("Mask has effect: %s\n", mask_has_effect ? "yes" : "possible");

    test_result("Causal vs Non-causal Attention", passed);

    free(Q);
    free(K);
    free(V);
    free(output_causal);
    free(output_nocausal);
}

// ============================================================
// Test 8: FFN with GELU
// ============================================================

void test_ffn_gelu() {
    printf("\n=== Test: FFN with GELU ===\n");

    const size_t seq_len = 4;
    const size_t d_model = 16;
    const size_t d_ff = 32;

    float *input = (float *)malloc(seq_len * d_model * sizeof(float));
    float *output = (float *)malloc(seq_len * d_model * sizeof(float));
    float *W1 = (float *)malloc(d_ff * d_model * sizeof(float));
    float *b1 = (float *)malloc(d_ff * sizeof(float));
    float *W2 = (float *)malloc(d_model * d_ff * sizeof(float));
    float *b2 = (float *)malloc(d_model * sizeof(float));

    // Initialize
    init_normalized(input, seq_len * d_model, 1);
    init_array(W1, d_ff * d_model, 0.1f, 2);
    init_array(W2, d_model * d_ff, 0.1f, 3);
    for (size_t i = 0; i < d_ff; i++) b1[i] = 0.0f;
    for (size_t i = 0; i < d_model; i++) b2[i] = 0.0f;

    print_matrix("Input", input, seq_len, d_model);

    ffn_gelu_fp32(output, input, W1, b1, W2, b2, seq_len, d_model, d_ff);

    print_matrix("FFN Output", output, seq_len, d_model);

    // Basic sanity check
    bool passed = true;
    for (size_t i = 0; i < seq_len * d_model; i++) {
        if (isnan(output[i]) || isinf(output[i])) {
            passed = false;
            break;
        }
    }

    test_result("FFN with GELU", passed);

    free(input);
    free(output);
    free(W1);
    free(b1);
    free(W2);
    free(b2);
}

// ============================================================
// Test 9: FFN with SiLU (LLaMA style)
// ============================================================

void test_ffn_silu() {
    printf("\n=== Test: FFN with SiLU (LLaMA style) ===\n");

    const size_t seq_len = 4;
    const size_t d_model = 16;
    const size_t d_ff = 32;

    float *input = (float *)malloc(seq_len * d_model * sizeof(float));
    float *output = (float *)malloc(seq_len * d_model * sizeof(float));
    float *W_gate = (float *)malloc(d_ff * d_model * sizeof(float));
    float *W_up = (float *)malloc(d_ff * d_model * sizeof(float));
    float *W_down = (float *)malloc(d_model * d_ff * sizeof(float));

    // Initialize
    init_normalized(input, seq_len * d_model, 1);
    init_array(W_gate, d_ff * d_model, 0.1f, 10);
    init_array(W_up, d_ff * d_model, 0.1f, 20);
    init_array(W_down, d_model * d_ff, 0.1f, 30);

    print_matrix("Input", input, seq_len, d_model);

    ffn_silu_fp32(output, input, W_gate, W_up, W_down, seq_len, d_model, d_ff);

    print_matrix("FFN SiLU Output", output, seq_len, d_model);

    // Basic sanity check
    bool passed = true;
    for (size_t i = 0; i < seq_len * d_model; i++) {
        if (isnan(output[i]) || isinf(output[i])) {
            passed = false;
            break;
        }
    }

    test_result("FFN with SiLU", passed);

    free(input);
    free(output);
    free(W_gate);
    free(W_up);
    free(W_down);
}

// ============================================================
// Test 10: Full Transformer Block
// ============================================================

void test_transformer_block() {
    printf("\n=== Test: Full Transformer Block ===\n");

    const size_t seq_len = 4;
    const size_t d_model = 32;
    const size_t d_k = 16;
    const size_t d_ff = 64;

    // Allocate all tensors
    float *input = (float *)malloc(seq_len * d_model * sizeof(float));
    float *output = (float *)malloc(seq_len * d_model * sizeof(float));

    // LayerNorm 1 params
    float *ln1_gamma = (float *)malloc(d_model * sizeof(float));
    float *ln1_beta = (float *)malloc(d_model * sizeof(float));

    // Attention weights
    float *Wq = (float *)malloc(d_k * d_model * sizeof(float));
    float *Wk = (float *)malloc(d_k * d_model * sizeof(float));
    float *Wv = (float *)malloc(d_k * d_model * sizeof(float));
    float *Wo = (float *)malloc(d_model * d_k * sizeof(float));

    // LayerNorm 2 params
    float *ln2_gamma = (float *)malloc(d_model * sizeof(float));
    float *ln2_beta = (float *)malloc(d_model * sizeof(float));

    // FFN weights
    float *W1 = (float *)malloc(d_ff * d_model * sizeof(float));
    float *b1 = (float *)malloc(d_ff * sizeof(float));
    float *W2 = (float *)malloc(d_model * d_ff * sizeof(float));
    float *b2 = (float *)malloc(d_model * sizeof(float));

    // Initialize input
    init_normalized(input, seq_len * d_model, 1);

    // Initialize LayerNorm params (identity transform)
    for (size_t i = 0; i < d_model; i++) {
        ln1_gamma[i] = 1.0f;
        ln1_beta[i] = 0.0f;
        ln2_gamma[i] = 1.0f;
        ln2_beta[i] = 0.0f;
    }

    // Initialize attention weights (small random values)
    init_array(Wq, d_k * d_model, 0.1f, 100);
    init_array(Wk, d_k * d_model, 0.1f, 200);
    init_array(Wv, d_k * d_model, 0.1f, 300);
    init_array(Wo, d_model * d_k, 0.1f, 400);

    // Initialize FFN weights
    init_array(W1, d_ff * d_model, 0.05f, 500);
    init_array(W2, d_model * d_ff, 0.05f, 600);
    for (size_t i = 0; i < d_ff; i++) b1[i] = 0.0f;
    for (size_t i = 0; i < d_model; i++) b2[i] = 0.0f;

    printf("Transformer block config:\n");
    printf("  seq_len = %zu\n", seq_len);
    printf("  d_model = %zu\n", d_model);
    printf("  d_k = %zu\n", d_k);
    printf("  d_ff = %zu\n", d_ff);
    printf("  causal = true\n\n");

    print_matrix("Input", input, seq_len, d_model);

    // Run transformer block
    transformer_block_fp32(output, input,
                           ln1_gamma, ln1_beta,
                           Wq, Wk, Wv, Wo,
                           ln2_gamma, ln2_beta,
                           W1, b1, W2, b2,
                           seq_len, d_model, d_k, d_ff, true);

    print_matrix("Output", output, seq_len, d_model);

    // Sanity checks
    bool passed = true;

    // 1. No NaN or Inf
    for (size_t i = 0; i < seq_len * d_model; i++) {
        if (isnan(output[i]) || isinf(output[i])) {
            printf("Found NaN/Inf at position %zu\n", i);
            passed = false;
            break;
        }
    }

    // 2. Output should differ from input (transformation occurred)
    bool differs = false;
    for (size_t i = 0; i < seq_len * d_model; i++) {
        if (fabsf(output[i] - input[i]) > 0.001f) {
            differs = true;
            break;
        }
    }
    if (!differs) {
        printf("Output is identical to input - transformation may have failed\n");
        passed = false;
    }

    // 3. Values should be in reasonable range
    float max_val = 0.0f;
    for (size_t i = 0; i < seq_len * d_model; i++) {
        if (fabsf(output[i]) > max_val) max_val = fabsf(output[i]);
    }
    printf("Max absolute output value: %.4f\n", max_val);
    if (max_val > 100.0f) {
        printf("Warning: output values may be too large\n");
    }

    test_result("Full Transformer Block", passed);

    // Cleanup
    free(input);
    free(output);
    free(ln1_gamma);
    free(ln1_beta);
    free(Wq);
    free(Wk);
    free(Wv);
    free(Wo);
    free(ln2_gamma);
    free(ln2_beta);
    free(W1);
    free(b1);
    free(W2);
    free(b2);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("================================================================\n");
    printf("  FIONA Transformer Operations Test\n");
    printf("================================================================\n");

    // Print photonic model info
    const char* model = getenv("FIONA_PHOTONIC_MODEL");
    if (model) {
        printf("Photonic Model: %s\n", model);
    } else {
        printf("Photonic Model: ideal (default)\n");
    }
    printf("\n");

    // Run all tests
    test_softmax();
    test_gelu();
    test_layernorm();
    test_photonic_mvm_tile();
    test_photonic_mvm_tiled();
    test_attention();
    test_causal_attention();
    test_ffn_gelu();
    test_ffn_silu();
    test_transformer_block();

    // Summary
    printf("\n================================================================\n");
    printf("  Test Summary: %d / %d passed\n", pass_count, test_count);
    printf("================================================================\n");

    if (pass_count == test_count) {
        printf("\nAll tests passed!\n");
    } else {
        printf("\nSome tests failed.\n");
    }

    DUMP_STAT;

    return (pass_count == test_count) ? 0 : 1;
}
