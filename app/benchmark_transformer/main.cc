/**
 * @file main.cc
 * @brief Transformer Photonic Model Benchmark
 *
 * Benchmarks Transformer operations across different photonic models:
 * - ideal: No noise, perfect computation
 * - noisy: Gaussian noise
 * - mzi_realistic: Full MZI noise model with insertion loss
 *
 * Metrics:
 * 1. Numerical accuracy (MSE, max error vs ideal)
 * 2. Attention weight quality (entropy, sparsity)
 * 3. Output distribution statistics
 *
 * Architecture:
 * - seq_len = 8, d_model = 64, d_k = 32, d_ff = 128
 * - Single-head attention for simplicity
 * - Multiple iterations for statistical significance
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

// Benchmark configuration
#define SEQ_LEN      8
#define D_MODEL      64
#define D_K          32
#define D_FF         128
#define NUM_ITERS    10   // Number of test iterations

// Statistics structure
struct Stats {
    float mean;
    float std_dev;
    float min_val;
    float max_val;
};

// ============================================================
// Utility Functions
// ============================================================

// Deterministic pseudo-random number generator
static unsigned int rand_seed = 12345;

float pseudo_rand() {
    rand_seed = rand_seed * 1103515245 + 12345;
    return ((float)(rand_seed % 10000) / 10000.0f) * 2.0f - 1.0f;  // [-1, 1]
}

void reset_rand(unsigned int seed) {
    rand_seed = seed;
}

void init_random(float *arr, size_t len, float scale) {
    for (size_t i = 0; i < len; i++) {
        arr[i] = pseudo_rand() * scale;
    }
}

Stats compute_stats(const float *arr, size_t len) {
    Stats s;
    s.mean = 0.0f;
    s.min_val = arr[0];
    s.max_val = arr[0];

    for (size_t i = 0; i < len; i++) {
        s.mean += arr[i];
        if (arr[i] < s.min_val) s.min_val = arr[i];
        if (arr[i] > s.max_val) s.max_val = arr[i];
    }
    s.mean /= len;

    float var = 0.0f;
    for (size_t i = 0; i < len; i++) {
        float diff = arr[i] - s.mean;
        var += diff * diff;
    }
    s.std_dev = sqrtf(var / len);

    return s;
}

float compute_mse(const float *a, const float *b, size_t len) {
    float mse = 0.0f;
    for (size_t i = 0; i < len; i++) {
        float diff = a[i] - b[i];
        mse += diff * diff;
    }
    return mse / len;
}

float compute_max_error(const float *a, const float *b, size_t len) {
    float max_err = 0.0f;
    for (size_t i = 0; i < len; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

float compute_cosine_similarity(const float *a, const float *b, size_t len) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < len; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a < 1e-10f || norm_b < 1e-10f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

// Count how many values differ by more than threshold
int count_errors(const float *a, const float *b, size_t len, float threshold) {
    int count = 0;
    for (size_t i = 0; i < len; i++) {
        if (fabsf(a[i] - b[i]) > threshold) count++;
    }
    return count;
}

// ============================================================
// Reference (Electronic-only) Implementation
// ============================================================

// Simple matrix multiplication for reference
void ref_matmul(float *C, const float *A, const float *B,
                size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void ref_mvm(float *out, const float *W, const float *x, size_t M, size_t K) {
    for (size_t i = 0; i < M; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < K; j++) {
            sum += W[i * K + j] * x[j];
        }
        out[i] = sum;
    }
}

// Reference attention (electronic only)
void ref_attention(float *output, const float *Q, const float *K, const float *V,
                   size_t seq_len, size_t d_k, size_t d_v) {
    size_t scores_size = seq_len * seq_len;
    float *scores = (float *)malloc(scores_size * sizeof(float));
    float *attn_weights = (float *)malloc(scores_size * sizeof(float));
    float *K_T = (float *)malloc(d_k * seq_len * sizeof(float));

    // Transpose K
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_k; j++) {
            K_T[j * seq_len + i] = K[i * d_k + j];
        }
    }

    // scores = Q @ K^T (electronic reference)
    ref_matmul(scores, Q, K_T, seq_len, d_k, seq_len);

    // Scale
    float scale = 1.0f / sqrtf((float)d_k);
    for (size_t i = 0; i < scores_size; i++) {
        scores[i] *= scale;
    }

    // Softmax
    softmax_2d_fp32(attn_weights, scores, seq_len, seq_len);

    // output = attn_weights @ V (electronic reference)
    ref_matmul(output, attn_weights, V, seq_len, seq_len, d_v);

    free(scores);
    free(attn_weights);
    free(K_T);
}

// ============================================================
// Benchmark Tests
// ============================================================

void benchmark_mvm() {
    printf("\n========================================\n");
    printf("Benchmark 1: Matrix-Vector Multiplication\n");
    printf("========================================\n");
    printf("Matrix size: %d x %d\n\n", D_MODEL, D_MODEL);

    float *W = (float *)malloc(D_MODEL * D_MODEL * sizeof(float));
    float *x = (float *)malloc(D_MODEL * sizeof(float));
    float *ref_out = (float *)malloc(D_MODEL * sizeof(float));
    float *pho_out = (float *)malloc(D_MODEL * sizeof(float));

    float total_mse = 0.0f;
    float total_max_err = 0.0f;
    float total_cosine = 0.0f;

    for (int iter = 0; iter < NUM_ITERS; iter++) {
        // Initialize with deterministic random values
        reset_rand(iter * 1000);
        init_random(W, D_MODEL * D_MODEL, 0.1f);
        init_random(x, D_MODEL, 1.0f);

        // Reference computation
        ref_mvm(ref_out, W, x, D_MODEL, D_MODEL);

        // Photonic computation
        photonic_mvm_fp32(pho_out, W, x, D_MODEL, D_MODEL);

        // Compute metrics
        float mse = compute_mse(ref_out, pho_out, D_MODEL);
        float max_err = compute_max_error(ref_out, pho_out, D_MODEL);
        float cosine = compute_cosine_similarity(ref_out, pho_out, D_MODEL);

        total_mse += mse;
        total_max_err += max_err;
        total_cosine += cosine;

        if (iter < 3) {
            printf("Iter %d: MSE=%.6f, MaxErr=%.6f, CosSim=%.6f\n",
                   iter, mse, max_err, cosine);
        }
    }

    printf("\n--- MVM Average over %d iterations ---\n", NUM_ITERS);
    printf("  Mean MSE:      %.6f\n", total_mse / NUM_ITERS);
    printf("  Mean MaxErr:   %.6f\n", total_max_err / NUM_ITERS);
    printf("  Mean CosSim:   %.6f\n", total_cosine / NUM_ITERS);

    free(W);
    free(x);
    free(ref_out);
    free(pho_out);
}

void benchmark_attention() {
    printf("\n========================================\n");
    printf("Benchmark 2: Scaled Dot-Product Attention\n");
    printf("========================================\n");
    printf("seq_len=%d, d_k=%d\n\n", SEQ_LEN, D_K);

    size_t q_size = SEQ_LEN * D_K;
    float *Q = (float *)malloc(q_size * sizeof(float));
    float *K = (float *)malloc(q_size * sizeof(float));
    float *V = (float *)malloc(q_size * sizeof(float));
    float *ref_out = (float *)malloc(q_size * sizeof(float));
    float *pho_out = (float *)malloc(q_size * sizeof(float));

    float total_mse = 0.0f;
    float total_max_err = 0.0f;
    float total_cosine = 0.0f;

    for (int iter = 0; iter < NUM_ITERS; iter++) {
        reset_rand(iter * 2000);
        init_random(Q, q_size, 0.5f);
        init_random(K, q_size, 0.5f);
        init_random(V, q_size, 0.5f);

        // Reference attention
        ref_attention(ref_out, Q, K, V, SEQ_LEN, D_K, D_K);

        // Photonic attention
        scaled_dot_product_attention_fp32(pho_out, Q, K, V, SEQ_LEN, D_K, D_K, false);

        float mse = compute_mse(ref_out, pho_out, q_size);
        float max_err = compute_max_error(ref_out, pho_out, q_size);
        float cosine = compute_cosine_similarity(ref_out, pho_out, q_size);

        total_mse += mse;
        total_max_err += max_err;
        total_cosine += cosine;

        if (iter < 3) {
            printf("Iter %d: MSE=%.6f, MaxErr=%.6f, CosSim=%.6f\n",
                   iter, mse, max_err, cosine);
        }
    }

    printf("\n--- Attention Average over %d iterations ---\n", NUM_ITERS);
    printf("  Mean MSE:      %.6f\n", total_mse / NUM_ITERS);
    printf("  Mean MaxErr:   %.6f\n", total_max_err / NUM_ITERS);
    printf("  Mean CosSim:   %.6f\n", total_cosine / NUM_ITERS);

    free(Q);
    free(K);
    free(V);
    free(ref_out);
    free(pho_out);
}

void benchmark_ffn() {
    printf("\n========================================\n");
    printf("Benchmark 3: Feed-Forward Network (GELU)\n");
    printf("========================================\n");
    printf("seq_len=%d, d_model=%d, d_ff=%d\n\n", SEQ_LEN, D_MODEL, D_FF);

    size_t input_size = SEQ_LEN * D_MODEL;
    float *input = (float *)malloc(input_size * sizeof(float));
    float *W1 = (float *)malloc(D_FF * D_MODEL * sizeof(float));
    float *b1 = (float *)malloc(D_FF * sizeof(float));
    float *W2 = (float *)malloc(D_MODEL * D_FF * sizeof(float));
    float *b2 = (float *)malloc(D_MODEL * sizeof(float));
    float *pho_out = (float *)malloc(input_size * sizeof(float));

    // Reference computation buffers
    float *ref_hidden = (float *)malloc(SEQ_LEN * D_FF * sizeof(float));
    float *ref_gelu = (float *)malloc(SEQ_LEN * D_FF * sizeof(float));
    float *ref_out = (float *)malloc(input_size * sizeof(float));

    float total_mse = 0.0f;
    float total_max_err = 0.0f;
    float total_cosine = 0.0f;

    for (int iter = 0; iter < NUM_ITERS; iter++) {
        reset_rand(iter * 3000);
        init_random(input, input_size, 0.5f);
        init_random(W1, D_FF * D_MODEL, 0.05f);
        init_random(W2, D_MODEL * D_FF, 0.05f);
        for (size_t i = 0; i < D_FF; i++) b1[i] = pseudo_rand() * 0.01f;
        for (size_t i = 0; i < D_MODEL; i++) b2[i] = pseudo_rand() * 0.01f;

        // Reference FFN
        for (size_t t = 0; t < SEQ_LEN; t++) {
            ref_mvm(&ref_hidden[t * D_FF], W1, &input[t * D_MODEL], D_FF, D_MODEL);
            for (size_t i = 0; i < D_FF; i++) {
                ref_hidden[t * D_FF + i] += b1[i];
            }
        }
        gelu_fp32(ref_gelu, ref_hidden, SEQ_LEN * D_FF);
        for (size_t t = 0; t < SEQ_LEN; t++) {
            ref_mvm(&ref_out[t * D_MODEL], W2, &ref_gelu[t * D_FF], D_MODEL, D_FF);
            for (size_t i = 0; i < D_MODEL; i++) {
                ref_out[t * D_MODEL + i] += b2[i];
            }
        }

        // Photonic FFN
        ffn_gelu_fp32(pho_out, input, W1, b1, W2, b2, SEQ_LEN, D_MODEL, D_FF);

        float mse = compute_mse(ref_out, pho_out, input_size);
        float max_err = compute_max_error(ref_out, pho_out, input_size);
        float cosine = compute_cosine_similarity(ref_out, pho_out, input_size);

        total_mse += mse;
        total_max_err += max_err;
        total_cosine += cosine;

        if (iter < 3) {
            printf("Iter %d: MSE=%.6f, MaxErr=%.6f, CosSim=%.6f\n",
                   iter, mse, max_err, cosine);
        }
    }

    printf("\n--- FFN Average over %d iterations ---\n", NUM_ITERS);
    printf("  Mean MSE:      %.6f\n", total_mse / NUM_ITERS);
    printf("  Mean MaxErr:   %.6f\n", total_max_err / NUM_ITERS);
    printf("  Mean CosSim:   %.6f\n", total_cosine / NUM_ITERS);

    free(input);
    free(W1);
    free(b1);
    free(W2);
    free(b2);
    free(pho_out);
    free(ref_hidden);
    free(ref_gelu);
    free(ref_out);
}

void benchmark_output_statistics() {
    printf("\n========================================\n");
    printf("Benchmark 4: Output Distribution Analysis\n");
    printf("========================================\n");

    size_t out_size = SEQ_LEN * D_MODEL;
    float *input = (float *)malloc(out_size * sizeof(float));
    float *W = (float *)malloc(D_MODEL * D_MODEL * sizeof(float));
    float *output = (float *)malloc(out_size * sizeof(float));

    reset_rand(42);
    init_random(input, out_size, 1.0f);
    init_random(W, D_MODEL * D_MODEL, 0.1f);

    // Single MVM per token
    for (size_t t = 0; t < SEQ_LEN; t++) {
        photonic_mvm_fp32(&output[t * D_MODEL], W, &input[t * D_MODEL], D_MODEL, D_MODEL);
    }

    Stats in_stats = compute_stats(input, out_size);
    Stats out_stats = compute_stats(output, out_size);

    printf("\nInput distribution:\n");
    printf("  Mean: %.6f, Std: %.6f, Min: %.6f, Max: %.6f\n",
           in_stats.mean, in_stats.std_dev, in_stats.min_val, in_stats.max_val);

    printf("\nOutput distribution:\n");
    printf("  Mean: %.6f, Std: %.6f, Min: %.6f, Max: %.6f\n",
           out_stats.mean, out_stats.std_dev, out_stats.min_val, out_stats.max_val);

    // Check for any NaN or Inf
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < out_size; i++) {
        if (isnan(output[i])) nan_count++;
        if (isinf(output[i])) inf_count++;
    }
    printf("\nNaN count: %d, Inf count: %d\n", nan_count, inf_count);

    free(input);
    free(W);
    free(output);
}

// ============================================================
// Summary Report
// ============================================================

void print_summary_header() {
    printf("\n");
    printf("================================================================\n");
    printf("  FIONA Transformer Benchmark Summary\n");
    printf("================================================================\n");

    const char* model = getenv("FIONA_PHOTONIC_MODEL");
    const char* noise_sigma = getenv("FIONA_NOISE_SIGMA");

    printf("Photonic Model: %s\n", model ? model : "ideal (default)");
    if (noise_sigma) printf("Noise Sigma: %s\n", noise_sigma);

    printf("\nConfiguration:\n");
    printf("  seq_len  = %d\n", SEQ_LEN);
    printf("  d_model  = %d\n", D_MODEL);
    printf("  d_k      = %d\n", D_K);
    printf("  d_ff     = %d\n", D_FF);
    printf("  iters    = %d\n", NUM_ITERS);
    printf("================================================================\n");
}

// ============================================================
// Main
// ============================================================

int main() {
    print_summary_header();

    // Run benchmarks
    benchmark_mvm();
    benchmark_attention();
    benchmark_ffn();
    benchmark_output_statistics();

    printf("\n================================================================\n");
    printf("  Benchmark Complete!\n");
    printf("================================================================\n");

    DUMP_STAT;

    return 0;
}
