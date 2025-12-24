/**
 * @file transformer_int16.cc
 * @brief INT16 Transformer operations for FIONA Photonic Accelerator
 *
 * IMPORTANT: Uses static aligned buffers for VLSU 64-byte alignment.
 * Maximum dimensions are limited by buffer sizes.
 *
 * Photonic MVM: Uses tiled_mvm from palu.h which calls RoCC DOTP instruction.
 * This goes through Spike custom extension to Python (photonic_models.py).
 * Photonic model selection via FIONA_PHOTONIC_MODEL environment variable:
 *   - ideal: Perfect mathematical operations (default)
 *   - mzi_realistic: Realistic MZI model with phase error, loss, crosstalk
 *   - noisy: Simple Gaussian noise
 *   - quantized: DAC/ADC quantization effects only
 *
 * @author FIONA Project
 * @date 2025-12-22 (Updated 2025-12-24: Photonic MVM integration)
 */

#include "transformer_int16.h"
#include "math/palu.h"  // For tiled_mvm (photonic MVM)
#include <string.h>
#include <stdio.h>

// ============================================================
// Maximum dimensions (for static buffer allocation)
// ============================================================
#define MAX_SEQ_LEN  16
#define MAX_D_MODEL  64
#define MAX_D_K      64
#define MAX_D_FF     128

// ============================================================
// Static aligned buffers for internal use
// ============================================================

// For softmax
static float softmax_in_f[MAX_SEQ_LEN] __attribute__((aligned(64)));
static float softmax_out_f[MAX_SEQ_LEN] __attribute__((aligned(64)));

// For layernorm
static float ln_in_f[MAX_D_MODEL] __attribute__((aligned(64)));

// For attention
static int16_t attn_scores[MAX_SEQ_LEN * MAX_SEQ_LEN] __attribute__((aligned(64)));
static int16_t attn_weights[MAX_SEQ_LEN * MAX_SEQ_LEN] __attribute__((aligned(64)));

// For FFN
static int16_t ffn_hidden[MAX_SEQ_LEN * MAX_D_FF] __attribute__((aligned(64)));

// For transformer block
static int16_t block_Q[MAX_SEQ_LEN * MAX_D_K] __attribute__((aligned(64)));
static int16_t block_K[MAX_SEQ_LEN * MAX_D_K] __attribute__((aligned(64)));
static int16_t block_V[MAX_SEQ_LEN * MAX_D_K] __attribute__((aligned(64)));
static int16_t block_attn_out[MAX_SEQ_LEN * MAX_D_K] __attribute__((aligned(64)));
static int16_t block_proj_out[MAX_SEQ_LEN * MAX_D_MODEL] __attribute__((aligned(64)));
static int16_t block_residual1[MAX_SEQ_LEN * MAX_D_MODEL] __attribute__((aligned(64)));
static int16_t block_ffn_out[MAX_SEQ_LEN * MAX_D_MODEL] __attribute__((aligned(64)));

// ============================================================
// Quantization Helpers
// ============================================================

void quantize_array(int16_t *out, const float *in, size_t len, float scale) {
    for (size_t i = 0; i < len; i++) {
        out[i] = quantize(in[i], scale);
    }
}

void dequantize_array(float *out, const int16_t *in, size_t len, float scale) {
    for (size_t i = 0; i < len; i++) {
        out[i] = dequantize(in[i], scale);
    }
}

// ============================================================
// Softmax (INT16 I/O, float internal)
// ============================================================

void softmax_int16(int16_t *out, const int16_t *in, size_t len,
                   float in_scale, float out_scale) {
    // Use static buffers (check size)
    if (len > MAX_SEQ_LEN) {
        printf("[ERROR] softmax_int16: len=%zu > MAX_SEQ_LEN=%d\n", len, MAX_SEQ_LEN);
        return;
    }

    // Dequantize
    dequantize_array(softmax_in_f, in, len, in_scale);

    // Compute softmax in float
    float max_val = softmax_in_f[0];
    for (size_t i = 1; i < len; i++) {
        if (softmax_in_f[i] > max_val) max_val = softmax_in_f[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        softmax_out_f[i] = expf(softmax_in_f[i] - max_val);
        sum += softmax_out_f[i];
    }

    for (size_t i = 0; i < len; i++) {
        softmax_out_f[i] /= sum;
    }

    // Quantize output
    quantize_array(out, softmax_out_f, len, out_scale);
}

void softmax_2d_int16(int16_t *out, const int16_t *in, size_t rows, size_t cols,
                      float in_scale, float out_scale) {
    for (size_t i = 0; i < rows; i++) {
        softmax_int16(&out[i * cols], &in[i * cols], cols, in_scale, out_scale);
    }
}

// ============================================================
// GELU (INT16 I/O)
// ============================================================

void gelu_int16(int16_t *out, const int16_t *in, size_t len,
                float in_scale, float out_scale) {
    for (size_t i = 0; i < len; i++) {
        float x = dequantize(in[i], in_scale);
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        float gelu = 0.5f * x * (1.0f + tanhf(inner));
        out[i] = quantize(gelu, out_scale);
    }
}

void gelu_2d_int16(int16_t *out, const int16_t *in, size_t rows, size_t cols,
                   float in_scale, float out_scale) {
    gelu_int16(out, in, rows * cols, in_scale, out_scale);
}

// ============================================================
// ReLU (INT16, direct)
// ============================================================

void relu_int16(int16_t *out, const int16_t *in, size_t len) {
    for (size_t i = 0; i < len; i++) {
        out[i] = (in[i] > 0) ? in[i] : 0;
    }
}

// ============================================================
// LayerNorm (INT16 I/O)
// ============================================================

void layernorm_int16(int16_t *out, const int16_t *in, size_t len,
                     const float *gamma, const float *beta, float eps,
                     float in_scale, float out_scale) {
    if (len > MAX_D_MODEL) {
        printf("[ERROR] layernorm_int16: len=%zu > MAX_D_MODEL=%d\n", len, MAX_D_MODEL);
        return;
    }

    // Dequantize
    dequantize_array(ln_in_f, in, len, in_scale);

    // Compute mean
    float mean = 0.0f;
    for (size_t i = 0; i < len; i++) {
        mean += ln_in_f[i];
    }
    mean /= len;

    // Compute variance
    float var = 0.0f;
    for (size_t i = 0; i < len; i++) {
        float diff = ln_in_f[i] - mean;
        var += diff * diff;
    }
    var /= len;

    // Normalize and apply gamma/beta
    float inv_std = 1.0f / sqrtf(var + eps);
    for (size_t i = 0; i < len; i++) {
        float normed = (ln_in_f[i] - mean) * inv_std;
        float result = gamma[i] * normed + beta[i];
        out[i] = quantize(result, out_scale);
    }
}

void layernorm_2d_int16(int16_t *out, const int16_t *in, size_t rows, size_t cols,
                        const LayerNormParamInt16 &param,
                        float in_scale, float out_scale) {
    for (size_t i = 0; i < rows; i++) {
        layernorm_int16(&out[i * cols], &in[i * cols], cols,
                        param.gamma, param.beta, param.eps,
                        in_scale, out_scale);
    }
}

// ============================================================
// Element-wise Operations
// ============================================================

void elementwise_add_int16(int16_t *out, const int16_t *a, const int16_t *b, size_t len) {
    for (size_t i = 0; i < len; i++) {
        int32_t sum = (int32_t)a[i] + (int32_t)b[i];
        if (sum > QUANT_MAX) sum = QUANT_MAX;
        if (sum < QUANT_MIN) sum = QUANT_MIN;
        out[i] = (int16_t)sum;
    }
}

void scale_int16(int16_t *out, const int16_t *in, float scale_factor, size_t len,
                 float in_scale, float out_scale) {
    for (size_t i = 0; i < len; i++) {
        float val = dequantize(in[i], in_scale) * scale_factor;
        out[i] = quantize(val, out_scale);
    }
}

// ============================================================
// Attention Mask
// ============================================================

void apply_causal_mask_int16(int16_t *out, const int16_t *in, size_t seq_len) {
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            if (j > i) {
                out[i * seq_len + j] = QUANT_MIN;
            } else {
                out[i * seq_len + j] = in[i * seq_len + j];
            }
        }
    }
}

// ============================================================
// MVM (INT16) - Photonic implementation via tiled_mvm
// ============================================================
// Uses FIONA photonic MVM via RoCC instructions -> Spike -> Python
// Photonic model selected via FIONA_PHOTONIC_MODEL environment variable:
//   - ideal: Perfect mathematical operations (default)
//   - mzi_realistic: Realistic MZI model with phase error, loss, crosstalk
//   - noisy: Simple Gaussian noise
//   - quantized: DAC/ADC quantization effects only

void photonic_mvm_tile_int16(int16_t *out, const int16_t *weight, const int16_t *input,
                             size_t out_size, size_t in_size) {
    // Use FIONA photonic MVM (same as photonic_mvm_int16)
    tiled_mvm(out, weight, input, out_size, in_size);
}

void photonic_mvm_int16(int16_t *out, const int16_t *weight, const int16_t *input,
                        size_t out_size, size_t in_size) {
    // Use FIONA photonic MVM via RoCC instructions
    // tiled_mvm calls DOTP which goes through Spike custom extension to Python
    // This enables photonic model selection via FIONA_PHOTONIC_MODEL env var
    tiled_mvm(out, weight, input, out_size, in_size);
}

void photonic_linear_int16(int16_t *out, const int16_t *weight, const int16_t *input,
                           const float *bias, size_t out_size, size_t in_size,
                           float w_scale, float in_scale, float out_scale) {
    photonic_mvm_int16(out, weight, input, out_size, in_size);

    if (bias != NULL) {
        float acc_scale = w_scale * in_scale;
        for (size_t i = 0; i < out_size; i++) {
            float val = dequantize(out[i], acc_scale) + bias[i];
            out[i] = quantize(val, out_scale);
        }
    }
}

void photonic_matmul_int16(int16_t *C, const int16_t *A, const int16_t *B,
                           size_t M, size_t K, size_t N) {
    // C[M x N] = A[M x K] @ B[K x N]
    // Use tiled_matmul_transpose with B transposed
    // For now, compute column by column (less efficient but simpler)
    static int16_t B_col[MAX_D_MODEL] __attribute__((aligned(64)));
    static int16_t C_col[MAX_D_MODEL] __attribute__((aligned(64)));

    for (size_t j = 0; j < N; j++) {
        // Extract column j of B
        for (size_t k = 0; k < K; k++) {
            B_col[k] = B[k * N + j];
        }

        // C[:, j] = A @ B_col
        photonic_mvm_int16(C_col, A, B_col, M, K);

        // Store to C
        for (size_t i = 0; i < M; i++) {
            C[i * N + j] = C_col[i];
        }
    }
}

// ============================================================
// Transformer Attention (INT16)
// ============================================================

void attention_qkv_projection_int16(int16_t *Q, int16_t *K, int16_t *V,
                                    const int16_t *X,
                                    const int16_t *Wq, const int16_t *Wk, const int16_t *Wv,
                                    size_t seq_len, size_t d_model,
                                    size_t d_k, size_t d_v,
                                    float scale) {
    for (size_t t = 0; t < seq_len; t++) {
        const int16_t *x_t = &X[t * d_model];

        photonic_mvm_int16(&Q[t * d_k], Wq, x_t, d_k, d_model);
        photonic_mvm_int16(&K[t * d_k], Wk, x_t, d_k, d_model);
        photonic_mvm_int16(&V[t * d_v], Wv, x_t, d_v, d_model);
    }
}

void scaled_dot_product_attention_int16(int16_t *output,
                                        const int16_t *Q, const int16_t *K, const int16_t *V,
                                        size_t seq_len, size_t d_k, size_t d_v,
                                        bool causal, float scale) {
    if (seq_len > MAX_SEQ_LEN) {
        printf("[ERROR] attention: seq_len=%zu > MAX_SEQ_LEN=%d\n", seq_len, MAX_SEQ_LEN);
        return;
    }

    // 1. Compute scores = Q @ K^T (CPU, no VLSU)
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            int32_t sum = 0;
            for (size_t k = 0; k < d_k; k++) {
                sum += (int32_t)Q[i * d_k + k] * (int32_t)K[j * d_k + k];
            }
            float score_f = (float)sum / (scale * scale);
            score_f /= sqrtf((float)d_k);
            attn_scores[i * seq_len + j] = quantize(score_f, scale);
        }
    }

    // 2. Apply causal mask
    if (causal) {
        apply_causal_mask_int16(attn_scores, attn_scores, seq_len);
    }

    // 3. Softmax
    softmax_2d_int16(attn_weights, attn_scores, seq_len, seq_len, scale, scale);

    // 4. output = attn_weights @ V (CPU, no VLSU)
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t d = 0; d < d_v; d++) {
            int32_t sum = 0;
            for (size_t j = 0; j < seq_len; j++) {
                sum += (int32_t)attn_weights[i * seq_len + j] * (int32_t)V[j * d_v + d];
            }
            float val = (float)sum / (scale * scale);
            output[i * d_v + d] = quantize(val, scale);
        }
    }
}

// ============================================================
// Transformer FFN (INT16)
// ============================================================

void ffn_gelu_int16(int16_t *output, const int16_t *input,
                    const int16_t *W1, const int16_t *W2,
                    size_t seq_len, size_t d_model, size_t d_ff,
                    float scale) {
    if (seq_len * d_ff > MAX_SEQ_LEN * MAX_D_FF) {
        printf("[ERROR] ffn_gelu: buffer overflow\n");
        return;
    }

    // hidden = GELU(input @ W1)
    for (size_t t = 0; t < seq_len; t++) {
        photonic_mvm_int16(&ffn_hidden[t * d_ff], W1, &input[t * d_model], d_ff, d_model);
    }
    gelu_2d_int16(ffn_hidden, ffn_hidden, seq_len, d_ff, scale, scale);

    // output = hidden @ W2
    for (size_t t = 0; t < seq_len; t++) {
        photonic_mvm_int16(&output[t * d_model], W2, &ffn_hidden[t * d_ff], d_model, d_ff);
    }
}

void ffn_relu_int16(int16_t *output, const int16_t *input,
                    const int16_t *W1, const int16_t *W2,
                    size_t seq_len, size_t d_model, size_t d_ff,
                    float scale) {
    if (seq_len * d_ff > MAX_SEQ_LEN * MAX_D_FF) {
        printf("[ERROR] ffn_relu: buffer overflow\n");
        return;
    }

    // hidden = ReLU(input @ W1)
    for (size_t t = 0; t < seq_len; t++) {
        photonic_mvm_int16(&ffn_hidden[t * d_ff], W1, &input[t * d_model], d_ff, d_model);
    }
    relu_int16(ffn_hidden, ffn_hidden, seq_len * d_ff);

    // output = hidden @ W2
    for (size_t t = 0; t < seq_len; t++) {
        photonic_mvm_int16(&output[t * d_model], W2, &ffn_hidden[t * d_ff], d_model, d_ff);
    }
}

// ============================================================
// Full Transformer Block (INT16)
// ============================================================

void transformer_block_int16(int16_t *output, const int16_t *input,
                             const int16_t *Wq, const int16_t *Wk,
                             const int16_t *Wv, const int16_t *Wo,
                             const int16_t *W1, const int16_t *W2,
                             size_t seq_len, size_t d_model,
                             size_t d_k, size_t d_ff,
                             bool causal, float scale) {
    if (seq_len > MAX_SEQ_LEN || d_model > MAX_D_MODEL ||
        d_k > MAX_D_K || d_ff > MAX_D_FF) {
        printf("[ERROR] transformer_block: dimension overflow\n");
        return;
    }

    // 1. Self-Attention
    // Q, K, V projections
    attention_qkv_projection_int16(block_Q, block_K, block_V, input, Wq, Wk, Wv,
                                   seq_len, d_model, d_k, d_k, scale);

    // Scaled dot-product attention
    scaled_dot_product_attention_int16(block_attn_out, block_Q, block_K, block_V,
                                       seq_len, d_k, d_k, causal, scale);

    // Output projection
    for (size_t t = 0; t < seq_len; t++) {
        photonic_mvm_int16(&block_proj_out[t * d_model], Wo,
                           &block_attn_out[t * d_k], d_model, d_k);
    }

    // Residual connection
    elementwise_add_int16(block_residual1, input, block_proj_out, seq_len * d_model);

    // 2. FFN with ReLU
    ffn_relu_int16(block_ffn_out, block_residual1, W1, W2, seq_len, d_model, d_ff, scale);

    // Residual connection
    elementwise_add_int16(output, block_residual1, block_ffn_out, seq_len * d_model);
}
