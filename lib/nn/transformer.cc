/**
 * @file transformer.cc
 * @brief Implementation of Transformer-specific operations
 *
 * Includes both:
 * - Electronic operations (softmax, gelu, layernorm, etc.)
 * - Photonic operations (MVM-based linear, matmul, attention)
 *
 * @author FIONA Project
 * @date 2025-12-18
 */

#include "nn/transformer.h"
#include "base/instr.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

// Photonic vector register size
#define VREG_SIZE 32

// ============================================================
// Utility Functions
// ============================================================

float mean_fp32(const float *in, size_t len) {
    float sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        sum += in[i];
    }
    return sum / (float)len;
}

float variance_fp32(const float *in, size_t len) {
    float m = mean_fp32(in, len);
    float sum_sq = 0.0f;
    for (size_t i = 0; i < len; i++) {
        float diff = in[i] - m;
        sum_sq += diff * diff;
    }
    return sum_sq / (float)len;
}

float max_fp32(const float *in, size_t len) {
    float max_val = -FLT_MAX;
    for (size_t i = 0; i < len; i++) {
        if (in[i] > max_val) {
            max_val = in[i];
        }
    }
    return max_val;
}

size_t argmax_fp32(const float *in, size_t len) {
    size_t max_idx = 0;
    float max_val = in[0];
    for (size_t i = 1; i < len; i++) {
        if (in[i] > max_val) {
            max_val = in[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// ============================================================
// Softmax
// ============================================================

void softmax_fp32(float *out, const float *in, size_t len) {
    // Numerically stable softmax: subtract max before exp
    float max_val = max_fp32(in, len);

    // Compute exp(x - max)
    float sum_exp = 0.0f;
    for (size_t i = 0; i < len; i++) {
        out[i] = expf(in[i] - max_val);
        sum_exp += out[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (size_t i = 0; i < len; i++) {
        out[i] *= inv_sum;
    }
}

void softmax_2d_fp32(float *out, const float *in, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        softmax_fp32(&out[i * cols], &in[i * cols], cols);
    }
}

// ============================================================
// GELU (Gaussian Error Linear Unit)
// ============================================================

// Constants for GELU approximation
static const float GELU_SQRT_2_OVER_PI = 0.7978845608f;  // sqrt(2/pi)
static const float GELU_COEFF = 0.044715f;

void gelu_fp32(float *out, const float *in, size_t len) {
    // Exact GELU using tanh approximation:
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    for (size_t i = 0; i < len; i++) {
        float x = in[i];
        float x3 = x * x * x;
        float inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

void gelu_tanh_fp32(float *out, const float *in, size_t len) {
    // Same as gelu_fp32 (tanh approximation is the standard)
    gelu_fp32(out, in, len);
}

void gelu_2d_fp32(float *out, const float *in, size_t rows, size_t cols) {
    gelu_fp32(out, in, rows * cols);
}

// ============================================================
// SiLU (Sigmoid Linear Unit) / Swish
// ============================================================

static inline float sigmoid_scalar(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void silu_fp32(float *out, const float *in, size_t len) {
    // SiLU(x) = x * sigmoid(x)
    for (size_t i = 0; i < len; i++) {
        out[i] = in[i] * sigmoid_scalar(in[i]);
    }
}

void silu_2d_fp32(float *out, const float *in, size_t rows, size_t cols) {
    silu_fp32(out, in, rows * cols);
}

// ============================================================
// LayerNorm
// ============================================================

void layernorm_fp32(float *out, const float *in, size_t len,
                    const float *gamma, const float *beta, float eps) {
    // Compute mean
    float m = mean_fp32(in, len);

    // Compute variance
    float var = 0.0f;
    for (size_t i = 0; i < len; i++) {
        float diff = in[i] - m;
        var += diff * diff;
    }
    var /= (float)len;

    // Normalize and apply affine transform
    float inv_std = 1.0f / sqrtf(var + eps);
    for (size_t i = 0; i < len; i++) {
        float normalized = (in[i] - m) * inv_std;
        out[i] = gamma[i] * normalized + beta[i];
    }
}

void layernorm_2d_fp32(float *out, const float *in, size_t rows, size_t cols,
                       const LayerNormParamFP32 &param) {
    // Apply LayerNorm to each row (each token)
    for (size_t i = 0; i < rows; i++) {
        layernorm_fp32(&out[i * cols], &in[i * cols], cols,
                       param.gamma, param.beta, param.eps);
    }
}

// ============================================================
// RMSNorm
// ============================================================

void rmsnorm_fp32(float *out, const float *in, size_t len,
                  const float *gamma, float eps) {
    // Compute root mean square
    float sum_sq = 0.0f;
    for (size_t i = 0; i < len; i++) {
        sum_sq += in[i] * in[i];
    }
    float rms = sqrtf(sum_sq / (float)len + eps);
    float inv_rms = 1.0f / rms;

    // Scale
    for (size_t i = 0; i < len; i++) {
        out[i] = gamma[i] * in[i] * inv_rms;
    }
}

void rmsnorm_2d_fp32(float *out, const float *in, size_t rows, size_t cols,
                     const float *gamma, float eps) {
    for (size_t i = 0; i < rows; i++) {
        rmsnorm_fp32(&out[i * cols], &in[i * cols], cols, gamma, eps);
    }
}

// ============================================================
// Element-wise Operations
// ============================================================

void elementwise_mul_fp32(float *out, const float *a, const float *b, size_t len) {
    for (size_t i = 0; i < len; i++) {
        out[i] = a[i] * b[i];
    }
}

void elementwise_add_fp32(float *out, const float *a, const float *b, size_t len) {
    for (size_t i = 0; i < len; i++) {
        out[i] = a[i] + b[i];
    }
}

void scale_fp32(float *out, const float *in, float scale, size_t len) {
    for (size_t i = 0; i < len; i++) {
        out[i] = in[i] * scale;
    }
}

void scale_2d_fp32(float *out, const float *in, float scale, size_t rows, size_t cols) {
    scale_fp32(out, in, scale, rows * cols);
}

// ============================================================
// Attention Mask Application
// ============================================================

void apply_causal_mask_fp32(float *out, const float *in, size_t seq_len,
                            float mask_value) {
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            size_t idx = i * seq_len + j;
            if (j > i) {
                // Future position: mask out
                out[idx] = mask_value;
            } else {
                out[idx] = in[idx];
            }
        }
    }
}

void apply_padding_mask_fp32(float *out, const float *in,
                             const bool *padding_mask,
                             size_t rows, size_t cols,
                             float mask_value) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            size_t idx = i * cols + j;
            if (padding_mask[j]) {
                out[idx] = mask_value;
            } else {
                out[idx] = in[idx];
            }
        }
    }
}

// ============================================================
// Photonic MVM Operations (using FP32 instruction macros)
// ============================================================

// Temporary buffers for tiled MVM
static float pho_temp_vec[VREG_SIZE];
static float pho_temp_mat[VREG_SIZE][VREG_SIZE];
static float pho_temp_out[VREG_SIZE];

void photonic_mvm_tile_fp32(float *out, const float *weight, const float *input,
                            size_t out_size, size_t in_size) {
    // Prepare input vector (pad to VREG_SIZE)
    for (size_t i = 0; i < VREG_SIZE; i++) {
        pho_temp_vec[i] = (i < in_size) ? input[i] : 0.0f;
    }

    // Prepare weight matrix (pad to VREG_SIZE x VREG_SIZE)
    for (size_t i = 0; i < VREG_SIZE; i++) {
        for (size_t j = 0; j < VREG_SIZE; j++) {
            if (i < out_size && j < in_size) {
                pho_temp_mat[i][j] = weight[i * in_size + j];
            } else {
                pho_temp_mat[i][j] = 0.0f;
            }
        }
    }

    // Set vector length
    size_t vlen = VREG_SIZE;
    SET_VLEN_FP32(vlen);

    // Load input vector to FP32 vector register 1
    LOAD_V_FP32(1, pho_temp_vec);

    // Set weight matrix
    SET_MAT_FP32(&pho_temp_mat[0][0]);

    // Execute MVM (result in vector register 0)
    // This calls Python photonic model via DPI-C
    MVM_FP32(0, 1);

    // Store result
    STORE_V_FP32(0, pho_temp_out);

    // Copy valid outputs
    for (size_t i = 0; i < out_size; i++) {
        out[i] = pho_temp_out[i];
    }
}

void photonic_mvm_fp32(float *out, const float *weight, const float *input,
                       size_t out_size, size_t in_size) {
    // Initialize output to zero
    for (size_t i = 0; i < out_size; i++) {
        out[i] = 0.0f;
    }

    // Tile over output dimension
    for (size_t out_tile = 0; out_tile < out_size; out_tile += VREG_SIZE) {
        size_t out_tile_size = (out_tile + VREG_SIZE <= out_size) ?
                               VREG_SIZE : (out_size - out_tile);

        // Tile over input dimension (accumulate partial results)
        for (size_t in_tile = 0; in_tile < in_size; in_tile += VREG_SIZE) {
            size_t in_tile_size = (in_tile + VREG_SIZE <= in_size) ?
                                  VREG_SIZE : (in_size - in_tile);

            // Prepare tile inputs
            float tile_vec[VREG_SIZE] = {0};
            float tile_mat[VREG_SIZE * VREG_SIZE] = {0};
            float tile_out[VREG_SIZE] = {0};

            // Copy input vector tile
            for (size_t i = 0; i < in_tile_size; i++) {
                tile_vec[i] = input[in_tile + i];
            }

            // Copy weight matrix tile
            // Layout: tile_mat[i * in_tile_size + j] for photonic_mvm_tile_fp32
            for (size_t i = 0; i < out_tile_size; i++) {
                for (size_t j = 0; j < in_tile_size; j++) {
                    tile_mat[i * in_tile_size + j] =
                        weight[(out_tile + i) * in_size + (in_tile + j)];
                }
            }

            // Execute photonic MVM on tile
            photonic_mvm_tile_fp32(tile_out, tile_mat, tile_vec,
                                   out_tile_size, in_tile_size);

            // Accumulate results
            for (size_t i = 0; i < out_tile_size; i++) {
                out[out_tile + i] += tile_out[i];
            }
        }
    }
}

void photonic_linear_fp32(float *out, const float *weight, const float *input,
                          const float *bias, size_t out_size, size_t in_size) {
    // Execute photonic MVM
    photonic_mvm_fp32(out, weight, input, out_size, in_size);

    // Add bias (electronic operation)
    if (bias != NULL) {
        for (size_t i = 0; i < out_size; i++) {
            out[i] += bias[i];
        }
    }
}

void photonic_matmul_fp32(float *C, const float *A, const float *B,
                          size_t M, size_t K, size_t N) {
    // C[M x N] = A[M x K] @ B[K x N]
    // Process column by column using photonic MVM:
    // C[:, j] = A @ B[:, j]

    // Temporary buffer for B column
    float *B_col = (float *)malloc(K * sizeof(float));
    float *C_col = (float *)malloc(M * sizeof(float));

    for (size_t j = 0; j < N; j++) {
        // Extract column j from B
        for (size_t i = 0; i < K; i++) {
            B_col[i] = B[i * N + j];
        }

        // Compute C[:, j] = A @ B[:, j] using photonic MVM
        photonic_mvm_fp32(C_col, A, B_col, M, K);

        // Store result in column j of C
        for (size_t i = 0; i < M; i++) {
            C[i * N + j] = C_col[i];
        }
    }

    free(B_col);
    free(C_col);
}

// ============================================================
// Transformer Attention Operations
// ============================================================

void attention_qkv_projection_fp32(float *Q, float *K, float *V,
                                   const float *X,
                                   const float *Wq, const float *Wk, const float *Wv,
                                   size_t seq_len, size_t d_model,
                                   size_t d_k, size_t d_v) {
    // Q = X @ Wq^T  -> for each token: q_i = Wq @ x_i
    // K = X @ Wk^T
    // V = X @ Wv^T

    for (size_t t = 0; t < seq_len; t++) {
        const float *x_t = &X[t * d_model];
        float *q_t = &Q[t * d_k];
        float *k_t = &K[t * d_k];
        float *v_t = &V[t * d_v];

        // Each projection is a photonic MVM
        photonic_mvm_fp32(q_t, Wq, x_t, d_k, d_model);
        photonic_mvm_fp32(k_t, Wk, x_t, d_k, d_model);
        photonic_mvm_fp32(v_t, Wv, x_t, d_v, d_model);
    }
}

void scaled_dot_product_attention_fp32(float *output,
                                       const float *Q, const float *K, const float *V,
                                       size_t seq_len, size_t d_k, size_t d_v,
                                       bool causal) {
    // Allocate buffers for intermediate results
    size_t scores_size = seq_len * seq_len;
    float *scores = (float *)malloc(scores_size * sizeof(float));
    float *attn_weights = (float *)malloc(scores_size * sizeof(float));

    // Step 1: scores = Q @ K^T (photonic)
    // scores[i, j] = dot(Q[i], K[j])
    // This is a matrix multiplication: [seq_len x d_k] @ [d_k x seq_len]

    // First, transpose K: K^T[d_k x seq_len]
    float *K_T = (float *)malloc(d_k * seq_len * sizeof(float));
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_k; j++) {
            K_T[j * seq_len + i] = K[i * d_k + j];
        }
    }

    // Compute Q @ K^T using photonic matmul
    photonic_matmul_fp32(scores, Q, K_T, seq_len, d_k, seq_len);

    free(K_T);

    // Step 2: Scale by 1/sqrt(d_k) (electronic)
    float scale = 1.0f / sqrtf((float)d_k);
    scale_fp32(scores, scores, scale, scores_size);

    // Step 3: Apply causal mask if needed (electronic)
    if (causal) {
        apply_causal_mask_fp32(scores, scores, seq_len, -1e9f);
    }

    // Step 4: Softmax (electronic) - row-wise
    softmax_2d_fp32(attn_weights, scores, seq_len, seq_len);

    // Step 5: output = attn_weights @ V (photonic)
    // [seq_len x seq_len] @ [seq_len x d_v] = [seq_len x d_v]
    photonic_matmul_fp32(output, attn_weights, V, seq_len, seq_len, d_v);

    free(scores);
    free(attn_weights);
}

void attention_output_projection_fp32(float *output,
                                      const float *concat_heads,
                                      const float *Wo,
                                      size_t seq_len, size_t num_heads,
                                      size_t d_v, size_t d_model) {
    // output = concat_heads @ Wo
    // [seq_len x (num_heads * d_v)] @ [(num_heads * d_v) x d_model]

    size_t concat_dim = num_heads * d_v;

    for (size_t t = 0; t < seq_len; t++) {
        photonic_mvm_fp32(&output[t * d_model],
                          Wo,
                          &concat_heads[t * concat_dim],
                          d_model, concat_dim);
    }
}

// ============================================================
// Transformer Feed-Forward Network
// ============================================================

void ffn_gelu_fp32(float *output, const float *input,
                   const float *W1, const float *b1,
                   const float *W2, const float *b2,
                   size_t seq_len, size_t d_model, size_t d_ff) {
    // FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    // Allocate intermediate buffer
    float *hidden = (float *)malloc(seq_len * d_ff * sizeof(float));
    float *gelu_out = (float *)malloc(seq_len * d_ff * sizeof(float));

    // Step 1: hidden = input @ W1 + b1 (photonic linear)
    for (size_t t = 0; t < seq_len; t++) {
        photonic_linear_fp32(&hidden[t * d_ff],
                             W1,
                             &input[t * d_model],
                             b1, d_ff, d_model);
    }

    // Step 2: gelu_out = GELU(hidden) (electronic)
    gelu_fp32(gelu_out, hidden, seq_len * d_ff);

    // Step 3: output = gelu_out @ W2 + b2 (photonic linear)
    for (size_t t = 0; t < seq_len; t++) {
        photonic_linear_fp32(&output[t * d_model],
                             W2,
                             &gelu_out[t * d_ff],
                             b2, d_model, d_ff);
    }

    free(hidden);
    free(gelu_out);
}

void ffn_silu_fp32(float *output, const float *input,
                   const float *W_gate, const float *W_up, const float *W_down,
                   size_t seq_len, size_t d_model, size_t d_ff) {
    // FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    // Allocate intermediate buffers
    float *gate = (float *)malloc(seq_len * d_ff * sizeof(float));
    float *up = (float *)malloc(seq_len * d_ff * sizeof(float));
    float *silu_gate = (float *)malloc(seq_len * d_ff * sizeof(float));
    float *combined = (float *)malloc(seq_len * d_ff * sizeof(float));

    for (size_t t = 0; t < seq_len; t++) {
        // Step 1: gate = input @ W_gate (photonic)
        photonic_mvm_fp32(&gate[t * d_ff], W_gate, &input[t * d_model], d_ff, d_model);

        // Step 2: up = input @ W_up (photonic)
        photonic_mvm_fp32(&up[t * d_ff], W_up, &input[t * d_model], d_ff, d_model);
    }

    // Step 3: silu_gate = SiLU(gate) (electronic)
    silu_fp32(silu_gate, gate, seq_len * d_ff);

    // Step 4: combined = silu_gate * up (electronic - element-wise)
    elementwise_mul_fp32(combined, silu_gate, up, seq_len * d_ff);

    // Step 5: output = combined @ W_down (photonic)
    for (size_t t = 0; t < seq_len; t++) {
        photonic_mvm_fp32(&output[t * d_model], W_down, &combined[t * d_ff], d_model, d_ff);
    }

    free(gate);
    free(up);
    free(silu_gate);
    free(combined);
}

// ============================================================
// Full Transformer Block
// ============================================================

void transformer_block_fp32(float *output, const float *input,
                            const float *ln1_gamma, const float *ln1_beta,
                            const float *Wq, const float *Wk, const float *Wv, const float *Wo,
                            const float *ln2_gamma, const float *ln2_beta,
                            const float *W1, const float *b1,
                            const float *W2, const float *b2,
                            size_t seq_len, size_t d_model,
                            size_t d_k, size_t d_ff, bool causal) {
    // Transformer block (Pre-LN style):
    // x1 = LayerNorm(input)
    // attn_out = Attention(x1)
    // x2 = input + attn_out  (residual)
    // x3 = LayerNorm(x2)
    // ffn_out = FFN(x3)
    // output = x2 + ffn_out  (residual)

    size_t buffer_size = seq_len * d_model;
    float *ln1_out = (float *)malloc(buffer_size * sizeof(float));
    float *Q = (float *)malloc(seq_len * d_k * sizeof(float));
    float *K = (float *)malloc(seq_len * d_k * sizeof(float));
    float *V = (float *)malloc(seq_len * d_k * sizeof(float)); // d_v = d_k
    float *attn_out = (float *)malloc(seq_len * d_k * sizeof(float));
    float *proj_out = (float *)malloc(buffer_size * sizeof(float));
    float *x2 = (float *)malloc(buffer_size * sizeof(float));
    float *ln2_out = (float *)malloc(buffer_size * sizeof(float));
    float *ffn_out = (float *)malloc(buffer_size * sizeof(float));

    // Step 1: LayerNorm (electronic)
    for (size_t t = 0; t < seq_len; t++) {
        layernorm_fp32(&ln1_out[t * d_model], &input[t * d_model], d_model,
                       ln1_gamma, ln1_beta, 1e-5f);
    }

    // Step 2: Q, K, V projections (photonic)
    attention_qkv_projection_fp32(Q, K, V, ln1_out, Wq, Wk, Wv,
                                  seq_len, d_model, d_k, d_k);

    // Step 3: Scaled dot-product attention (photonic + electronic)
    scaled_dot_product_attention_fp32(attn_out, Q, K, V, seq_len, d_k, d_k, causal);

    // Step 4: Output projection (photonic)
    // Note: For single head, attn_out is [seq_len x d_k], Wo is [d_k x d_model]
    for (size_t t = 0; t < seq_len; t++) {
        photonic_mvm_fp32(&proj_out[t * d_model], Wo, &attn_out[t * d_k], d_model, d_k);
    }

    // Step 5: Residual connection (electronic)
    elementwise_add_fp32(x2, input, proj_out, buffer_size);

    // Step 6: LayerNorm 2 (electronic)
    for (size_t t = 0; t < seq_len; t++) {
        layernorm_fp32(&ln2_out[t * d_model], &x2[t * d_model], d_model,
                       ln2_gamma, ln2_beta, 1e-5f);
    }

    // Step 7: FFN (photonic + electronic)
    ffn_gelu_fp32(ffn_out, ln2_out, W1, b1, W2, b2, seq_len, d_model, d_ff);

    // Step 8: Final residual connection (electronic)
    elementwise_add_fp32(output, x2, ffn_out, buffer_size);

    // Free allocated memory
    free(ln1_out);
    free(Q);
    free(K);
    free(V);
    free(attn_out);
    free(proj_out);
    free(x2);
    free(ln2_out);
    free(ffn_out);
}
