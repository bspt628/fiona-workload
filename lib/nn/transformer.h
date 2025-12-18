/**
 * @file transformer.h
 * @brief Transformer-specific operations for FIONA
 *
 * This module provides electronic (non-photonic) operations required for
 * Transformer models that cannot be efficiently implemented in photonics:
 * - Softmax: Non-linear, requires exp() and division
 * - GELU: Non-linear activation function
 * - LayerNorm: Requires mean, variance, sqrt, division
 * - Scaled Dot-Product Attention helpers
 *
 * These operations are executed on the electronic (RISC-V) side, while
 * linear operations (Q/K/V projections, FFN) use photonic MVMs.
 *
 * @author FIONA Project
 * @date 2025-12-18
 */

#ifndef FIONA_NN_TRANSFORMER_H
#define FIONA_NN_TRANSFORMER_H

#include "math/all.h"
#include <math.h>

// ============================================================
// Softmax
// ============================================================

/**
 * @brief Softmax operation: softmax(x)_i = exp(x_i) / sum(exp(x))
 *
 * Numerically stable implementation using max subtraction:
 * softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
 *
 * @param out Output array (same size as in)
 * @param in Input array
 * @param len Length of arrays
 */
void softmax_fp32(float *out, const float *in, size_t len);

/**
 * @brief Row-wise softmax for 2D matrix
 *
 * Applies softmax to each row independently.
 * Used for attention scores: softmax(QK^T / sqrt(d_k))
 *
 * @param out Output matrix [rows x cols]
 * @param in Input matrix [rows x cols]
 * @param rows Number of rows (sequence length)
 * @param cols Number of columns (sequence length for attention)
 */
void softmax_2d_fp32(float *out, const float *in, size_t rows, size_t cols);

// ============================================================
// GELU (Gaussian Error Linear Unit)
// ============================================================

/**
 * @brief GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))
 *
 * Approximation (faster): x * sigmoid(1.702 * x)
 * Exact: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * Used in modern Transformers (BERT, GPT) as activation function.
 *
 * @param out Output array
 * @param in Input array
 * @param len Length of arrays
 */
void gelu_fp32(float *out, const float *in, size_t len);

/**
 * @brief Fast GELU approximation using tanh
 *
 * gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
void gelu_tanh_fp32(float *out, const float *in, size_t len);

/**
 * @brief GELU for 2D matrix
 */
void gelu_2d_fp32(float *out, const float *in, size_t rows, size_t cols);

// ============================================================
// SiLU (Sigmoid Linear Unit) / Swish
// ============================================================

/**
 * @brief SiLU/Swish activation: x * sigmoid(x)
 *
 * Used in some Transformer variants (e.g., LLaMA uses SiLU in FFN).
 *
 * @param out Output array
 * @param in Input array
 * @param len Length of arrays
 */
void silu_fp32(float *out, const float *in, size_t len);

/**
 * @brief SiLU for 2D matrix
 */
void silu_2d_fp32(float *out, const float *in, size_t rows, size_t cols);

// ============================================================
// LayerNorm (1D for Transformer)
// ============================================================

/**
 * @brief LayerNorm parameters for Transformer
 */
struct LayerNormParamFP32 {
    float *gamma;  // Scale parameter [normalized_shape]
    float *beta;   // Shift parameter [normalized_shape]
    float eps;     // Small constant for numerical stability
    size_t normalized_shape;  // Feature dimension

    LayerNormParamFP32(size_t dim) : normalized_shape(dim), eps(1e-5f) {
        gamma = new float[dim];
        beta = new float[dim];
        // Initialize to identity transform
        for (size_t i = 0; i < dim; i++) {
            gamma[i] = 1.0f;
            beta[i] = 0.0f;
        }
    }

    ~LayerNormParamFP32() {
        delete[] gamma;
        delete[] beta;
    }
};

/**
 * @brief LayerNorm for single vector (1D)
 *
 * LN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * @param out Output vector [len]
 * @param in Input vector [len]
 * @param len Length (feature dimension)
 * @param gamma Scale parameter [len]
 * @param beta Shift parameter [len]
 * @param eps Small constant
 */
void layernorm_fp32(float *out, const float *in, size_t len,
                    const float *gamma, const float *beta, float eps);

/**
 * @brief LayerNorm for 2D matrix (batch of vectors)
 *
 * Normalizes over the last dimension (feature dimension).
 * Input: [seq_len x hidden_dim]
 * Each row is normalized independently.
 *
 * @param out Output matrix [rows x cols]
 * @param in Input matrix [rows x cols]
 * @param rows Number of tokens (sequence length)
 * @param cols Feature dimension (hidden_dim)
 * @param param LayerNorm parameters
 */
void layernorm_2d_fp32(float *out, const float *in, size_t rows, size_t cols,
                       const LayerNormParamFP32 &param);

// ============================================================
// RMSNorm (Root Mean Square Layer Normalization)
// ============================================================

/**
 * @brief RMSNorm: simpler alternative to LayerNorm
 *
 * RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)
 *
 * Used in LLaMA and other efficient Transformers.
 * No centering (no mean subtraction), only scaling.
 *
 * @param out Output vector [len]
 * @param in Input vector [len]
 * @param len Length
 * @param gamma Scale parameter [len]
 * @param eps Small constant
 */
void rmsnorm_fp32(float *out, const float *in, size_t len,
                  const float *gamma, float eps);

/**
 * @brief RMSNorm for 2D matrix
 */
void rmsnorm_2d_fp32(float *out, const float *in, size_t rows, size_t cols,
                     const float *gamma, float eps);

// ============================================================
// Element-wise Operations for Attention
// ============================================================

/**
 * @brief Element-wise multiplication (Hadamard product)
 *
 * Used for gating mechanisms and attention score application.
 *
 * @param out Output array [len]
 * @param a First input [len]
 * @param b Second input [len]
 * @param len Length
 */
void elementwise_mul_fp32(float *out, const float *a, const float *b, size_t len);

/**
 * @brief Element-wise addition
 *
 * Used for residual connections: out = a + b
 */
void elementwise_add_fp32(float *out, const float *a, const float *b, size_t len);

/**
 * @brief Scale vector by scalar
 *
 * Used for attention scaling: x / sqrt(d_k)
 */
void scale_fp32(float *out, const float *in, float scale, size_t len);

/**
 * @brief Scale 2D matrix by scalar
 */
void scale_2d_fp32(float *out, const float *in, float scale, size_t rows, size_t cols);

// ============================================================
// Attention Mask Application
// ============================================================

/**
 * @brief Apply causal (autoregressive) attention mask
 *
 * Sets upper triangular elements to -inf (or large negative value).
 * mask[i][j] = -inf if j > i (can't attend to future tokens)
 *
 * @param out Output attention scores [seq_len x seq_len]
 * @param in Input attention scores [seq_len x seq_len]
 * @param seq_len Sequence length
 * @param mask_value Value to use for masked positions (default: -1e9)
 */
void apply_causal_mask_fp32(float *out, const float *in, size_t seq_len,
                            float mask_value = -1e9f);

/**
 * @brief Apply padding mask to attention scores
 *
 * @param out Output attention scores [rows x cols]
 * @param in Input attention scores [rows x cols]
 * @param padding_mask Boolean mask [cols], true = padded (mask out)
 * @param rows Number of query positions
 * @param cols Number of key positions
 * @param mask_value Value for masked positions
 */
void apply_padding_mask_fp32(float *out, const float *in,
                             const bool *padding_mask,
                             size_t rows, size_t cols,
                             float mask_value = -1e9f);

// ============================================================
// Utility Functions
// ============================================================

/**
 * @brief Compute mean of vector
 */
float mean_fp32(const float *in, size_t len);

/**
 * @brief Compute variance of vector
 */
float variance_fp32(const float *in, size_t len);

/**
 * @brief Compute max of vector
 */
float max_fp32(const float *in, size_t len);

/**
 * @brief Find argmax of vector
 */
size_t argmax_fp32(const float *in, size_t len);

// ============================================================
// Photonic MVM Operations (FP32)
// ============================================================

/**
 * @brief Photonic Matrix-Vector Multiplication (single tile)
 *
 * Performs y = W @ x using FP32 photonic MVM instruction.
 * Limited to VREG_SIZE (32) dimensions.
 *
 * @param out Output vector [out_size]
 * @param weight Weight matrix [out_size x in_size]
 * @param input Input vector [in_size]
 * @param out_size Output dimension (<= 32)
 * @param in_size Input dimension (<= 32)
 */
void photonic_mvm_tile_fp32(float *out, const float *weight, const float *input,
                            size_t out_size, size_t in_size);

/**
 * @brief Photonic MVM with tiling for arbitrary sizes
 *
 * Tiles the computation into 32x32 blocks for large matrices.
 * Accumulates partial results across tiles.
 *
 * @param out Output vector [out_size]
 * @param weight Weight matrix [out_size x in_size] (row-major)
 * @param input Input vector [in_size]
 * @param out_size Output dimension
 * @param in_size Input dimension
 */
void photonic_mvm_fp32(float *out, const float *weight, const float *input,
                       size_t out_size, size_t in_size);

/**
 * @brief Photonic Linear layer: y = W @ x + b
 *
 * @param out Output vector [out_size]
 * @param weight Weight matrix [out_size x in_size]
 * @param input Input vector [in_size]
 * @param bias Bias vector [out_size] (can be nullptr)
 * @param out_size Output dimension
 * @param in_size Input dimension
 */
void photonic_linear_fp32(float *out, const float *weight, const float *input,
                          const float *bias, size_t out_size, size_t in_size);

/**
 * @brief Photonic Matrix-Matrix Multiplication
 *
 * C = A @ B using photonic MVMs (column by column)
 *
 * @param C Output matrix [M x N]
 * @param A Left matrix [M x K]
 * @param B Right matrix [K x N]
 * @param M Rows of A and C
 * @param K Columns of A, Rows of B
 * @param N Columns of B and C
 */
void photonic_matmul_fp32(float *C, const float *A, const float *B,
                          size_t M, size_t K, size_t N);

// ============================================================
// Transformer Attention (Photonic + Electronic)
// ============================================================

/**
 * @brief Q, K, V Projection using Photonic MVM
 *
 * Q = X @ Wq, K = X @ Wk, V = X @ Wv
 *
 * @param Q Output query matrix [seq_len x d_k]
 * @param K Output key matrix [seq_len x d_k]
 * @param V Output value matrix [seq_len x d_v]
 * @param X Input matrix [seq_len x d_model]
 * @param Wq Query weight [d_model x d_k]
 * @param Wk Key weight [d_model x d_k]
 * @param Wv Value weight [d_model x d_v]
 * @param seq_len Sequence length
 * @param d_model Model dimension
 * @param d_k Key/Query dimension
 * @param d_v Value dimension
 */
void attention_qkv_projection_fp32(float *Q, float *K, float *V,
                                   const float *X,
                                   const float *Wq, const float *Wk, const float *Wv,
                                   size_t seq_len, size_t d_model,
                                   size_t d_k, size_t d_v);

/**
 * @brief Scaled Dot-Product Attention
 *
 * Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * Computation:
 * 1. scores = Q @ K^T (photonic)
 * 2. scores = scores / sqrt(d_k) (electronic)
 * 3. [optional] apply causal mask (electronic)
 * 4. attn_weights = softmax(scores) (electronic)
 * 5. output = attn_weights @ V (photonic)
 *
 * @param output Output matrix [seq_len x d_v]
 * @param Q Query matrix [seq_len x d_k]
 * @param K Key matrix [seq_len x d_k]
 * @param V Value matrix [seq_len x d_v]
 * @param seq_len Sequence length
 * @param d_k Key dimension
 * @param d_v Value dimension
 * @param causal Whether to apply causal mask
 */
void scaled_dot_product_attention_fp32(float *output,
                                       const float *Q, const float *K, const float *V,
                                       size_t seq_len, size_t d_k, size_t d_v,
                                       bool causal);

/**
 * @brief Output projection for attention
 *
 * out = concat_heads @ Wo
 *
 * @param output Output [seq_len x d_model]
 * @param concat_heads Concatenated attention heads [seq_len x (num_heads * d_v)]
 * @param Wo Output projection weight [num_heads * d_v x d_model]
 * @param seq_len Sequence length
 * @param num_heads Number of attention heads
 * @param d_v Value dimension per head
 * @param d_model Model dimension
 */
void attention_output_projection_fp32(float *output,
                                      const float *concat_heads,
                                      const float *Wo,
                                      size_t seq_len, size_t num_heads,
                                      size_t d_v, size_t d_model);

// ============================================================
// Transformer Feed-Forward Network
// ============================================================

/**
 * @brief FFN with GELU activation
 *
 * FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
 *
 * Used in BERT, GPT-2, etc.
 *
 * @param output Output [seq_len x d_model]
 * @param input Input [seq_len x d_model]
 * @param W1 First linear weight [d_model x d_ff]
 * @param b1 First bias [d_ff] (can be nullptr)
 * @param W2 Second linear weight [d_ff x d_model]
 * @param b2 Second bias [d_model] (can be nullptr)
 * @param seq_len Sequence length
 * @param d_model Model dimension
 * @param d_ff Feed-forward dimension (typically 4 * d_model)
 */
void ffn_gelu_fp32(float *output, const float *input,
                   const float *W1, const float *b1,
                   const float *W2, const float *b2,
                   size_t seq_len, size_t d_model, size_t d_ff);

/**
 * @brief FFN with SiLU activation (LLaMA style)
 *
 * FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
 *
 * Used in LLaMA, Mistral, etc.
 *
 * @param output Output [seq_len x d_model]
 * @param input Input [seq_len x d_model]
 * @param W_gate Gate projection [d_model x d_ff]
 * @param W_up Up projection [d_model x d_ff]
 * @param W_down Down projection [d_ff x d_model]
 * @param seq_len Sequence length
 * @param d_model Model dimension
 * @param d_ff Feed-forward dimension
 */
void ffn_silu_fp32(float *output, const float *input,
                   const float *W_gate, const float *W_up, const float *W_down,
                   size_t seq_len, size_t d_model, size_t d_ff);

// ============================================================
// Full Transformer Block
// ============================================================

/**
 * @brief Single Transformer decoder block (GPT-style)
 *
 * block(x) = x + Attention(LayerNorm(x))
 *            + FFN(LayerNorm(x + Attention(LayerNorm(x))))
 *
 * @param output Output [seq_len x d_model]
 * @param input Input [seq_len x d_model]
 * @param ln1_gamma LayerNorm1 gamma [d_model]
 * @param ln1_beta LayerNorm1 beta [d_model]
 * @param Wq, Wk, Wv, Wo Attention weights
 * @param ln2_gamma LayerNorm2 gamma [d_model]
 * @param ln2_beta LayerNorm2 beta [d_model]
 * @param W1, b1, W2, b2 FFN weights and biases
 * @param seq_len Sequence length
 * @param d_model Model dimension
 * @param d_k Key dimension
 * @param d_ff FFN hidden dimension
 * @param causal Whether to use causal masking
 */
void transformer_block_fp32(float *output, const float *input,
                            const float *ln1_gamma, const float *ln1_beta,
                            const float *Wq, const float *Wk, const float *Wv, const float *Wo,
                            const float *ln2_gamma, const float *ln2_beta,
                            const float *W1, const float *b1,
                            const float *W2, const float *b2,
                            size_t seq_len, size_t d_model,
                            size_t d_k, size_t d_ff, bool causal);

#endif /* FIONA_NN_TRANSFORMER_H */
