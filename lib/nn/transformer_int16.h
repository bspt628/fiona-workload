/**
 * @file transformer_int16.h
 * @brief INT16 Transformer operations for FIONA RTL (Verilator)
 *
 * This module provides INT16 versions of Transformer operations that
 * work with the FIONA-V RTL implementation (which only supports INT16).
 *
 * Key differences from FP32 version:
 * - Photonic operations use INT16 (dotp, mvm via palu.cc)
 * - Non-linear operations (softmax, gelu, layernorm) use float internally
 *   but quantize inputs/outputs to INT16
 * - Requires quantization scale factors for accuracy
 *
 * @author FIONA Project
 * @date 2025-12-22
 */

#ifndef FIONA_NN_TRANSFORMER_INT16_H
#define FIONA_NN_TRANSFORMER_INT16_H

#include "math/all.h"
#include "base/config.h"
#include <math.h>
#include <stdint.h>

// ============================================================
// Quantization Parameters
// ============================================================

#define QUANT_SCALE 128.0f    // Default quantization scale
#define QUANT_MAX   32767     // INT16 max
#define QUANT_MIN   (-32768)  // INT16 min

/**
 * @brief Quantization configuration
 */
struct QuantConfig {
    float scale;      // Quantization scale factor
    int zero_point;   // Zero point (usually 0 for symmetric)

    QuantConfig(float s = QUANT_SCALE, int zp = 0)
        : scale(s), zero_point(zp) {}
};

// ============================================================
// Quantization Helpers
// ============================================================

/**
 * @brief Quantize float to int16_t
 */
static inline int16_t quantize(float val, float scale) {
    int32_t q = (int32_t)roundf(val * scale);
    if (q > QUANT_MAX) q = QUANT_MAX;
    if (q < QUANT_MIN) q = QUANT_MIN;
    return (int16_t)q;
}

/**
 * @brief Dequantize int16_t to float
 */
static inline float dequantize(int16_t val, float scale) {
    return (float)val / scale;
}

/**
 * @brief Quantize float array to int16_t array
 */
void quantize_array(int16_t *out, const float *in, size_t len, float scale);

/**
 * @brief Dequantize int16_t array to float array
 */
void dequantize_array(float *out, const int16_t *in, size_t len, float scale);

// ============================================================
// Softmax (INT16 input/output, float internal)
// ============================================================

/**
 * @brief Softmax with INT16 I/O
 *
 * 1. Dequantize input to float
 * 2. Compute softmax in float
 * 3. Quantize output to INT16
 */
void softmax_int16(int16_t *out, const int16_t *in, size_t len,
                   float in_scale, float out_scale);

/**
 * @brief Row-wise softmax for 2D matrix (INT16)
 */
void softmax_2d_int16(int16_t *out, const int16_t *in, size_t rows, size_t cols,
                      float in_scale, float out_scale);

// ============================================================
// GELU (INT16 input/output)
// ============================================================

/**
 * @brief GELU activation with INT16 I/O
 */
void gelu_int16(int16_t *out, const int16_t *in, size_t len,
                float in_scale, float out_scale);

/**
 * @brief GELU for 2D matrix (INT16)
 */
void gelu_2d_int16(int16_t *out, const int16_t *in, size_t rows, size_t cols,
                   float in_scale, float out_scale);

// ============================================================
// ReLU (INT16, can be done directly)
// ============================================================

/**
 * @brief ReLU activation (INT16)
 *
 * Simple: out = max(in, 0)
 */
void relu_int16(int16_t *out, const int16_t *in, size_t len);

// ============================================================
// LayerNorm (INT16 input/output)
// ============================================================

/**
 * @brief LayerNorm parameters for INT16
 */
struct LayerNormParamInt16 {
    float *gamma;  // Scale parameter (float, applied after norm)
    float *beta;   // Shift parameter (float)
    float eps;
    size_t dim;

    LayerNormParamInt16(size_t d) : dim(d), eps(1e-5f) {
        gamma = new float[d];
        beta = new float[d];
        for (size_t i = 0; i < d; i++) {
            gamma[i] = 1.0f;
            beta[i] = 0.0f;
        }
    }

    ~LayerNormParamInt16() {
        delete[] gamma;
        delete[] beta;
    }
};

/**
 * @brief LayerNorm with INT16 I/O
 */
void layernorm_int16(int16_t *out, const int16_t *in, size_t len,
                     const float *gamma, const float *beta, float eps,
                     float in_scale, float out_scale);

/**
 * @brief LayerNorm 2D with INT16 I/O
 */
void layernorm_2d_int16(int16_t *out, const int16_t *in, size_t rows, size_t cols,
                        const LayerNormParamInt16 &param,
                        float in_scale, float out_scale);

// ============================================================
// Element-wise Operations (INT16)
// ============================================================

/**
 * @brief Element-wise addition (INT16)
 * Note: May need rescaling if scales differ
 */
void elementwise_add_int16(int16_t *out, const int16_t *a, const int16_t *b, size_t len);

/**
 * @brief Scale INT16 array by float scalar
 */
void scale_int16(int16_t *out, const int16_t *in, float scale_factor, size_t len,
                 float in_scale, float out_scale);

// ============================================================
// Attention Mask (INT16)
// ============================================================

/**
 * @brief Apply causal mask (INT16)
 * Sets upper triangular to minimum value
 */
void apply_causal_mask_int16(int16_t *out, const int16_t *in, size_t seq_len);

// ============================================================
// Photonic MVM (INT16) - Uses existing palu.cc functions
// ============================================================

/**
 * @brief Photonic MVM tile (INT16)
 *
 * Uses FIONA-V RTL's MVM instruction via palu.cc
 */
void photonic_mvm_tile_int16(int16_t *out, const int16_t *weight, const int16_t *input,
                             size_t out_size, size_t in_size);

/**
 * @brief Photonic MVM with tiling (INT16)
 *
 * Wraps tiled_mvm_strided for arbitrary sizes
 */
void photonic_mvm_int16(int16_t *out, const int16_t *weight, const int16_t *input,
                        size_t out_size, size_t in_size);

/**
 * @brief Photonic Linear: y = W @ x + b (INT16)
 *
 * Bias is added in float then quantized
 */
void photonic_linear_int16(int16_t *out, const int16_t *weight, const int16_t *input,
                           const float *bias, size_t out_size, size_t in_size,
                           float w_scale, float in_scale, float out_scale);

/**
 * @brief Photonic Matrix-Matrix Multiplication (INT16)
 *
 * C = A @ B using tiled MVMs
 */
void photonic_matmul_int16(int16_t *C, const int16_t *A, const int16_t *B,
                           size_t M, size_t K, size_t N);

// ============================================================
// Transformer Attention (INT16)
// ============================================================

/**
 * @brief Q, K, V Projection (INT16)
 */
void attention_qkv_projection_int16(int16_t *Q, int16_t *K, int16_t *V,
                                    const int16_t *X,
                                    const int16_t *Wq, const int16_t *Wk, const int16_t *Wv,
                                    size_t seq_len, size_t d_model,
                                    size_t d_k, size_t d_v,
                                    float scale);

/**
 * @brief Scaled Dot-Product Attention (INT16)
 *
 * Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 */
void scaled_dot_product_attention_int16(int16_t *output,
                                        const int16_t *Q, const int16_t *K, const int16_t *V,
                                        size_t seq_len, size_t d_k, size_t d_v,
                                        bool causal, float scale);

// ============================================================
// Transformer FFN (INT16)
// ============================================================

/**
 * @brief FFN with GELU (INT16)
 *
 * FFN(x) = GELU(x @ W1) @ W2
 */
void ffn_gelu_int16(int16_t *output, const int16_t *input,
                    const int16_t *W1, const int16_t *W2,
                    size_t seq_len, size_t d_model, size_t d_ff,
                    float scale);

/**
 * @brief FFN with ReLU (INT16) - simpler, fully RTL-compatible
 *
 * FFN(x) = ReLU(x @ W1) @ W2
 */
void ffn_relu_int16(int16_t *output, const int16_t *input,
                    const int16_t *W1, const int16_t *W2,
                    size_t seq_len, size_t d_model, size_t d_ff,
                    float scale);

// ============================================================
// Full Transformer Block (INT16)
// ============================================================

/**
 * @brief Transformer block (INT16)
 *
 * Simplified version without LayerNorm for pure RTL execution:
 * block(x) = x + Attention(x) + FFN(x + Attention(x))
 */
void transformer_block_int16(int16_t *output, const int16_t *input,
                             const int16_t *Wq, const int16_t *Wk,
                             const int16_t *Wv, const int16_t *Wo,
                             const int16_t *W1, const int16_t *W2,
                             size_t seq_len, size_t d_model,
                             size_t d_k, size_t d_ff,
                             bool causal, float scale);

#endif /* FIONA_NN_TRANSFORMER_INT16_H */
