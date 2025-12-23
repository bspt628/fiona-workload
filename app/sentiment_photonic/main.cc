/**
 * @file main.cc
 * @brief Sentiment Classification with Photonic Acceleration
 *
 * This implementation uses FIONA photonic MVM operations for linear layers.
 * The photonic model can be selected via environment variable:
 *   FIONA_PHOTONIC_MODEL=ideal|noisy|mzi_realistic|quantized|all_effects
 *
 * Electronic operations (LayerNorm, GELU, Softmax) remain on CPU.
 * Photonic operations (Linear/MVM) use the FIONA accelerator.
 *
 * @author FIONA Project
 * @date 2025-12-23
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fiona.h"
#include "base/instr.h"
#include "nn/transformer.h"

// Include exported weights
#include "weights/weights.h"

// ============================================================
// Model Configuration
// ============================================================

#define D_MODEL        128
#define N_HEADS        2
#define D_K            (D_MODEL / N_HEADS)  // 64
#define D_FF           256
#define N_LAYERS       2
#define MAX_SEQ_LEN    64
#define NUM_LABELS     2
#define VOCAB_SIZE     30522

// ============================================================
// Static Buffers
// ============================================================

static float embeddings[MAX_SEQ_LEN * D_MODEL];
static float hidden[MAX_SEQ_LEN * D_MODEL];
static float hidden2[MAX_SEQ_LEN * D_MODEL];
static float temp[MAX_SEQ_LEN * D_MODEL];
static float normed[MAX_SEQ_LEN * D_MODEL];

static float Q[MAX_SEQ_LEN * D_MODEL];
static float K[MAX_SEQ_LEN * D_MODEL];
static float V[MAX_SEQ_LEN * D_MODEL];
static float attn_scores[N_HEADS * MAX_SEQ_LEN * MAX_SEQ_LEN];
static float attn_out[MAX_SEQ_LEN * D_MODEL];
static float attn_proj[MAX_SEQ_LEN * D_MODEL];

static float ffn_hidden[MAX_SEQ_LEN * D_FF];
static float ffn_out[MAX_SEQ_LEN * D_MODEL];

// ============================================================
// Electronic Operations (CPU)
// ============================================================

void layer_norm(float* output, const float* input,
                const float* gamma, const float* beta,
                int dim, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += input[i];
    mean /= dim;

    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= dim;

    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; i++) {
        output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

void layer_norm_2d(float* output, const float* input,
                   const float* gamma, const float* beta,
                   int seq_len, int dim, float eps) {
    for (int t = 0; t < seq_len; t++) {
        layer_norm(&output[t * dim], &input[t * dim], gamma, beta, dim, eps);
    }
}

void softmax_row(float* scores, int len) {
    float max_val = scores[0];
    for (int i = 1; i < len; i++) {
        if (scores[i] > max_val) max_val = scores[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    for (int i = 0; i < len; i++) {
        scores[i] /= sum;
    }
}

float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ============================================================
// Photonic Operations (FIONA Accelerator)
// ============================================================

/**
 * @brief Photonic linear layer using FIONA MVM
 *
 * Uses photonic_linear_fp32 from transformer.cc which internally
 * calls the FIONA MVM instructions. The photonic model is selected
 * via FIONA_PHOTONIC_MODEL environment variable.
 */
void photonic_linear_layer(float* output, const float* input,
                           const float* weight, const float* bias,
                           int out_dim, int in_dim) {
    // Use FIONA photonic MVM
    photonic_linear_fp32(output, weight, input, bias, out_dim, in_dim);
}

/**
 * @brief Photonic linear for 2D input (batch of vectors)
 */
void photonic_linear_2d(float* output, const float* input,
                        const float* weight, const float* bias,
                        int seq_len, int out_dim, int in_dim) {
    for (int t = 0; t < seq_len; t++) {
        photonic_linear_layer(&output[t * out_dim], &input[t * in_dim],
                              weight, bias, out_dim, in_dim);
    }
}

// ============================================================
// Multi-Head Attention (Photonic + Electronic)
// ============================================================

void multihead_attention_photonic(
    float* output, const float* input, int seq_len,
    const float* Wq, const float* bq,
    const float* Wk, const float* bk,
    const float* Wv, const float* bv,
    const float* Wo, const float* bo
) {
    // Q, K, V projections (PHOTONIC)
    photonic_linear_2d(Q, input, Wq, bq, seq_len, D_MODEL, D_MODEL);
    photonic_linear_2d(K, input, Wk, bk, seq_len, D_MODEL, D_MODEL);
    photonic_linear_2d(V, input, Wv, bv, seq_len, D_MODEL, D_MODEL);

    float scale = 1.0f / sqrtf((float)D_K);

    // Attention computation per head (ELECTRONIC - complex data dependencies)
    for (int h = 0; h < N_HEADS; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (int k = 0; k < D_K; k++) {
                    int q_idx = i * D_MODEL + h * D_K + k;
                    int k_idx = j * D_MODEL + h * D_K + k;
                    score += Q[q_idx] * K[k_idx];
                }
                attn_scores[h * seq_len * seq_len + i * seq_len + j] = score * scale;
            }
            softmax_row(&attn_scores[h * seq_len * seq_len + i * seq_len], seq_len);
        }

        for (int i = 0; i < seq_len; i++) {
            for (int k = 0; k < D_K; k++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    float score = attn_scores[h * seq_len * seq_len + i * seq_len + j];
                    int v_idx = j * D_MODEL + h * D_K + k;
                    sum += score * V[v_idx];
                }
                attn_out[i * D_MODEL + h * D_K + k] = sum;
            }
        }
    }

    // Output projection (PHOTONIC)
    photonic_linear_2d(output, attn_out, Wo, bo, seq_len, D_MODEL, D_MODEL);
}

// ============================================================
// Feed-Forward Network (Photonic + Electronic)
// ============================================================

void feed_forward_photonic(
    float* output, const float* input, int seq_len,
    const float* W1, const float* b1,
    const float* W2, const float* b2
) {
    // First linear (PHOTONIC)
    for (int t = 0; t < seq_len; t++) {
        photonic_linear_layer(&ffn_hidden[t * D_FF], &input[t * D_MODEL],
                              W1, b1, D_FF, D_MODEL);

        // GELU activation (ELECTRONIC)
        for (int i = 0; i < D_FF; i++) {
            ffn_hidden[t * D_FF + i] = gelu(ffn_hidden[t * D_FF + i]);
        }
    }

    // Second linear (PHOTONIC)
    photonic_linear_2d(output, ffn_hidden, W2, b2, seq_len, D_MODEL, D_FF);
}

// ============================================================
// Transformer Encoder Layer (Photonic + Electronic)
// ============================================================

void transformer_encoder_layer_photonic(
    float* output, const float* input, int seq_len,
    const float* Wq, const float* bq,
    const float* Wk, const float* bk,
    const float* Wv, const float* bv,
    const float* Wo, const float* bo,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* ln1_gamma, const float* ln1_beta,
    const float* ln2_gamma, const float* ln2_beta
) {
    // Pre-norm (ELECTRONIC)
    layer_norm_2d(normed, input, ln1_gamma, ln1_beta, seq_len, D_MODEL, 1e-5f);

    // Multi-head attention (PHOTONIC + ELECTRONIC)
    multihead_attention_photonic(attn_proj, normed, seq_len,
                                 Wq, bq, Wk, bk, Wv, bv, Wo, bo);

    // Residual connection (ELECTRONIC)
    for (int i = 0; i < seq_len * D_MODEL; i++) {
        temp[i] = input[i] + attn_proj[i];
    }

    // Pre-norm (ELECTRONIC)
    layer_norm_2d(normed, temp, ln2_gamma, ln2_beta, seq_len, D_MODEL, 1e-5f);

    // FFN (PHOTONIC + ELECTRONIC)
    feed_forward_photonic(ffn_out, normed, seq_len, W1, b1, W2, b2);

    // Residual connection (ELECTRONIC)
    for (int i = 0; i < seq_len * D_MODEL; i++) {
        output[i] = temp[i] + ffn_out[i];
    }
}

// ============================================================
// Embedding Layer (Electronic - Memory bound)
// ============================================================

void embed_tokens(const int* token_ids, int seq_len, float* output) {
    for (int i = 0; i < seq_len; i++) {
        int token_id = token_ids[i];
        if (token_id < 0 || token_id >= VOCAB_SIZE) token_id = 100;

        for (int j = 0; j < D_MODEL; j++) {
            output[i * D_MODEL + j] =
                token_embedding[token_id * D_MODEL + j] +
                position_embedding[i * D_MODEL + j];
        }
    }
}

// ============================================================
// Classification Head (Photonic + Electronic)
// ============================================================

int classify_photonic(const float* encoder_output, float* probs) {
    float cls_hidden[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        cls_hidden[i] = encoder_output[i];
    }

    // Pooler linear (PHOTONIC)
    float pooled[D_MODEL];
    photonic_linear_layer(pooled, cls_hidden, pooler_weight, pooler_bias,
                          D_MODEL, D_MODEL);

    // Tanh activation (ELECTRONIC)
    for (int i = 0; i < D_MODEL; i++) {
        pooled[i] = tanhf(pooled[i]);
    }

    // Classifier linear (PHOTONIC)
    float logits[NUM_LABELS];
    photonic_linear_layer(logits, pooled, classifier_weight, classifier_bias,
                          NUM_LABELS, D_MODEL);

    // Softmax (ELECTRONIC)
    float max_logit = logits[0];
    for (int i = 1; i < NUM_LABELS; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < NUM_LABELS; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < NUM_LABELS; i++) {
        probs[i] /= sum;
    }

    return (probs[1] > probs[0]) ? 1 : 0;
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("============================================\n");
    printf("  TinySentiment with Photonic Acceleration\n");
    printf("============================================\n\n");

    printf("Model configuration:\n");
    printf("  d_model:      %d\n", D_MODEL);
    printf("  n_heads:      %d\n", N_HEADS);
    printf("  d_k:          %d\n", D_K);
    printf("  d_ff:         %d\n", D_FF);
    printf("  n_layers:     %d\n", N_LAYERS);
    printf("  vocab_size:   %d\n\n", VOCAB_SIZE);

    printf("Photonic model: Set via FIONA_PHOTONIC_MODEL env var\n");
    printf("  Available: ideal, noisy, mzi_realistic, quantized, all_effects\n\n");

    // Test sentences with pre-tokenized IDs
    const char* test_sentences[] = {
        "This movie is great",
        "I hate this film",
        "The acting was wonderful",
        "Terrible waste of time",
    };

    const int pretokenized[][32] = {
        {101, 2023, 3185, 2003, 2307, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {101, 1045, 5223, 2023, 2143, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {101, 1996, 3772, 2001, 6919, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {101, 6659, 5949, 1997, 2051, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };

    const int expected[] = {1, 0, 1, 0};
    const int n_tests = 4;
    const int seq_len = 32;

    printf("Running inference on %d test sentences...\n\n", n_tests);

    int correct = 0;

    for (int t = 0; t < n_tests; t++) {
        printf("[%d] Input: \"%s\"\n", t + 1, test_sentences[t]);

        int token_ids[MAX_SEQ_LEN];
        for (int i = 0; i < seq_len; i++) {
            token_ids[i] = pretokenized[t][i];
        }

        // Embedding (Electronic)
        embed_tokens(token_ids, seq_len, embeddings);

        // Transformer Layer 0 (Photonic + Electronic)
        transformer_encoder_layer_photonic(
            hidden, embeddings, seq_len,
            layer0_Wq, layer0_bq,
            layer0_Wk, layer0_bk,
            layer0_Wv, layer0_bv,
            layer0_Wo, layer0_bo,
            layer0_W1, layer0_b1,
            layer0_W2, layer0_b2,
            layer0_ln1_gamma, layer0_ln1_beta,
            layer0_ln2_gamma, layer0_ln2_beta
        );

        // Transformer Layer 1 (Photonic + Electronic)
        transformer_encoder_layer_photonic(
            hidden2, hidden, seq_len,
            layer1_Wq, layer1_bq,
            layer1_Wk, layer1_bk,
            layer1_Wv, layer1_bv,
            layer1_Wo, layer1_bo,
            layer1_W1, layer1_b1,
            layer1_W2, layer1_b2,
            layer1_ln1_gamma, layer1_ln1_beta,
            layer1_ln2_gamma, layer1_ln2_beta
        );

        // Classification (Photonic + Electronic)
        float probs[NUM_LABELS];
        int pred = classify_photonic(hidden2, probs);

        const char* label = (pred == 1) ? "POSITIVE" : "NEGATIVE";
        const char* expected_label = (expected[t] == 1) ? "POSITIVE" : "NEGATIVE";
        const char* status = (pred == expected[t]) ? "CORRECT" : "WRONG";

        printf("    Prediction: %s (%.1f%%)\n", label, probs[pred] * 100.0f);
        printf("    Expected:   %s [%s]\n\n", expected_label, status);

        if (pred == expected[t]) correct++;
    }

    printf("============================================\n");
    printf("  Results: %d/%d correct (%.1f%% accuracy)\n",
           correct, n_tests, 100.0f * correct / n_tests);
    printf("============================================\n");

    return 0;
}
