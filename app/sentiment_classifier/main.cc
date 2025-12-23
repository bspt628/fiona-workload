/**
 * @file main.cc
 * @brief Sentiment Classification - PyTorch Compatible Implementation
 *
 * This implementation matches PyTorch's TransformerEncoderLayer exactly:
 * - Multi-head attention with proper head splitting
 * - Pre-LN (norm_first=True) architecture
 * - PyTorch linear: output = input @ weight.T + bias
 *
 * @author FIONA Project
 * @date 2025-12-23
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fiona.h"

// Include exported weights
#include "weights/weights.h"

// ============================================================
// Model Configuration (must match PyTorch training config)
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

// Main buffers
static float embeddings[MAX_SEQ_LEN * D_MODEL];
static float hidden[MAX_SEQ_LEN * D_MODEL];
static float hidden2[MAX_SEQ_LEN * D_MODEL];
static float temp[MAX_SEQ_LEN * D_MODEL];
static float normed[MAX_SEQ_LEN * D_MODEL];

// Attention buffers
static float Q[MAX_SEQ_LEN * D_MODEL];
static float K[MAX_SEQ_LEN * D_MODEL];
static float V[MAX_SEQ_LEN * D_MODEL];
static float attn_scores[N_HEADS * MAX_SEQ_LEN * MAX_SEQ_LEN];
static float attn_out[MAX_SEQ_LEN * D_MODEL];
static float attn_proj[MAX_SEQ_LEN * D_MODEL];

// FFN buffers
static float ffn_hidden[MAX_SEQ_LEN * D_FF];
static float ffn_out[MAX_SEQ_LEN * D_MODEL];

// ============================================================
// PyTorch-compatible Linear Layer
// ============================================================

/**
 * @brief PyTorch-compatible linear layer: output = input @ weight.T + bias
 *
 * PyTorch stores weights as [out_features, in_features]
 * So we compute: output[i] = sum_j(input[j] * weight[i][j]) + bias[i]
 *
 * @param output Output vector [out_dim]
 * @param input Input vector [in_dim]
 * @param weight Weight matrix [out_dim * in_dim] (row-major, out_dim rows)
 * @param bias Bias vector [out_dim] (can be NULL)
 * @param out_dim Output dimension
 * @param in_dim Input dimension
 */
void pytorch_linear(float* output, const float* input,
                    const float* weight, const float* bias,
                    int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            // weight is [out_dim, in_dim], row-major
            // weight[i][j] = weight[i * in_dim + j]
            sum += input[j] * weight[i * in_dim + j];
        }
        output[i] = sum + (bias ? bias[i] : 0.0f);
    }
}

/**
 * @brief Apply linear layer to each row of a 2D input
 */
void pytorch_linear_2d(float* output, const float* input,
                       const float* weight, const float* bias,
                       int seq_len, int out_dim, int in_dim) {
    for (int t = 0; t < seq_len; t++) {
        pytorch_linear(&output[t * out_dim], &input[t * in_dim],
                       weight, bias, out_dim, in_dim);
    }
}

// ============================================================
// LayerNorm (matches PyTorch)
// ============================================================

/**
 * @brief LayerNorm for a single vector
 */
void layer_norm(float* output, const float* input,
                const float* gamma, const float* beta,
                int dim, float eps) {
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) {
        mean += input[i];
    }
    mean /= dim;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= dim;

    // Normalize
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; i++) {
        output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

/**
 * @brief LayerNorm for 2D input (apply to each row)
 */
void layer_norm_2d(float* output, const float* input,
                   const float* gamma, const float* beta,
                   int seq_len, int dim, float eps) {
    for (int t = 0; t < seq_len; t++) {
        layer_norm(&output[t * dim], &input[t * dim], gamma, beta, dim, eps);
    }
}

// ============================================================
// Multi-Head Attention (PyTorch compatible)
// ============================================================

/**
 * @brief Softmax for attention scores (in-place, per row)
 */
void softmax_row(float* scores, int len) {
    // Find max for numerical stability
    float max_val = scores[0];
    for (int i = 1; i < len; i++) {
        if (scores[i] > max_val) max_val = scores[i];
    }

    // Exp and sum
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    // Normalize
    for (int i = 0; i < len; i++) {
        scores[i] /= sum;
    }
}

/**
 * @brief Multi-head self-attention (PyTorch compatible)
 *
 * PyTorch MultiheadAttention:
 * 1. Project Q, K, V: Q = x @ Wq.T, K = x @ Wk.T, V = x @ Wv.T
 * 2. Reshape to [seq, n_heads, d_k]
 * 3. Transpose to [n_heads, seq, d_k]
 * 4. Compute attention per head: softmax(Q @ K.T / sqrt(d_k)) @ V
 * 5. Transpose back to [seq, n_heads, d_k]
 * 6. Reshape to [seq, d_model]
 * 7. Output projection: out = attn @ Wo.T
 */
void multihead_attention(
    float* output,           // [seq_len, d_model]
    const float* input,      // [seq_len, d_model]
    int seq_len,
    // Weights (from PyTorch in_proj split into Q, K, V)
    const float* Wq, const float* bq,  // [d_model, d_model]
    const float* Wk, const float* bk,
    const float* Wv, const float* bv,
    const float* Wo, const float* bo   // [d_model, d_model]
) {
    // 1. Project Q, K, V for all positions
    // Q, K, V each have shape [seq_len, d_model]
    pytorch_linear_2d(Q, input, Wq, bq, seq_len, D_MODEL, D_MODEL);
    pytorch_linear_2d(K, input, Wk, bk, seq_len, D_MODEL, D_MODEL);
    pytorch_linear_2d(V, input, Wv, bv, seq_len, D_MODEL, D_MODEL);

    // 2-4. Compute attention per head
    // Q, K, V are [seq_len, d_model] = [seq_len, n_heads * d_k]
    // We treat this as [seq_len, n_heads, d_k] and compute attention per head

    float scale = 1.0f / sqrtf((float)D_K);

    // For each head
    for (int h = 0; h < N_HEADS; h++) {
        // Compute attention scores for this head
        // scores[i][j] = Q[i, h, :] @ K[j, h, :].T / sqrt(d_k)
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (int k = 0; k < D_K; k++) {
                    // Q[i, h, k] = Q[i * D_MODEL + h * D_K + k]
                    // K[j, h, k] = K[j * D_MODEL + h * D_K + k]
                    int q_idx = i * D_MODEL + h * D_K + k;
                    int k_idx = j * D_MODEL + h * D_K + k;
                    score += Q[q_idx] * K[k_idx];
                }
                attn_scores[h * seq_len * seq_len + i * seq_len + j] = score * scale;
            }

            // Softmax over j dimension
            softmax_row(&attn_scores[h * seq_len * seq_len + i * seq_len], seq_len);
        }

        // Compute attention output for this head
        // attn_out[i, h, :] = sum_j(scores[i][j] * V[j, h, :])
        for (int i = 0; i < seq_len; i++) {
            for (int k = 0; k < D_K; k++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    float score = attn_scores[h * seq_len * seq_len + i * seq_len + j];
                    int v_idx = j * D_MODEL + h * D_K + k;
                    sum += score * V[v_idx];
                }
                // attn_out[i, h, k] = attn_out[i * D_MODEL + h * D_K + k]
                attn_out[i * D_MODEL + h * D_K + k] = sum;
            }
        }
    }

    // 5-7. Output projection: output = attn_out @ Wo.T + bo
    pytorch_linear_2d(output, attn_out, Wo, bo, seq_len, D_MODEL, D_MODEL);
}

// ============================================================
// Feed-Forward Network (PyTorch compatible)
// ============================================================

/**
 * @brief GELU activation (exact formula matching PyTorch)
 */
float gelu(float x) {
    // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

/**
 * @brief Feed-forward network: GELU(x @ W1.T + b1) @ W2.T + b2
 */
void feed_forward(
    float* output,           // [seq_len, d_model]
    const float* input,      // [seq_len, d_model]
    int seq_len,
    const float* W1, const float* b1,  // [d_ff, d_model]
    const float* W2, const float* b2   // [d_model, d_ff]
) {
    // First linear + GELU
    for (int t = 0; t < seq_len; t++) {
        pytorch_linear(&ffn_hidden[t * D_FF], &input[t * D_MODEL],
                       W1, b1, D_FF, D_MODEL);

        // Apply GELU
        for (int i = 0; i < D_FF; i++) {
            ffn_hidden[t * D_FF + i] = gelu(ffn_hidden[t * D_FF + i]);
        }
    }

    // Second linear
    pytorch_linear_2d(output, ffn_hidden, W2, b2, seq_len, D_MODEL, D_FF);
}

// ============================================================
// Transformer Encoder Layer (PyTorch Pre-LN compatible)
// ============================================================

/**
 * @brief Single Transformer encoder layer (Pre-LN, norm_first=True)
 *
 * PyTorch TransformerEncoderLayer with norm_first=True:
 *   x = x + self_attn(norm1(x))
 *   x = x + ffn(norm2(x))
 */
void transformer_encoder_layer(
    float* output,           // [seq_len, d_model]
    const float* input,      // [seq_len, d_model]
    int seq_len,
    // Self-attention weights
    const float* Wq, const float* bq,
    const float* Wk, const float* bk,
    const float* Wv, const float* bv,
    const float* Wo, const float* bo,
    // FFN weights
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    // LayerNorm weights
    const float* ln1_gamma, const float* ln1_beta,
    const float* ln2_gamma, const float* ln2_beta
) {
    // 1. Pre-norm + Self-attention + Residual
    layer_norm_2d(normed, input, ln1_gamma, ln1_beta, seq_len, D_MODEL, 1e-5f);
    multihead_attention(attn_proj, normed, seq_len, Wq, bq, Wk, bk, Wv, bv, Wo, bo);

    // Residual connection
    for (int i = 0; i < seq_len * D_MODEL; i++) {
        temp[i] = input[i] + attn_proj[i];
    }

    // 2. Pre-norm + FFN + Residual
    layer_norm_2d(normed, temp, ln2_gamma, ln2_beta, seq_len, D_MODEL, 1e-5f);
    feed_forward(ffn_out, normed, seq_len, W1, b1, W2, b2);

    // Residual connection
    for (int i = 0; i < seq_len * D_MODEL; i++) {
        output[i] = temp[i] + ffn_out[i];
    }
}

// ============================================================
// Embedding Layer
// ============================================================

void embed_tokens(const int* token_ids, int seq_len, float* output) {
    for (int i = 0; i < seq_len; i++) {
        int token_id = token_ids[i];

        // Bounds check
        if (token_id < 0 || token_id >= VOCAB_SIZE) {
            token_id = 100;  // [UNK]
        }

        // Token embedding + Position embedding
        for (int j = 0; j < D_MODEL; j++) {
            output[i * D_MODEL + j] =
                token_embedding[token_id * D_MODEL + j] +
                position_embedding[i * D_MODEL + j];
        }
    }
}

// ============================================================
// Classification Head
// ============================================================

int classify(const float* encoder_output, float* probs) {
    // Pool [CLS] token (first position)
    float cls_hidden[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        cls_hidden[i] = encoder_output[i];
    }

    // Pooler: tanh(x @ W.T + b)
    float pooled[D_MODEL];
    pytorch_linear(pooled, cls_hidden, pooler_weight, pooler_bias, D_MODEL, D_MODEL);
    for (int i = 0; i < D_MODEL; i++) {
        pooled[i] = tanhf(pooled[i]);
    }

    // Classifier: x @ W.T + b
    float logits[NUM_LABELS];
    pytorch_linear(logits, pooled, classifier_weight, classifier_bias, NUM_LABELS, D_MODEL);

    // Softmax
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

    // Argmax
    return (probs[1] > probs[0]) ? 1 : 0;
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("============================================\n");
    printf("  TinySentiment Classifier (PyTorch Compat)\n");
    printf("============================================\n\n");

    printf("Model configuration:\n");
    printf("  d_model:      %d\n", D_MODEL);
    printf("  n_heads:      %d\n", N_HEADS);
    printf("  d_k:          %d\n", D_K);
    printf("  d_ff:         %d\n", D_FF);
    printf("  n_layers:     %d\n", N_LAYERS);
    printf("  vocab_size:   %d\n\n", VOCAB_SIZE);

    // Test sentences with pre-tokenized IDs (BERT WordPiece)
    const char* test_sentences[] = {
        "This movie is great",
        "I hate this film",
        "The acting was wonderful",
        "Terrible waste of time",
    };

    // Pre-tokenized token IDs (BERT WordPiece tokenizer)
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

        // 1. Copy pre-tokenized IDs
        int token_ids[MAX_SEQ_LEN];
        for (int i = 0; i < seq_len; i++) {
            token_ids[i] = pretokenized[t][i];
        }

        // 2. Embed
        embed_tokens(token_ids, seq_len, embeddings);

        // 3. Transformer encoder - Layer 0
        transformer_encoder_layer(
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

        // 4. Transformer encoder - Layer 1
        transformer_encoder_layer(
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

        // 5. Classify
        float probs[NUM_LABELS];
        int pred = classify(hidden2, probs);

        // Print result
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
