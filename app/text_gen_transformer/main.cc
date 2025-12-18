/**
 * @file main.cc
 * @brief Character-level Text Generation using Photonic Transformer
 *
 * Demonstrates autoregressive text generation using a small Transformer model.
 * Uses photonic MVM for linear layers (embedding projection, attention, FFN).
 *
 * Architecture:
 * - Vocabulary: 128 ASCII characters
 * - seq_len: 32 (context window)
 * - d_model: 64 (embedding dimension)
 * - n_layers: 2 (Transformer blocks)
 * - d_k: 32 (attention dimension)
 * - d_ff: 128 (FFN hidden dimension)
 *
 * Generation process:
 * 1. Encode input text as character indices
 * 2. Embed characters to vectors
 * 3. Add positional encoding
 * 4. Pass through Transformer blocks
 * 5. Project to vocabulary logits
 * 6. Sample next character
 * 7. Append and repeat
 *
 * To use pretrained weights:
 * 1. Run: python scripts/train_char_transformer.py --demo
 * 2. Compile with: -DUSE_PRETRAINED_WEIGHTS
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

// Include pretrained weights if available
#ifdef USE_PRETRAINED_WEIGHTS
#include "weights/pretrained_weights.h"
#endif

// ============================================================
// Model Configuration
// ============================================================

#define VOCAB_SIZE    128   // ASCII characters
#define SEQ_LEN       32    // Context window
#define D_MODEL       64    // Embedding dimension
#define D_K           32    // Attention key/query dimension
#define D_FF          128   // FFN hidden dimension
#define N_LAYERS      2     // Number of Transformer blocks

#define MAX_GEN_LEN   100   // Maximum generation length

// ============================================================
// Model Weights (statically allocated)
// ============================================================

// Embedding matrix [VOCAB_SIZE x D_MODEL]
static float embedding_weight[VOCAB_SIZE * D_MODEL];

// Positional encoding [SEQ_LEN x D_MODEL]
static float pos_encoding[SEQ_LEN * D_MODEL];

// Transformer block weights (per layer)
struct TransformerBlockWeights {
    float ln1_gamma[D_MODEL];
    float ln1_beta[D_MODEL];
    float Wq[D_K * D_MODEL];
    float Wk[D_K * D_MODEL];
    float Wv[D_K * D_MODEL];
    float Wo[D_MODEL * D_K];
    float ln2_gamma[D_MODEL];
    float ln2_beta[D_MODEL];
    float W1[D_FF * D_MODEL];
    float b1[D_FF];
    float W2[D_MODEL * D_FF];
    float b2[D_MODEL];
};

static TransformerBlockWeights layer_weights[N_LAYERS];

// Output projection [VOCAB_SIZE x D_MODEL]
static float output_weight[VOCAB_SIZE * D_MODEL];
static float output_bias[VOCAB_SIZE];

// ============================================================
// Pseudo-random number generator (deterministic)
// ============================================================

static unsigned int g_seed = 42;

float randf() {
    g_seed = g_seed * 1103515245 + 12345;
    return ((float)(g_seed % 10000) / 10000.0f) * 2.0f - 1.0f;
}

void set_seed(unsigned int seed) {
    g_seed = seed;
}

// Xavier/Glorot initialization
float xavier_init(size_t fan_in, size_t fan_out) {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    return randf() * limit;
}

// ============================================================
// Weight Initialization
// ============================================================

void init_embedding() {
    // Initialize embedding matrix with Xavier initialization
    for (size_t i = 0; i < VOCAB_SIZE * D_MODEL; i++) {
        embedding_weight[i] = xavier_init(VOCAB_SIZE, D_MODEL);
    }
}

void init_positional_encoding() {
    // Sinusoidal positional encoding (Attention Is All You Need)
    for (size_t pos = 0; pos < SEQ_LEN; pos++) {
        for (size_t i = 0; i < D_MODEL; i++) {
            float angle = (float)pos / powf(10000.0f, (float)(2 * (i / 2)) / D_MODEL);
            if (i % 2 == 0) {
                pos_encoding[pos * D_MODEL + i] = sinf(angle);
            } else {
                pos_encoding[pos * D_MODEL + i] = cosf(angle);
            }
        }
    }
}

void init_transformer_weights() {
    for (int l = 0; l < N_LAYERS; l++) {
        // LayerNorm 1
        for (size_t i = 0; i < D_MODEL; i++) {
            layer_weights[l].ln1_gamma[i] = 1.0f;
            layer_weights[l].ln1_beta[i] = 0.0f;
        }

        // Attention weights
        for (size_t i = 0; i < D_K * D_MODEL; i++) {
            layer_weights[l].Wq[i] = xavier_init(D_MODEL, D_K);
            layer_weights[l].Wk[i] = xavier_init(D_MODEL, D_K);
            layer_weights[l].Wv[i] = xavier_init(D_MODEL, D_K);
        }
        for (size_t i = 0; i < D_MODEL * D_K; i++) {
            layer_weights[l].Wo[i] = xavier_init(D_K, D_MODEL);
        }

        // LayerNorm 2
        for (size_t i = 0; i < D_MODEL; i++) {
            layer_weights[l].ln2_gamma[i] = 1.0f;
            layer_weights[l].ln2_beta[i] = 0.0f;
        }

        // FFN weights
        for (size_t i = 0; i < D_FF * D_MODEL; i++) {
            layer_weights[l].W1[i] = xavier_init(D_MODEL, D_FF);
        }
        for (size_t i = 0; i < D_FF; i++) {
            layer_weights[l].b1[i] = 0.0f;
        }
        for (size_t i = 0; i < D_MODEL * D_FF; i++) {
            layer_weights[l].W2[i] = xavier_init(D_FF, D_MODEL);
        }
        for (size_t i = 0; i < D_MODEL; i++) {
            layer_weights[l].b2[i] = 0.0f;
        }
    }
}

void init_output_layer() {
    for (size_t i = 0; i < VOCAB_SIZE * D_MODEL; i++) {
        output_weight[i] = xavier_init(D_MODEL, VOCAB_SIZE);
    }
    for (size_t i = 0; i < VOCAB_SIZE; i++) {
        output_bias[i] = 0.0f;
    }
}

#ifdef USE_PRETRAINED_WEIGHTS
void load_pretrained_weights() {
    printf("Loading pretrained weights...\n");

    // Load embedding
    memcpy(embedding_weight, PRETRAINED_EMBEDDING, sizeof(embedding_weight));

    // Load output layer
    memcpy(output_weight, PRETRAINED_OUTPUT_WEIGHT, sizeof(output_weight));
    memcpy(output_bias, PRETRAINED_OUTPUT_BIAS, sizeof(output_bias));

    // Load layer 0
    memcpy(layer_weights[0].ln1_gamma, PRETRAINED_L0_LN1_GAMMA, sizeof(layer_weights[0].ln1_gamma));
    memcpy(layer_weights[0].ln1_beta, PRETRAINED_L0_LN1_BETA, sizeof(layer_weights[0].ln1_beta));
    memcpy(layer_weights[0].Wq, PRETRAINED_L0_WQ, sizeof(layer_weights[0].Wq));
    memcpy(layer_weights[0].Wk, PRETRAINED_L0_WK, sizeof(layer_weights[0].Wk));
    memcpy(layer_weights[0].Wv, PRETRAINED_L0_WV, sizeof(layer_weights[0].Wv));
    memcpy(layer_weights[0].Wo, PRETRAINED_L0_WO, sizeof(layer_weights[0].Wo));
    memcpy(layer_weights[0].ln2_gamma, PRETRAINED_L0_LN2_GAMMA, sizeof(layer_weights[0].ln2_gamma));
    memcpy(layer_weights[0].ln2_beta, PRETRAINED_L0_LN2_BETA, sizeof(layer_weights[0].ln2_beta));
    memcpy(layer_weights[0].W1, PRETRAINED_L0_FFN_W1, sizeof(layer_weights[0].W1));
    memcpy(layer_weights[0].b1, PRETRAINED_L0_FFN_B1, sizeof(layer_weights[0].b1));
    memcpy(layer_weights[0].W2, PRETRAINED_L0_FFN_W2, sizeof(layer_weights[0].W2));
    memcpy(layer_weights[0].b2, PRETRAINED_L0_FFN_B2, sizeof(layer_weights[0].b2));

    // Load layer 1
    memcpy(layer_weights[1].ln1_gamma, PRETRAINED_L1_LN1_GAMMA, sizeof(layer_weights[1].ln1_gamma));
    memcpy(layer_weights[1].ln1_beta, PRETRAINED_L1_LN1_BETA, sizeof(layer_weights[1].ln1_beta));
    memcpy(layer_weights[1].Wq, PRETRAINED_L1_WQ, sizeof(layer_weights[1].Wq));
    memcpy(layer_weights[1].Wk, PRETRAINED_L1_WK, sizeof(layer_weights[1].Wk));
    memcpy(layer_weights[1].Wv, PRETRAINED_L1_WV, sizeof(layer_weights[1].Wv));
    memcpy(layer_weights[1].Wo, PRETRAINED_L1_WO, sizeof(layer_weights[1].Wo));
    memcpy(layer_weights[1].ln2_gamma, PRETRAINED_L1_LN2_GAMMA, sizeof(layer_weights[1].ln2_gamma));
    memcpy(layer_weights[1].ln2_beta, PRETRAINED_L1_LN2_BETA, sizeof(layer_weights[1].ln2_beta));
    memcpy(layer_weights[1].W1, PRETRAINED_L1_FFN_W1, sizeof(layer_weights[1].W1));
    memcpy(layer_weights[1].b1, PRETRAINED_L1_FFN_B1, sizeof(layer_weights[1].b1));
    memcpy(layer_weights[1].W2, PRETRAINED_L1_FFN_W2, sizeof(layer_weights[1].W2));
    memcpy(layer_weights[1].b2, PRETRAINED_L1_FFN_B2, sizeof(layer_weights[1].b2));

    printf("  Pretrained weights loaded successfully!\n");
}
#endif

void init_all_weights() {
    printf("Initializing model weights...\n");

#ifdef USE_PRETRAINED_WEIGHTS
    // Use pretrained weights
    load_pretrained_weights();
    init_positional_encoding();  // Positional encoding is fixed, not trained
    printf("  Mode: PRETRAINED\n");
#else
    // Use random initialization
    set_seed(12345);  // Reproducible initialization
    init_embedding();
    init_positional_encoding();
    init_transformer_weights();
    init_output_layer();
    printf("  Mode: RANDOM INITIALIZATION\n");
#endif

    printf("  Embedding: [%d x %d]\n", VOCAB_SIZE, D_MODEL);
    printf("  Positional encoding: [%d x %d]\n", SEQ_LEN, D_MODEL);
    printf("  Transformer layers: %d\n", N_LAYERS);
    printf("  Output projection: [%d x %d]\n", VOCAB_SIZE, D_MODEL);
    printf("Done.\n\n");
}

// ============================================================
// Text Encoding/Decoding
// ============================================================

// Convert character to index (ASCII)
int char_to_idx(char c) {
    int idx = (int)(unsigned char)c;
    if (idx >= VOCAB_SIZE) idx = 0;  // Unknown -> null
    return idx;
}

// Convert index to character
char idx_to_char(int idx) {
    if (idx < 0 || idx >= VOCAB_SIZE) return '?';
    return (char)idx;
}

// Encode string to indices
void encode_string(const char *str, int *indices, size_t max_len) {
    size_t len = strlen(str);
    if (len > max_len) len = max_len;
    for (size_t i = 0; i < len; i++) {
        indices[i] = char_to_idx(str[i]);
    }
}

// ============================================================
// Model Forward Pass
// ============================================================

// Embedding lookup
void embed_tokens(float *output, const int *indices, size_t seq_len) {
    for (size_t t = 0; t < seq_len; t++) {
        int idx = indices[t];
        for (size_t i = 0; i < D_MODEL; i++) {
            output[t * D_MODEL + i] = embedding_weight[idx * D_MODEL + i];
        }
    }
}

// Add positional encoding
void add_positional_encoding(float *x, size_t seq_len) {
    for (size_t t = 0; t < seq_len; t++) {
        for (size_t i = 0; i < D_MODEL; i++) {
            x[t * D_MODEL + i] += pos_encoding[t * D_MODEL + i];
        }
    }
}

// Forward through one Transformer block
void forward_transformer_block(float *output, const float *input,
                                const TransformerBlockWeights &w,
                                size_t seq_len) {
    transformer_block_fp32(output, input,
                           w.ln1_gamma, w.ln1_beta,
                           w.Wq, w.Wk, w.Wv, w.Wo,
                           w.ln2_gamma, w.ln2_beta,
                           w.W1, w.b1, w.W2, w.b2,
                           seq_len, D_MODEL, D_K, D_FF, true);  // causal=true
}

// Project to vocabulary logits
void project_to_vocab(float *logits, const float *hidden, size_t seq_len) {
    // Only compute logits for the last token (for generation)
    // logits = hidden @ output_weight^T + output_bias
    const float *last_hidden = &hidden[(seq_len - 1) * D_MODEL];

    for (size_t v = 0; v < VOCAB_SIZE; v++) {
        float sum = output_bias[v];
        for (size_t i = 0; i < D_MODEL; i++) {
            sum += last_hidden[i] * output_weight[v * D_MODEL + i];
        }
        logits[v] = sum;
    }
}

// Softmax for logits
void softmax_logits(float *probs, const float *logits, size_t vocab_size, float temperature) {
    // Find max for numerical stability
    float max_val = logits[0];
    for (size_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    // Apply temperature and compute softmax
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_val) / temperature);
        sum += probs[i];
    }
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }
}

// Sample from probability distribution
int sample_from_probs(const float *probs, size_t vocab_size) {
    // Simple sampling using pseudo-random
    float r = (float)(g_seed % 10000) / 10000.0f;
    g_seed = g_seed * 1103515245 + 12345;

    float cumsum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (r < cumsum) {
            return (int)i;
        }
    }
    return (int)(vocab_size - 1);
}

// Argmax sampling (greedy)
int argmax(const float *probs, size_t vocab_size) {
    int max_idx = 0;
    float max_val = probs[0];
    for (size_t i = 1; i < vocab_size; i++) {
        if (probs[i] > max_val) {
            max_val = probs[i];
            max_idx = (int)i;
        }
    }
    return max_idx;
}

// ============================================================
// Text Generation
// ============================================================

void generate_text(const char *prompt, int gen_length, float temperature, bool use_sampling) {
    printf("Prompt: \"%s\"\n", prompt);
    printf("Generating %d characters (temperature=%.2f, %s)...\n\n",
           gen_length, temperature, use_sampling ? "sampling" : "greedy");

    // Encode prompt
    int indices[SEQ_LEN];
    memset(indices, 0, sizeof(indices));

    size_t prompt_len = strlen(prompt);
    if (prompt_len > SEQ_LEN) prompt_len = SEQ_LEN;

    // Right-align prompt in context window
    size_t start_pos = SEQ_LEN - prompt_len;
    encode_string(prompt, &indices[start_pos], prompt_len);

    // Allocate working buffers
    float *embedded = (float *)malloc(SEQ_LEN * D_MODEL * sizeof(float));
    float *hidden = (float *)malloc(SEQ_LEN * D_MODEL * sizeof(float));
    float *temp_buf = (float *)malloc(SEQ_LEN * D_MODEL * sizeof(float));
    float *logits = (float *)malloc(VOCAB_SIZE * sizeof(float));
    float *probs = (float *)malloc(VOCAB_SIZE * sizeof(float));

    // Output buffer
    char generated[MAX_GEN_LEN + 1];
    int gen_idx = 0;

    // Copy prompt to output
    printf("Output: ");
    for (size_t i = 0; i < prompt_len; i++) {
        printf("%c", prompt[i]);
    }
    fflush(stdout);

    // Generation loop
    for (int step = 0; step < gen_length && gen_idx < MAX_GEN_LEN; step++) {
        // 1. Embed tokens
        embed_tokens(embedded, indices, SEQ_LEN);

        // 2. Add positional encoding
        add_positional_encoding(embedded, SEQ_LEN);

        // 3. Forward through Transformer layers
        memcpy(hidden, embedded, SEQ_LEN * D_MODEL * sizeof(float));

        for (int l = 0; l < N_LAYERS; l++) {
            forward_transformer_block(temp_buf, hidden, layer_weights[l], SEQ_LEN);
            memcpy(hidden, temp_buf, SEQ_LEN * D_MODEL * sizeof(float));
        }

        // 4. Project to vocabulary
        project_to_vocab(logits, hidden, SEQ_LEN);

        // 5. Convert to probabilities
        softmax_logits(probs, logits, VOCAB_SIZE, temperature);

        // 6. Sample next token
        int next_token;
        if (use_sampling) {
            next_token = sample_from_probs(probs, VOCAB_SIZE);
        } else {
            next_token = argmax(probs, VOCAB_SIZE);
        }

        // 7. Output character
        char next_char = idx_to_char(next_token);
        printf("%c", next_char);
        fflush(stdout);

        generated[gen_idx++] = next_char;

        // 8. Shift context and append new token
        for (size_t i = 0; i < SEQ_LEN - 1; i++) {
            indices[i] = indices[i + 1];
        }
        indices[SEQ_LEN - 1] = next_token;
    }

    generated[gen_idx] = '\0';
    printf("\n\n");

    // Print the complete generated text
    printf("Generated text: \"%s\"\n\n", generated);

    // Statistics
    printf("Generation complete.\n");
    printf("  Prompt length: %zu\n", prompt_len);
    printf("  Generated length: %d\n", gen_idx);

    // Character frequency analysis
    int char_counts[VOCAB_SIZE] = {0};
    for (int i = 0; i < gen_idx; i++) {
        char_counts[(int)(unsigned char)generated[i]]++;
    }

    printf("  Top 5 characters generated:\n");
    for (int top = 0; top < 5; top++) {
        int max_idx = 0;
        int max_count = 0;
        for (int i = 32; i < 127; i++) {  // Printable ASCII
            if (char_counts[i] > max_count) {
                max_count = char_counts[i];
                max_idx = i;
            }
        }
        if (max_count > 0) {
            char c = (char)max_idx;
            printf("    '%c' (%d): %d times (%.1f%%)\n",
                   c == ' ' ? '_' : c, max_idx, max_count,
                   100.0f * max_count / gen_idx);
            char_counts[max_idx] = 0;  // Remove for next iteration
        }
    }

    // Cleanup
    free(embedded);
    free(hidden);
    free(temp_buf);
    free(logits);
    free(probs);
}

// ============================================================
// Demo: Pattern Learning Verification
// ============================================================

void demo_forward_pass() {
    printf("\n========================================\n");
    printf("Demo: Single Forward Pass\n");
    printf("========================================\n");

    const char *test_input = "Hello";
    int indices[SEQ_LEN];
    memset(indices, 0, sizeof(indices));

    size_t len = strlen(test_input);
    size_t start = SEQ_LEN - len;
    encode_string(test_input, &indices[start], len);

    printf("Input: \"%s\"\n", test_input);
    printf("Token indices (last %zu): ", len);
    for (size_t i = start; i < SEQ_LEN; i++) {
        printf("%d ", indices[i]);
    }
    printf("\n");

    // Forward pass
    float *embedded = (float *)malloc(SEQ_LEN * D_MODEL * sizeof(float));
    float *hidden = (float *)malloc(SEQ_LEN * D_MODEL * sizeof(float));
    float *temp_buf = (float *)malloc(SEQ_LEN * D_MODEL * sizeof(float));
    float *logits = (float *)malloc(VOCAB_SIZE * sizeof(float));
    float *probs = (float *)malloc(VOCAB_SIZE * sizeof(float));

    embed_tokens(embedded, indices, SEQ_LEN);
    add_positional_encoding(embedded, SEQ_LEN);

    printf("\nEmbedded (first 8 dims of last token):\n  ");
    for (int i = 0; i < 8; i++) {
        printf("%.3f ", embedded[(SEQ_LEN - 1) * D_MODEL + i]);
    }
    printf("...\n");

    memcpy(hidden, embedded, SEQ_LEN * D_MODEL * sizeof(float));
    for (int l = 0; l < N_LAYERS; l++) {
        forward_transformer_block(temp_buf, hidden, layer_weights[l], SEQ_LEN);
        memcpy(hidden, temp_buf, SEQ_LEN * D_MODEL * sizeof(float));
    }

    printf("\nAfter Transformer (first 8 dims of last token):\n  ");
    for (int i = 0; i < 8; i++) {
        printf("%.3f ", hidden[(SEQ_LEN - 1) * D_MODEL + i]);
    }
    printf("...\n");

    project_to_vocab(logits, hidden, SEQ_LEN);
    softmax_logits(probs, logits, VOCAB_SIZE, 1.0f);

    // Top 5 predictions
    printf("\nTop 5 next character predictions:\n");
    for (int top = 0; top < 5; top++) {
        int max_idx = 0;
        float max_prob = 0.0f;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            if (probs[i] > max_prob) {
                max_prob = probs[i];
                max_idx = i;
            }
        }
        char c = idx_to_char(max_idx);
        printf("  %d. '%c' (ASCII %d): %.4f\n", top + 1,
               (c >= 32 && c < 127) ? c : '?', max_idx, max_prob);
        probs[max_idx] = 0.0f;  // Remove for next iteration
    }

    free(embedded);
    free(hidden);
    free(temp_buf);
    free(logits);
    free(probs);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("================================================================\n");
    printf("  FIONA Character-level Text Generation\n");
    printf("  Using Photonic Transformer\n");
    printf("================================================================\n\n");

    // Print configuration
    printf("Model Configuration:\n");
    printf("  VOCAB_SIZE:  %d (ASCII)\n", VOCAB_SIZE);
    printf("  SEQ_LEN:     %d\n", SEQ_LEN);
    printf("  D_MODEL:     %d\n", D_MODEL);
    printf("  D_K:         %d\n", D_K);
    printf("  D_FF:        %d\n", D_FF);
    printf("  N_LAYERS:    %d\n", N_LAYERS);
    printf("\n");

    // Print photonic model
    const char *model = getenv("FIONA_PHOTONIC_MODEL");
    printf("Photonic Model: %s\n\n", model ? model : "ideal (default)");

    // Initialize weights
    init_all_weights();

    // Demo forward pass
    demo_forward_pass();

    // Text generation examples
    printf("\n========================================\n");
    printf("Text Generation Examples\n");
    printf("========================================\n");

    // Example 1: Greedy generation
    printf("\n--- Example 1: Greedy Generation ---\n");
    generate_text("The ", 50, 1.0f, false);

    // Example 2: Sampling with temperature
    printf("\n--- Example 2: Sampling (temperature=0.8) ---\n");
    set_seed(42);  // Reset for reproducibility
    generate_text("Hello ", 50, 0.8f, true);

    // Example 3: Higher temperature (more random)
    printf("\n--- Example 3: High Temperature (1.5) ---\n");
    set_seed(123);
    generate_text("AI ", 50, 1.5f, true);

    // Example 4: Lower temperature (more focused)
    printf("\n--- Example 4: Low Temperature (0.5) ---\n");
    set_seed(456);
    generate_text("Data ", 30, 0.5f, true);

    printf("\n================================================================\n");
    printf("  Generation Complete!\n");
    printf("================================================================\n");
#ifdef USE_PRETRAINED_WEIGHTS
    printf("\nNote: Using PRETRAINED weights from PyTorch training.\n");
#else
    printf("\nNote: Using RANDOM initialization (no training).\n");
    printf("For meaningful text, compile with -DUSE_PRETRAINED_WEIGHTS.\n");
#endif
    printf("This demo shows the architecture and photonic computation flow.\n\n");

    DUMP_STAT;

    return 0;
}
