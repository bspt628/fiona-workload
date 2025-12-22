/**
 * @file main.cc
 * @brief Photonic INT16 Transformer for FIONA RTL (Verilator)
 *
 * This workload uses FIONA-V photonic instructions (CONFIG_PALU, MVM)
 * for matrix-vector multiplication in Transformer attention and FFN.
 * The photonic model is invoked via DPI-C to fiona-photonic Python models.
 *
 * Key design decisions:
 * - All dimensions are multiples of 32 for VLSU 64-byte alignment
 * - Uses CONFIG_PALU + MVM instructions instead of tiled DOTP
 * - Static aligned arrays for bare-metal execution
 *
 * Photonic instructions used:
 * - SET_VLEN (funct7=10): Set vector length
 * - LOAD_V (funct7=8): Vector load (64-byte aligned)
 * - STORE_V (funct7=9): Vector store (64-byte aligned)
 * - CONFIG_PALU (funct7=12): Load weight matrix row
 * - MVM (funct7=14): Matrix-vector multiply
 *
 * Photonic Model Selection:
 * Set FIONA_PHOTONIC_MODEL environment variable before running:
 *   - ideal (default): Perfect mathematical operations
 *   - mzi_realistic: Realistic MZI model with phase error, loss, crosstalk
 *   - all_effects: Combined all realistic effects
 *   - noisy: Simple Gaussian noise
 *   - quantized: DAC/ADC quantization effects
 *
 * Additional environment variables for realistic models:
 *   - FIONA_PHASE_ERROR_SIGMA: Phase error std dev (default: 0.02)
 *   - FIONA_THERMAL_CROSSTALK_SIGMA: Thermal crosstalk (default: 0.01)
 *   - FIONA_INSERTION_LOSS_DB: Insertion loss per stage (default: 0.3)
 *   - FIONA_QUANT_BITS: DAC/ADC bits (default: 8)
 *   - FIONA_VERBOSE: Set to 1 for debug output
 *
 * @author FIONA Project
 * @date 2025-12-22
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fiona.h"

// ============================================================
// Configuration
// ============================================================

// Model dimensions - ALL MUST BE MULTIPLES OF 32 for VLSU alignment
#define SEQ_LEN     4       // Number of tokens
#define D_MODEL     32      // Model dimension (must be multiple of 32)
#define D_K         32      // Key/Query dimension (must be multiple of 32)
#define D_FF        64      // FFN intermediate dimension

// Quantization
#define SCALE       128.0f  // Quantization scale factor
#define QUANT_MAX   32767
#define QUANT_MIN   (-32768)

// Photonic unit constraints
#define PU_ROWS     8       // Max rows per MVM (PU_MRR_RING)
#define VREG_SIZE   32      // Vector register size (EU_VEC_ELEM)

// ============================================================
// 64-byte aligned static arrays
// ============================================================

// Weight matrices - each row is 32 elements (64 bytes)
static int16_t Wq[D_MODEL * D_K] __attribute__((aligned(64)));
static int16_t Wk[D_MODEL * D_K] __attribute__((aligned(64)));
static int16_t Wv[D_MODEL * D_K] __attribute__((aligned(64)));
static int16_t Wo[D_K * D_MODEL] __attribute__((aligned(64)));
static int16_t W1[D_MODEL * D_FF] __attribute__((aligned(64)));
static int16_t W2[D_FF * D_MODEL] __attribute__((aligned(64)));

// Input/Output buffers
static int16_t input[SEQ_LEN * D_MODEL] __attribute__((aligned(64)));
static int16_t output[SEQ_LEN * D_MODEL] __attribute__((aligned(64)));

// Intermediate buffers
static int16_t Q[SEQ_LEN * D_K] __attribute__((aligned(64)));
static int16_t K[SEQ_LEN * D_K] __attribute__((aligned(64)));
static int16_t V[SEQ_LEN * D_K] __attribute__((aligned(64)));
static int16_t attn_scores[SEQ_LEN * SEQ_LEN] __attribute__((aligned(64)));
static int16_t attn_weights[SEQ_LEN * SEQ_LEN] __attribute__((aligned(64)));
static int16_t attn_out[SEQ_LEN * D_K] __attribute__((aligned(64)));
static int16_t proj_out[SEQ_LEN * D_MODEL] __attribute__((aligned(64)));
static int16_t residual1[SEQ_LEN * D_MODEL] __attribute__((aligned(64)));
static int16_t ffn_hidden[SEQ_LEN * D_FF] __attribute__((aligned(64)));
static int16_t ffn_out[SEQ_LEN * D_MODEL] __attribute__((aligned(64)));

// Temporary buffers for photonic MVM (must be 64-byte aligned)
static int16_t pho_vec_in[VREG_SIZE] __attribute__((aligned(64)));
static int16_t pho_vec_out[VREG_SIZE] __attribute__((aligned(64)));
static int16_t pho_mat_tile[PU_ROWS * VREG_SIZE] __attribute__((aligned(64)));

// ============================================================
// Quantization helpers
// ============================================================

static inline int16_t quantize(float val) {
    int32_t q = (int32_t)roundf(val * SCALE);
    if (q > QUANT_MAX) q = QUANT_MAX;
    if (q < QUANT_MIN) q = QUANT_MIN;
    return (int16_t)q;
}

static inline float dequantize(int16_t val) {
    return (float)val / SCALE;
}

// ============================================================
// Photonic MVM using CONFIG_PALU + MVM instructions
// ============================================================

/**
 * @brief Photonic MVM tile (up to 8 rows x 32 cols)
 *
 * Uses CONFIG_PALU to load weight rows, then MVM instruction.
 * All addresses must be 64-byte aligned.
 *
 * @param out Output vector [out_size]
 * @param weight Weight matrix [out_size x in_size], row-major
 * @param input Input vector [in_size]
 * @param out_size Number of rows (max 8)
 * @param in_size Number of columns (max 32)
 */
void photonic_mvm_tile(int16_t *out, const int16_t *weight, const int16_t *input,
                       size_t out_size, size_t in_size) {
    // Pad input to pho_vec_in (VREG_SIZE elements)
    memset(pho_vec_in, 0, sizeof(pho_vec_in));
    for (size_t i = 0; i < in_size && i < VREG_SIZE; i++) {
        pho_vec_in[i] = input[i];
    }

    // Pad weight matrix to pho_mat_tile (PU_ROWS x VREG_SIZE)
    // Each row must be at 64-byte aligned offset
    memset(pho_mat_tile, 0, sizeof(pho_mat_tile));
    for (size_t i = 0; i < out_size && i < PU_ROWS; i++) {
        for (size_t j = 0; j < in_size && j < VREG_SIZE; j++) {
            pho_mat_tile[i * VREG_SIZE + j] = weight[i * in_size + j];
        }
    }

    // Set vector length
    SET_VLEN(in_size);

    // Load weight matrix rows using CONFIG_PALU
    for (size_t i = 0; i < out_size && i < PU_ROWS; i++) {
        CONFIG_PALU(&pho_mat_tile[i * VREG_SIZE], i);
    }

    // Load input vector
    LOAD_V(1, pho_vec_in);

    // Execute MVM
    SET_VLEN(out_size);
    MVM(0, 1);

    // Store result
    STORE_V(0, pho_vec_out);

    // Copy output
    for (size_t i = 0; i < out_size; i++) {
        out[i] = pho_vec_out[i];
    }
}

/**
 * @brief Tiled photonic MVM for arbitrary sizes
 *
 * Tiles the computation into 8x32 blocks using photonic_mvm_tile.
 */
void photonic_mvm(int16_t *out, const int16_t *weight, const int16_t *input,
                  size_t out_size, size_t in_size) {
    // Initialize output to zero
    for (size_t i = 0; i < out_size; i++) {
        out[i] = 0;
    }

    // Tile over output dimension (8 rows at a time)
    for (size_t out_tile = 0; out_tile < out_size; out_tile += PU_ROWS) {
        size_t tile_out_size = (out_tile + PU_ROWS <= out_size) ?
                               PU_ROWS : (out_size - out_tile);

        // Tile over input dimension (32 cols at a time, accumulate)
        for (size_t in_tile = 0; in_tile < in_size; in_tile += VREG_SIZE) {
            size_t tile_in_size = (in_tile + VREG_SIZE <= in_size) ?
                                  VREG_SIZE : (in_size - in_tile);

            // Prepare tile weight matrix
            static int16_t tile_weight[PU_ROWS * VREG_SIZE] __attribute__((aligned(64)));
            memset(tile_weight, 0, sizeof(tile_weight));

            for (size_t i = 0; i < tile_out_size; i++) {
                for (size_t j = 0; j < tile_in_size; j++) {
                    tile_weight[i * tile_in_size + j] =
                        weight[(out_tile + i) * in_size + (in_tile + j)];
                }
            }

            // Prepare tile input
            static int16_t tile_input[VREG_SIZE] __attribute__((aligned(64)));
            memset(tile_input, 0, sizeof(tile_input));
            for (size_t j = 0; j < tile_in_size; j++) {
                tile_input[j] = input[in_tile + j];
            }

            // Execute photonic MVM on tile
            static int16_t tile_out[PU_ROWS] __attribute__((aligned(64)));
            photonic_mvm_tile(tile_out, tile_weight, tile_input,
                              tile_out_size, tile_in_size);

            // Accumulate results
            for (size_t i = 0; i < tile_out_size; i++) {
                int32_t sum = (int32_t)out[out_tile + i] + (int32_t)tile_out[i];
                if (sum > QUANT_MAX) sum = QUANT_MAX;
                if (sum < QUANT_MIN) sum = QUANT_MIN;
                out[out_tile + i] = (int16_t)sum;
            }
        }
    }
}

// ============================================================
// Activation functions
// ============================================================

void relu_int16(int16_t *out, const int16_t *in, size_t len) {
    for (size_t i = 0; i < len; i++) {
        out[i] = (in[i] > 0) ? in[i] : 0;
    }
}

void softmax_int16(int16_t *out, const int16_t *in, size_t len) {
    // Convert to float, compute softmax, convert back
    float max_val = dequantize(in[0]);
    for (size_t i = 1; i < len; i++) {
        float v = dequantize(in[i]);
        if (v > max_val) max_val = v;
    }

    float sum = 0.0f;
    static float temp[SEQ_LEN];
    for (size_t i = 0; i < len; i++) {
        temp[i] = expf(dequantize(in[i]) - max_val);
        sum += temp[i];
    }

    for (size_t i = 0; i < len; i++) {
        out[i] = quantize(temp[i] / sum);
    }
}

// ============================================================
// Attention mechanism
// ============================================================

void self_attention(int16_t *attn_output,
                    const int16_t *X,
                    const int16_t *Wq_, const int16_t *Wk_,
                    const int16_t *Wv_, const int16_t *Wo_,
                    size_t seq_len, size_t d_model, size_t d_k) {
    printf("  Computing Q, K, V projections (photonic MVM)...\n");

    // Q = X @ Wq, K = X @ Wk, V = X @ Wv
    for (size_t t = 0; t < seq_len; t++) {
        photonic_mvm(&Q[t * d_k], Wq_, &X[t * d_model], d_k, d_model);
        photonic_mvm(&K[t * d_k], Wk_, &X[t * d_model], d_k, d_model);
        photonic_mvm(&V[t * d_k], Wv_, &X[t * d_model], d_k, d_model);
    }

    printf("  Computing attention scores (CPU)...\n");

    // Attention scores = Q @ K^T / sqrt(d_k)
    float scale_factor = 1.0f / sqrtf((float)d_k);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            int32_t sum = 0;
            for (size_t k = 0; k < d_k; k++) {
                sum += (int32_t)Q[i * d_k + k] * (int32_t)K[j * d_k + k];
            }
            float score = (float)sum / (SCALE * SCALE) * scale_factor;
            attn_scores[i * seq_len + j] = quantize(score);
        }
    }

    // Apply causal mask
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = i + 1; j < seq_len; j++) {
            attn_scores[i * seq_len + j] = QUANT_MIN;
        }
    }

    printf("  Computing softmax (CPU)...\n");

    // Softmax
    for (size_t i = 0; i < seq_len; i++) {
        softmax_int16(&attn_weights[i * seq_len], &attn_scores[i * seq_len], seq_len);
    }

    printf("  Computing attention output (CPU)...\n");

    // Attention output = attn_weights @ V
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t d = 0; d < d_k; d++) {
            int32_t sum = 0;
            for (size_t j = 0; j < seq_len; j++) {
                sum += (int32_t)attn_weights[i * seq_len + j] * (int32_t)V[j * d_k + d];
            }
            float val = (float)sum / (SCALE * SCALE);
            attn_out[i * d_k + d] = quantize(val);
        }
    }

    printf("  Computing output projection (photonic MVM)...\n");

    // Output projection
    for (size_t t = 0; t < seq_len; t++) {
        photonic_mvm(&attn_output[t * d_model], Wo_, &attn_out[t * d_k], d_model, d_k);
    }
}

// ============================================================
// Feed-forward network
// ============================================================

void ffn_relu(int16_t *ffn_output, const int16_t *X,
              const int16_t *W1_, const int16_t *W2_,
              size_t seq_len, size_t d_model, size_t d_ff) {
    printf("  FFN layer 1 (photonic MVM)...\n");

    // Hidden = ReLU(X @ W1)
    for (size_t t = 0; t < seq_len; t++) {
        photonic_mvm(&ffn_hidden[t * d_ff], W1_, &X[t * d_model], d_ff, d_model);
    }

    printf("  FFN ReLU activation...\n");
    relu_int16(ffn_hidden, ffn_hidden, seq_len * d_ff);

    printf("  FFN layer 2 (photonic MVM)...\n");

    // Output = Hidden @ W2
    for (size_t t = 0; t < seq_len; t++) {
        photonic_mvm(&ffn_output[t * d_model], W2_, &ffn_hidden[t * d_ff], d_model, d_ff);
    }
}

// ============================================================
// Transformer block
// ============================================================

void transformer_block(int16_t *block_output, const int16_t *block_input) {
    printf("\n[Transformer Block]\n");

    // 1. Self-Attention
    printf("1. Self-Attention\n");
    self_attention(proj_out, block_input, Wq, Wk, Wv, Wo,
                   SEQ_LEN, D_MODEL, D_K);

    // Residual connection
    printf("  Residual connection 1...\n");
    for (size_t i = 0; i < SEQ_LEN * D_MODEL; i++) {
        int32_t sum = (int32_t)block_input[i] + (int32_t)proj_out[i];
        if (sum > QUANT_MAX) sum = QUANT_MAX;
        if (sum < QUANT_MIN) sum = QUANT_MIN;
        residual1[i] = (int16_t)sum;
    }

    // 2. FFN
    printf("2. Feed-Forward Network\n");
    ffn_relu(ffn_out, residual1, W1, W2, SEQ_LEN, D_MODEL, D_FF);

    // Residual connection
    printf("  Residual connection 2...\n");
    for (size_t i = 0; i < SEQ_LEN * D_MODEL; i++) {
        int32_t sum = (int32_t)residual1[i] + (int32_t)ffn_out[i];
        if (sum > QUANT_MAX) sum = QUANT_MAX;
        if (sum < QUANT_MIN) sum = QUANT_MIN;
        block_output[i] = (int16_t)sum;
    }
}

// ============================================================
// Initialization
// ============================================================

static unsigned int rand_seed = 12345;

int16_t pseudo_rand_int16() {
    rand_seed = rand_seed * 1103515245 + 12345;
    return (int16_t)((rand_seed >> 8) % 256 - 128);
}

void init_weights() {
    printf("Initializing weights...\n");

    // Initialize all weights with small random values
    for (size_t i = 0; i < D_MODEL * D_K; i++) {
        Wq[i] = pseudo_rand_int16() / 4;  // Scale down to avoid overflow
        Wk[i] = pseudo_rand_int16() / 4;
        Wv[i] = pseudo_rand_int16() / 4;
    }
    for (size_t i = 0; i < D_K * D_MODEL; i++) {
        Wo[i] = pseudo_rand_int16() / 4;
    }
    for (size_t i = 0; i < D_MODEL * D_FF; i++) {
        W1[i] = pseudo_rand_int16() / 4;
    }
    for (size_t i = 0; i < D_FF * D_MODEL; i++) {
        W2[i] = pseudo_rand_int16() / 4;
    }

    // Initialize input
    for (size_t i = 0; i < SEQ_LEN * D_MODEL; i++) {
        input[i] = pseudo_rand_int16() / 2;
    }
}

void print_vector(const char *name, const int16_t *v, size_t len) {
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
// Main
// ============================================================

int main() {
    printf("========================================\n");
    printf("FIONA Photonic INT16 Transformer\n");
    printf("========================================\n");
    printf("Using CONFIG_PALU + MVM photonic instructions\n");
    printf("Photonic computation via DPI-C to Python\n");
    printf("All dimensions: multiples of 32 (64-byte aligned)\n");
    printf("========================================\n\n");

    printf("Photonic Model:\n");
    printf("  Set FIONA_PHOTONIC_MODEL env var to select:\n");
    printf("  - ideal (default): Perfect math ops\n");
    printf("  - mzi_realistic: MZI with phase error, loss\n");
    printf("  - all_effects: All realistic effects\n");
    printf("\n");

    printf("Transformer Configuration:\n");
    printf("  SEQ_LEN  = %d\n", SEQ_LEN);
    printf("  D_MODEL  = %d\n", D_MODEL);
    printf("  D_K      = %d\n", D_K);
    printf("  D_FF     = %d\n", D_FF);
    printf("  SCALE    = %.1f\n", SCALE);
    printf("\n");

    printf("Photonic Tile Size:\n");
    printf("  Max rows/tile = %d (PU_MRR_RING)\n", PU_ROWS);
    printf("  Max cols/tile = %d (EU_VEC_ELEM)\n", VREG_SIZE);
    printf("\n");

    // Initialize
    init_weights();

    printf("\nInput:\n");
    print_vector("  Token 0", &input[0], D_MODEL);
    print_vector("  Token 1", &input[D_MODEL], D_MODEL);

    // Run transformer block
    transformer_block(output, input);

    printf("\nOutput:\n");
    print_vector("  Token 0", &output[0], D_MODEL);
    print_vector("  Token 1", &output[D_MODEL], D_MODEL);

    // Verify output is not all zeros
    bool has_nonzero = false;
    for (size_t i = 0; i < SEQ_LEN * D_MODEL; i++) {
        if (output[i] != 0) {
            has_nonzero = true;
            break;
        }
    }

    printf("\n========================================\n");
    if (has_nonzero) {
        printf("Result: PASS (output is non-zero)\n");
    } else {
        printf("Result: FAIL (output is all zeros)\n");
    }
    printf("========================================\n");

    return has_nonzero ? 0 : 1;
}
