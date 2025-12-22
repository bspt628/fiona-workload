// Test large static arrays for bare-metal
#include <stdio.h>
#include <string.h>

// Simulate text_gen_transformer static arrays
#define VOCAB_SIZE 128
#define D_MODEL 64
#define SEQ_LEN 32
#define D_K 32
#define D_FF 128
#define N_LAYERS 2

// Embedding matrix [VOCAB_SIZE x D_MODEL] = 128 * 64 = 8192 floats = 32KB
static float embedding_weight[VOCAB_SIZE * D_MODEL];

// Positional encoding [SEQ_LEN x D_MODEL] = 32 * 64 = 2048 floats = 8KB
static float pos_encoding[SEQ_LEN * D_MODEL];

// Per-layer weights
struct LayerWeights {
    float ln1_gamma[D_MODEL];   // 64
    float ln1_beta[D_MODEL];    // 64
    float Wq[D_K * D_MODEL];    // 2048
    float Wk[D_K * D_MODEL];    // 2048
    float Wv[D_K * D_MODEL];    // 2048
    float Wo[D_MODEL * D_K];    // 2048
    float ln2_gamma[D_MODEL];   // 64
    float ln2_beta[D_MODEL];    // 64
    float W1[D_FF * D_MODEL];   // 8192
    float b1[D_FF];             // 128
    float W2[D_MODEL * D_FF];   // 8192
    float b2[D_MODEL];          // 64
    // Total per layer: ~27KB
};

static LayerWeights layer_weights[N_LAYERS];  // ~54KB total

// Output layer [VOCAB_SIZE x D_MODEL] = 32KB
static float output_weight[VOCAB_SIZE * D_MODEL];
static float output_bias[VOCAB_SIZE];

int main() {
    printf("Large Static Array Test\n");

    // Initialize some arrays to trigger bss initialization
    printf("Initializing embedding...\n");
    for (int i = 0; i < 100; i++) {
        embedding_weight[i] = (float)i * 0.01f;
    }
    printf("  embedding[0] = %d\n", (int)(embedding_weight[0] * 1000));
    printf("  embedding[99] = %d\n", (int)(embedding_weight[99] * 1000));

    printf("Initializing layer weights...\n");
    for (int l = 0; l < N_LAYERS; l++) {
        layer_weights[l].ln1_gamma[0] = 1.0f;
        layer_weights[l].ln1_beta[0] = 0.0f;
        printf("  Layer %d: ln1_gamma[0] = %d\n", l, (int)(layer_weights[l].ln1_gamma[0] * 1000));
    }

    printf("Initializing output layer...\n");
    for (int i = 0; i < VOCAB_SIZE; i++) {
        output_bias[i] = 0.1f;
    }
    printf("  output_bias[0] = %d\n", (int)(output_bias[0] * 1000));

    // Compute total static memory
    size_t total = sizeof(embedding_weight) + sizeof(pos_encoding) +
                   sizeof(layer_weights) + sizeof(output_weight) + sizeof(output_bias);
    printf("Total static memory: %lu bytes (%lu KB)\n", (unsigned long)total, (unsigned long)(total/1024));

    printf("Large Static Array Test Complete!\n");
    return 0;
}
