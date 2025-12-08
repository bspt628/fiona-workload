/**
 * @file train_mlp_host.cc
 * @brief Host-side MLP Training for Iris Dataset
 *
 * Trains an MLP on the host machine and outputs:
 * 1. Binary weight file (mlp_iris_weights.bin)
 * 2. C header file with quantized weights for embedding in RISC-V binary
 *
 * Compile: g++ -o train_mlp_host train_mlp_host.cc -O2 -lm
 * Run: ./train_mlp_host [quant_bits]
 *
 * @author FIONA Project
 * @date 2025-12-05
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Dataset Configuration
#define NUM_TRAIN    120
#define NUM_TEST     30
#define NUM_FEATURES 4
#define NUM_CLASSES  3

// Hyperparameters
#define HIDDEN_SIZE  32
#define NUM_EPOCHS   500
#define LEARNING_RATE 0.1f

// Default quantization
#define DEFAULT_QUANT_BITS 5

// Iris Dataset
static const float train_X[NUM_TRAIN][NUM_FEATURES] = {
    // Setosa (40 samples)
    {0.222, 0.625, 0.068, 0.042}, {0.167, 0.417, 0.068, 0.042}, {0.111, 0.500, 0.051, 0.042},
    {0.083, 0.458, 0.085, 0.042}, {0.194, 0.667, 0.068, 0.042}, {0.306, 0.792, 0.119, 0.125},
    {0.083, 0.583, 0.068, 0.083}, {0.194, 0.583, 0.085, 0.042}, {0.028, 0.375, 0.068, 0.042},
    {0.167, 0.500, 0.085, 0.042}, {0.306, 0.708, 0.085, 0.042}, {0.139, 0.583, 0.102, 0.042},
    {0.139, 0.417, 0.068, 0.042}, {0.000, 0.417, 0.017, 0.000}, {0.417, 0.833, 0.034, 0.042},
    {0.389, 1.000, 0.085, 0.125}, {0.306, 0.792, 0.051, 0.125}, {0.222, 0.625, 0.085, 0.083},
    {0.389, 0.750, 0.119, 0.083}, {0.222, 0.750, 0.085, 0.042}, {0.306, 0.583, 0.119, 0.042},
    {0.222, 0.708, 0.085, 0.125}, {0.083, 0.667, 0.000, 0.042}, {0.222, 0.542, 0.119, 0.167},
    {0.139, 0.583, 0.153, 0.042}, {0.194, 0.417, 0.102, 0.042}, {0.194, 0.583, 0.102, 0.125},
    {0.250, 0.625, 0.085, 0.042}, {0.250, 0.583, 0.068, 0.042}, {0.111, 0.500, 0.102, 0.042},
    {0.139, 0.458, 0.102, 0.042}, {0.306, 0.583, 0.085, 0.125}, {0.250, 0.875, 0.085, 0.042},
    {0.333, 0.917, 0.068, 0.042}, {0.167, 0.458, 0.085, 0.000}, {0.194, 0.500, 0.034, 0.042},
    {0.333, 0.625, 0.051, 0.042}, {0.167, 0.667, 0.068, 0.000}, {0.028, 0.417, 0.051, 0.042},
    {0.222, 0.583, 0.085, 0.042},
    // Versicolor (40 samples)
    {0.500, 0.333, 0.627, 0.458}, {0.333, 0.250, 0.576, 0.458}, {0.556, 0.542, 0.627, 0.583},
    {0.194, 0.125, 0.390, 0.375}, {0.389, 0.167, 0.525, 0.500}, {0.278, 0.375, 0.424, 0.375},
    {0.472, 0.417, 0.644, 0.417}, {0.056, 0.125, 0.254, 0.167}, {0.444, 0.292, 0.627, 0.458},
    {0.167, 0.167, 0.390, 0.292}, {0.028, 0.000, 0.254, 0.250}, {0.306, 0.417, 0.593, 0.500},
    {0.194, 0.083, 0.390, 0.292}, {0.500, 0.333, 0.644, 0.375}, {0.222, 0.333, 0.322, 0.417},
    {0.472, 0.417, 0.593, 0.458}, {0.361, 0.417, 0.542, 0.500}, {0.278, 0.292, 0.492, 0.417},
    {0.472, 0.083, 0.508, 0.375}, {0.194, 0.208, 0.424, 0.375}, {0.556, 0.292, 0.627, 0.583},
    {0.361, 0.333, 0.474, 0.417}, {0.556, 0.208, 0.661, 0.583}, {0.500, 0.333, 0.508, 0.458},
    {0.389, 0.333, 0.559, 0.500}, {0.333, 0.417, 0.559, 0.417}, {0.417, 0.292, 0.610, 0.500},
    {0.472, 0.375, 0.610, 0.500}, {0.306, 0.333, 0.508, 0.458}, {0.194, 0.208, 0.390, 0.375},
    {0.194, 0.167, 0.390, 0.417}, {0.389, 0.417, 0.542, 0.458}, {0.194, 0.208, 0.475, 0.417},
    {0.417, 0.292, 0.593, 0.458}, {0.222, 0.417, 0.441, 0.333}, {0.472, 0.375, 0.593, 0.500},
    {0.528, 0.458, 0.627, 0.458}, {0.333, 0.333, 0.508, 0.375}, {0.194, 0.333, 0.441, 0.417},
    {0.361, 0.292, 0.542, 0.500},
    // Virginica (40 samples)
    {0.722, 0.458, 0.864, 0.917}, {0.500, 0.333, 0.627, 0.708}, {0.750, 0.417, 0.831, 0.833},
    {0.528, 0.333, 0.695, 0.708}, {0.611, 0.417, 0.763, 0.792}, {0.778, 0.458, 0.898, 0.833},
    {0.222, 0.208, 0.576, 0.583}, {0.750, 0.333, 0.864, 0.750}, {0.611, 0.208, 0.729, 0.708},
    {0.806, 0.667, 0.864, 1.000}, {0.611, 0.500, 0.695, 0.792}, {0.528, 0.250, 0.644, 0.708},
    {0.611, 0.333, 0.763, 0.708}, {0.444, 0.208, 0.695, 0.625}, {0.556, 0.375, 0.780, 0.792},
    {0.611, 0.417, 0.813, 0.875}, {0.528, 0.458, 0.695, 0.625}, {0.944, 0.750, 0.966, 0.875},
    {0.917, 0.250, 0.932, 0.750}, {0.306, 0.042, 0.610, 0.583}, {0.722, 0.417, 0.864, 0.917},
    {0.444, 0.292, 0.695, 0.750}, {0.833, 0.208, 0.898, 0.708}, {0.500, 0.333, 0.661, 0.625},
    {0.611, 0.417, 0.729, 0.792}, {0.667, 0.292, 0.780, 0.750}, {0.472, 0.292, 0.695, 0.625},
    {0.556, 0.458, 0.729, 0.708}, {0.583, 0.333, 0.780, 0.833}, {0.750, 0.458, 0.780, 0.917},
    {0.750, 0.292, 0.847, 0.708}, {0.944, 0.667, 0.932, 0.958}, {0.556, 0.208, 0.780, 0.958},
    {0.472, 0.292, 0.695, 0.583}, {0.500, 0.333, 0.678, 0.708}, {0.750, 0.417, 0.847, 0.833},
    {0.611, 0.458, 0.763, 0.708}, {0.528, 0.333, 0.695, 0.708}, {0.389, 0.333, 0.610, 0.750},
    {0.611, 0.417, 0.763, 0.792}
};

static const int train_Y[NUM_TRAIN] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
};

static const float test_X[NUM_TEST][NUM_FEATURES] = {
    {0.167, 0.542, 0.068, 0.042}, {0.250, 0.583, 0.085, 0.042}, {0.333, 0.792, 0.051, 0.042},
    {0.222, 0.708, 0.068, 0.083}, {0.278, 0.667, 0.102, 0.083}, {0.167, 0.625, 0.085, 0.042},
    {0.306, 0.667, 0.085, 0.042}, {0.194, 0.625, 0.068, 0.042}, {0.222, 0.667, 0.085, 0.083},
    {0.139, 0.500, 0.051, 0.042},
    {0.361, 0.208, 0.508, 0.417}, {0.472, 0.333, 0.576, 0.500}, {0.389, 0.333, 0.508, 0.417},
    {0.306, 0.333, 0.492, 0.458}, {0.472, 0.208, 0.593, 0.500}, {0.222, 0.208, 0.458, 0.458},
    {0.361, 0.333, 0.559, 0.458}, {0.389, 0.375, 0.559, 0.500}, {0.306, 0.292, 0.542, 0.458},
    {0.528, 0.333, 0.593, 0.458},
    {0.611, 0.417, 0.780, 0.875}, {0.694, 0.333, 0.814, 0.708}, {0.528, 0.375, 0.729, 0.792},
    {0.639, 0.542, 0.797, 0.875}, {0.778, 0.458, 0.864, 0.833}, {0.667, 0.292, 0.797, 0.750},
    {0.583, 0.333, 0.780, 0.792}, {0.500, 0.333, 0.763, 0.792}, {0.722, 0.542, 0.814, 0.917},
    {0.722, 0.417, 0.797, 0.833}
};

static const int test_Y[NUM_TEST] = {
    0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2
};

// Network Weights
static float W1[NUM_FEATURES][HIDDEN_SIZE];
static float b1[HIDDEN_SIZE];
static float W2[HIDDEN_SIZE][NUM_CLASSES];
static float b2[NUM_CLASSES];

// Gradients
static float dW1[NUM_FEATURES][HIDDEN_SIZE];
static float db1[HIDDEN_SIZE];
static float dW2[HIDDEN_SIZE][NUM_CLASSES];
static float db2[NUM_CLASSES];

// Intermediate values
static float h1[NUM_TRAIN][HIDDEN_SIZE];
static float a1[NUM_TRAIN][HIDDEN_SIZE];
static float out[NUM_TRAIN][NUM_CLASSES];

// Random initialization
static uint32_t rand_seed = 42;

static float random_uniform() {
    rand_seed = rand_seed * 1103515245 + 12345;
    return (float)(rand_seed & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

static void init_weights() {
    float scale1 = sqrtf(2.0f / NUM_FEATURES);
    for (int i = 0; i < NUM_FEATURES; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            W1[i][j] = (random_uniform() - 0.5f) * scale1;
        }
    }
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        b1[j] = 0.0f;
    }

    float scale2 = sqrtf(2.0f / HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            W2[i][j] = (random_uniform() - 0.5f) * scale2;
        }
    }
    for (int j = 0; j < NUM_CLASSES; j++) {
        b2[j] = 0.0f;
    }
}

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static void forward(const float X[][NUM_FEATURES], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            float sum = b1[j];
            for (int k = 0; k < NUM_FEATURES; k++) {
                sum += X[i][k] * W1[k][j];
            }
            h1[i][j] = sum;
            a1[i][j] = relu(sum);
        }
    }

    for (int i = 0; i < n; i++) {
        float max_val = -1e9f;
        for (int j = 0; j < NUM_CLASSES; j++) {
            float sum = b2[j];
            for (int k = 0; k < HIDDEN_SIZE; k++) {
                sum += a1[i][k] * W2[k][j];
            }
            out[i][j] = sum;
            if (sum > max_val) max_val = sum;
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < NUM_CLASSES; j++) {
            out[i][j] = expf(out[i][j] - max_val);
            sum_exp += out[i][j];
        }
        for (int j = 0; j < NUM_CLASSES; j++) {
            out[i][j] /= sum_exp;
        }
    }
}

static float backward(const float X[][NUM_FEATURES], const int Y[], int n) {
    float total_loss = 0.0f;

    memset(dW1, 0, sizeof(dW1));
    memset(db1, 0, sizeof(db1));
    memset(dW2, 0, sizeof(dW2));
    memset(db2, 0, sizeof(db2));

    for (int i = 0; i < n; i++) {
        float prob = out[i][Y[i]];
        if (prob < 1e-7f) prob = 1e-7f;
        total_loss -= logf(prob);

        float d_out[NUM_CLASSES];
        for (int j = 0; j < NUM_CLASSES; j++) {
            d_out[j] = out[i][j] - (j == Y[i] ? 1.0f : 0.0f);
        }

        for (int j = 0; j < NUM_CLASSES; j++) {
            for (int k = 0; k < HIDDEN_SIZE; k++) {
                dW2[k][j] += a1[i][k] * d_out[j];
            }
            db2[j] += d_out[j];
        }

        float d_a1[HIDDEN_SIZE];
        for (int k = 0; k < HIDDEN_SIZE; k++) {
            float sum = 0.0f;
            for (int j = 0; j < NUM_CLASSES; j++) {
                sum += d_out[j] * W2[k][j];
            }
            d_a1[k] = sum;
        }

        float d_h1[HIDDEN_SIZE];
        for (int k = 0; k < HIDDEN_SIZE; k++) {
            d_h1[k] = h1[i][k] > 0.0f ? d_a1[k] : 0.0f;
        }

        for (int j = 0; j < HIDDEN_SIZE; j++) {
            for (int k = 0; k < NUM_FEATURES; k++) {
                dW1[k][j] += X[i][k] * d_h1[j];
            }
            db1[j] += d_h1[j];
        }
    }

    float scale = 1.0f / n;
    for (int i = 0; i < NUM_FEATURES; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            dW1[i][j] *= scale;
    for (int j = 0; j < HIDDEN_SIZE; j++)
        db1[j] *= scale;
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < NUM_CLASSES; j++)
            dW2[i][j] *= scale;
    for (int j = 0; j < NUM_CLASSES; j++)
        db2[j] *= scale;

    return total_loss / n;
}

static void update_weights() {
    for (int i = 0; i < NUM_FEATURES; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            W1[i][j] -= LEARNING_RATE * dW1[i][j];

    for (int j = 0; j < HIDDEN_SIZE; j++)
        b1[j] -= LEARNING_RATE * db1[j];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < NUM_CLASSES; j++)
            W2[i][j] -= LEARNING_RATE * dW2[i][j];

    for (int j = 0; j < NUM_CLASSES; j++)
        b2[j] -= LEARNING_RATE * db2[j];
}

static float evaluate(const float X[][NUM_FEATURES], const int Y[], int n) {
    int correct = 0;

    for (int i = 0; i < n; i++) {
        float hidden[HIDDEN_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            float sum = b1[j];
            for (int k = 0; k < NUM_FEATURES; k++) {
                sum += X[i][k] * W1[k][j];
            }
            hidden[j] = relu(sum);
        }

        int pred = 0;
        float max_val = -1e9f;
        for (int j = 0; j < NUM_CLASSES; j++) {
            float sum = b2[j];
            for (int k = 0; k < HIDDEN_SIZE; k++) {
                sum += hidden[k] * W2[k][j];
            }
            if (sum > max_val) {
                max_val = sum;
                pred = j;
            }
        }

        if (pred == Y[i]) correct++;
    }

    return (float)correct / n;
}

// Find max absolute value
static float find_max_abs(const float *arr, int size) {
    float max_val = 0.0f;
    for (int i = 0; i < size; i++) {
        float abs_val = fabsf(arr[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    return max_val;
}

// Quantize to int16
static int16_t quantize(float val, float scale) {
    float q = roundf(val * scale);
    if (q > 32767.0f) q = 32767.0f;
    if (q < -32768.0f) q = -32768.0f;
    return (int16_t)q;
}

// Save C header with quantized weights
static void save_weights_header(const char *filename, int quant_bits) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("[ERROR] Cannot open %s for writing\n", filename);
        return;
    }

    float quant_scale = (float)(1 << (quant_bits - 1));

    // Find scales
    float w1_max = find_max_abs(&W1[0][0], NUM_FEATURES * HIDDEN_SIZE);
    float w2_max = find_max_abs(&W2[0][0], HIDDEN_SIZE * NUM_CLASSES);
    float w1_scale = quant_scale / w1_max;
    float w2_scale = quant_scale / w2_max;

    fprintf(fp, "/**\n");
    fprintf(fp, " * @file %s\n", filename);
    fprintf(fp, " * @brief Auto-generated MLP weights for Iris dataset\n");
    fprintf(fp, " * \n");
    fprintf(fp, " * Architecture: %d -> %d -> %d\n", NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES);
    fprintf(fp, " * Quantization: %d-bit\n", quant_bits);
    fprintf(fp, " * W1 scale: %.4f, W2 scale: %.4f\n", w1_scale, w2_scale);
    fprintf(fp, " * \n");
    fprintf(fp, " * Generated by train_mlp_host.cc\n");
    fprintf(fp, " */\n\n");

    fprintf(fp, "#ifndef MLP_IRIS_WEIGHTS_H\n");
    fprintf(fp, "#define MLP_IRIS_WEIGHTS_H\n\n");

    fprintf(fp, "#include \"base/config.h\"\n\n");

    fprintf(fp, "// Quantization info\n");
    fprintf(fp, "#define MLP_QUANT_BITS %d\n", quant_bits);
    fprintf(fp, "#define MLP_W1_SCALE %.6ff\n", w1_scale);
    fprintf(fp, "#define MLP_W2_SCALE %.6ff\n\n", w2_scale);

    // W1: [HIDDEN_SIZE][NUM_FEATURES] (transposed for efficient MVM)
    fprintf(fp, "// FC1 weights: %d x %d (transposed)\n", HIDDEN_SIZE, NUM_FEATURES);
    fprintf(fp, "static const elem_t mlp_w1[%d][%d] = {\n", HIDDEN_SIZE, NUM_FEATURES);
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        fprintf(fp, "    {");
        for (int i = 0; i < NUM_FEATURES; i++) {
            int16_t q = quantize(W1[i][j], w1_scale);
            fprintf(fp, "%d", q);
            if (i < NUM_FEATURES - 1) fprintf(fp, ", ");
        }
        fprintf(fp, "}");
        if (j < HIDDEN_SIZE - 1) fprintf(fp, ",");
        fprintf(fp, "\n");
    }
    fprintf(fp, "};\n\n");

    // b1: [HIDDEN_SIZE]
    fprintf(fp, "// FC1 biases: %d\n", HIDDEN_SIZE);
    fprintf(fp, "static const elem_t mlp_b1[%d] = {", HIDDEN_SIZE);
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        int16_t q = quantize(b1[j], w1_scale * quant_scale);  // scale bias appropriately
        fprintf(fp, "%d", q);
        if (j < HIDDEN_SIZE - 1) fprintf(fp, ", ");
    }
    fprintf(fp, "};\n\n");

    // W2: [NUM_CLASSES][HIDDEN_SIZE] (transposed)
    fprintf(fp, "// FC2 weights: %d x %d (transposed)\n", NUM_CLASSES, HIDDEN_SIZE);
    fprintf(fp, "static const elem_t mlp_w2[%d][%d] = {\n", NUM_CLASSES, HIDDEN_SIZE);
    for (int j = 0; j < NUM_CLASSES; j++) {
        fprintf(fp, "    {");
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            int16_t q = quantize(W2[i][j], w2_scale);
            fprintf(fp, "%d", q);
            if (i < HIDDEN_SIZE - 1) fprintf(fp, ", ");
        }
        fprintf(fp, "}");
        if (j < NUM_CLASSES - 1) fprintf(fp, ",");
        fprintf(fp, "\n");
    }
    fprintf(fp, "};\n\n");

    // b2: [NUM_CLASSES]
    fprintf(fp, "// FC2 biases: %d\n", NUM_CLASSES);
    fprintf(fp, "static const elem_t mlp_b2[%d] = {", NUM_CLASSES);
    for (int j = 0; j < NUM_CLASSES; j++) {
        int16_t q = quantize(b2[j], w2_scale);
        fprintf(fp, "%d", q);
        if (j < NUM_CLASSES - 1) fprintf(fp, ", ");
    }
    fprintf(fp, "};\n\n");

    // Test data (quantized)
    fprintf(fp, "// Test data: %d samples x %d features\n", NUM_TEST, NUM_FEATURES);
    fprintf(fp, "static const elem_t mlp_test_X[%d][%d] = {\n", NUM_TEST, NUM_FEATURES);
    for (int i = 0; i < NUM_TEST; i++) {
        fprintf(fp, "    {");
        for (int j = 0; j < NUM_FEATURES; j++) {
            int16_t q = quantize(test_X[i][j], quant_scale);
            fprintf(fp, "%d", q);
            if (j < NUM_FEATURES - 1) fprintf(fp, ", ");
        }
        fprintf(fp, "}");
        if (i < NUM_TEST - 1) fprintf(fp, ",");
        fprintf(fp, "\n");
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "// Test labels\n");
    fprintf(fp, "static const elem_t mlp_test_Y[%d] = {", NUM_TEST);
    for (int i = 0; i < NUM_TEST; i++) {
        fprintf(fp, "%d", test_Y[i]);
        if (i < NUM_TEST - 1) fprintf(fp, ", ");
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "#endif /* MLP_IRIS_WEIGHTS_H */\n");

    fclose(fp);
    printf("[INFO] Saved quantized weights to: %s\n", filename);
}

int main(int argc, char *argv[]) {
    int quant_bits = DEFAULT_QUANT_BITS;
    if (argc > 1) {
        quant_bits = atoi(argv[1]);
        if (quant_bits < 2 || quant_bits > 16) {
            printf("Invalid quant_bits. Using default: %d\n", DEFAULT_QUANT_BITS);
            quant_bits = DEFAULT_QUANT_BITS;
        }
    }

    printf("============================================\n");
    printf("  MLP Training (Host) for FIONA\n");
    printf("============================================\n");
    printf("  Architecture: %d -> %d -> %d\n", NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES);
    printf("  Epochs: %d\n", NUM_EPOCHS);
    printf("  Learning rate: %.4f\n", LEARNING_RATE);
    printf("  Quantization: %d-bit\n", quant_bits);
    printf("============================================\n\n");

    init_weights();

    printf("=== Training ===\n");
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        forward(train_X, NUM_TRAIN);
        float loss = backward(train_X, train_Y, NUM_TRAIN);
        update_weights();

        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            float train_acc = evaluate(train_X, train_Y, NUM_TRAIN);
            float test_acc = evaluate(test_X, test_Y, NUM_TEST);
            printf("Epoch %3d: loss=%.4f, train=%.2f%%, test=%.2f%%\n",
                   epoch + 1, loss, train_acc * 100.0f, test_acc * 100.0f);
        }
    }

    printf("\n=== Final Results ===\n");
    float train_acc = evaluate(train_X, train_Y, NUM_TRAIN);
    float test_acc = evaluate(test_X, test_Y, NUM_TEST);
    printf("Train accuracy: %.2f%%\n", train_acc * 100.0f);
    printf("Test accuracy:  %.2f%%\n", test_acc * 100.0f);

    // Save weights as C header
    printf("\n=== Saving Weights ===\n");
    save_weights_header("mlp_iris_weights.h", quant_bits);

    printf("\n============================================\n");
    printf("  Training Complete!\n");
    printf("============================================\n");

    return 0;
}
