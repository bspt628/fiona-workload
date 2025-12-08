/**
 * @file weights_io.cc
 * @brief Weight Save/Load Utilities Implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn/weights_io.h"

int save_mlp_weights(const char *filename, const MLPWeights *mlp) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("[ERROR] Cannot open file for writing: %s\n", filename);
        return -1;
    }

    // Write header
    WeightFileHeader header;
    header.magic = FIONA_WEIGHT_MAGIC;
    header.version = FIONA_WEIGHT_VERSION;
    header.num_layers = (uint32_t)(mlp->num_layers - 1);  // Number of weight layers
    header.reserved = 0;
    fwrite(&header, sizeof(WeightFileHeader), 1, fp);

    // Write architecture (layer sizes)
    fwrite(&mlp->num_layers, sizeof(int), 1, fp);
    fwrite(mlp->layer_sizes, sizeof(int), mlp->num_layers, fp);

    // Write each layer's weights and biases
    for (int l = 0; l < mlp->num_layers - 1; l++) {
        int in_size = mlp->layer_sizes[l];
        int out_size = mlp->layer_sizes[l + 1];

        LayerMeta meta;
        meta.in_features = (uint32_t)in_size;
        meta.out_features = (uint32_t)out_size;
        meta.has_bias = 1;
        meta.data_type = 0;  // float32
        fwrite(&meta, sizeof(LayerMeta), 1, fp);

        // Write weights (row-major: out_size x in_size)
        fwrite(mlp->weights[l], sizeof(float), in_size * out_size, fp);

        // Write biases
        fwrite(mlp->biases[l], sizeof(float), out_size, fp);
    }

    fclose(fp);
    printf("[INFO] Saved weights to: %s\n", filename);
    return 0;
}

int load_mlp_weights(const char *filename, MLPWeights *mlp) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("[ERROR] Cannot open file for reading: %s\n", filename);
        return -1;
    }

    // Read and validate header
    WeightFileHeader header;
    fread(&header, sizeof(WeightFileHeader), 1, fp);

    if (header.magic != FIONA_WEIGHT_MAGIC) {
        printf("[ERROR] Invalid weight file format (magic mismatch)\n");
        fclose(fp);
        return -1;
    }

    if (header.version != FIONA_WEIGHT_VERSION) {
        printf("[ERROR] Unsupported weight file version: %u\n", header.version);
        fclose(fp);
        return -1;
    }

    // Read architecture
    fread(&mlp->num_layers, sizeof(int), 1, fp);
    fread(mlp->layer_sizes, sizeof(int), mlp->num_layers, fp);

    // Allocate arrays
    int num_weight_layers = mlp->num_layers - 1;
    mlp->weights = (float **)malloc(num_weight_layers * sizeof(float *));
    mlp->biases = (float **)malloc(num_weight_layers * sizeof(float *));

    // Read each layer's weights and biases
    for (int l = 0; l < num_weight_layers; l++) {
        LayerMeta meta;
        fread(&meta, sizeof(LayerMeta), 1, fp);

        int in_size = meta.in_features;
        int out_size = meta.out_features;

        // Allocate and read weights
        mlp->weights[l] = (float *)malloc(in_size * out_size * sizeof(float));
        fread(mlp->weights[l], sizeof(float), in_size * out_size, fp);

        // Allocate and read biases
        mlp->biases[l] = (float *)malloc(out_size * sizeof(float));
        fread(mlp->biases[l], sizeof(float), out_size, fp);
    }

    fclose(fp);
    printf("[INFO] Loaded weights from: %s\n", filename);
    printf("[INFO] Architecture: ");
    for (int i = 0; i < mlp->num_layers; i++) {
        printf("%d", mlp->layer_sizes[i]);
        if (i < mlp->num_layers - 1) printf(" -> ");
    }
    printf("\n");

    return 0;
}

void free_mlp_weights(MLPWeights *mlp) {
    if (mlp->weights == NULL && mlp->biases == NULL) {
        return;
    }

    int num_weight_layers = mlp->num_layers - 1;
    for (int l = 0; l < num_weight_layers; l++) {
        if (mlp->weights && mlp->weights[l]) free(mlp->weights[l]);
        if (mlp->biases && mlp->biases[l]) free(mlp->biases[l]);
    }
    if (mlp->weights) free(mlp->weights);
    if (mlp->biases) free(mlp->biases);
    mlp->weights = NULL;
    mlp->biases = NULL;
}

void print_mlp_summary(const MLPWeights *mlp) {
    printf("=== MLP Summary ===\n");
    printf("Layers: %d\n", mlp->num_layers);
    printf("Architecture: ");
    for (int i = 0; i < mlp->num_layers; i++) {
        printf("%d", mlp->layer_sizes[i]);
        if (i < mlp->num_layers - 1) printf(" -> ");
    }
    printf("\n");

    int total_params = 0;
    for (int l = 0; l < mlp->num_layers - 1; l++) {
        int in_size = mlp->layer_sizes[l];
        int out_size = mlp->layer_sizes[l + 1];
        int layer_params = in_size * out_size + out_size;
        total_params += layer_params;
        printf("  Layer %d: %d x %d weights + %d biases = %d params\n",
               l + 1, in_size, out_size, out_size, layer_params);
    }
    printf("Total parameters: %d\n", total_params);
    printf("==================\n");
}
