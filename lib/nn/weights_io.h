/**
 * @file weights_io.h
 * @brief Weight Save/Load Utilities for Neural Networks
 *
 * Provides functions to save trained weights to files and load them
 * for inference with different photonic models.
 *
 * File format:
 *   - Header: magic (4B), version (4B), num_layers (4B), reserved (4B)
 *   - Architecture: num_layers (4B), layer_sizes[] (4B each)
 *   - For each layer: LayerMeta, weights, biases
 */

#ifndef FIONA_WEIGHTS_IO_H
#define FIONA_WEIGHTS_IO_H

#include <stdint.h>

// Magic number for weight file validation
#define FIONA_WEIGHT_MAGIC 0x464E4E57  // "FNNW" = FIONA Neural Network Weights
#define FIONA_WEIGHT_VERSION 1

// Maximum supported layers
#define MAX_LAYERS 16

/**
 * @brief Weight file header structure
 */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t num_layers;
    uint32_t reserved;
} WeightFileHeader;

/**
 * @brief Layer metadata structure
 */
typedef struct {
    uint32_t in_features;
    uint32_t out_features;
    uint32_t has_bias;
    uint32_t data_type;  // 0 = float32, 1 = int16
} LayerMeta;

/**
 * @brief MLP weights structure for easy save/load
 */
typedef struct {
    int num_layers;
    int layer_sizes[MAX_LAYERS];  // Input size, hidden sizes..., output size
    float **weights;              // weights[layer][in * out]
    float **biases;               // biases[layer][out]
} MLPWeights;

/**
 * @brief Save MLP weights to a binary file
 *
 * @param filename Output file path
 * @param mlp Pointer to MLP weights structure
 * @return 0 on success, -1 on error
 */
int save_mlp_weights(const char *filename, const MLPWeights *mlp);

/**
 * @brief Load MLP weights from a binary file
 *
 * @param filename Input file path
 * @param mlp Pointer to MLP weights structure (will be allocated)
 * @return 0 on success, -1 on error
 */
int load_mlp_weights(const char *filename, MLPWeights *mlp);

/**
 * @brief Free allocated MLP weights
 *
 * @param mlp Pointer to MLP weights structure
 */
void free_mlp_weights(MLPWeights *mlp);

/**
 * @brief Print MLP architecture summary
 *
 * @param mlp Pointer to MLP weights structure
 */
void print_mlp_summary(const MLPWeights *mlp);

#endif /* FIONA_WEIGHTS_IO_H */
