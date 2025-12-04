/**
 * @file backprop.h
 * @brief Backpropagation training library for FIONA
 *
 * This library provides traditional gradient descent training
 * with backpropagation for neural networks.
 *
 * Supported operations:
 * - Forward pass: Linear, ReLU, Softmax
 * - Backward pass: Gradient computation
 * - Weight update: SGD, SGD with momentum
 *
 * @author FIONA Project
 * @date 2025-12-05
 */

#ifndef FIONA_BACKPROP_H
#define FIONA_BACKPROP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../base/config.h"

// ============================================================
// Configuration
// ============================================================

// Use fixed-point arithmetic for hardware compatibility
// All values are scaled by FIXED_SCALE
#define FIXED_SCALE 256
#define FIXED_SHIFT 8

// Learning rate (scaled)
#define DEFAULT_LR (FIXED_SCALE / 10)  // 0.1 in fixed-point

// ============================================================
// Utility Functions
// ============================================================

/**
 * @brief Fixed-point multiplication
 */
static inline elem_t fixed_mul(elem_t a, elem_t b) {
    return (elem_t)(((int32_t)a * (int32_t)b) >> FIXED_SHIFT);
}

/**
 * @brief Fixed-point division
 */
static inline elem_t fixed_div(elem_t a, elem_t b) {
    if (b == 0) return 0;
    return (elem_t)(((int32_t)a << FIXED_SHIFT) / (int32_t)b);
}

// ============================================================
// Layer Structure
// ============================================================

typedef struct {
    size_t in_features;
    size_t out_features;
    elem_t *weights;      // [out_features x in_features]
    elem_t *bias;         // [out_features]
    elem_t *grad_weights; // Gradient of weights
    elem_t *grad_bias;    // Gradient of bias
    elem_t *input_cache;  // Cache input for backward pass
    elem_t *output_cache; // Cache output for backward pass
} LinearLayer;

typedef struct {
    size_t size;
    elem_t *mask;  // ReLU mask (1 if x > 0, else 0)
} ReLUCache;

// ============================================================
// Layer Initialization
// ============================================================

/**
 * @brief Initialize a linear layer
 */
static void linear_init(LinearLayer *layer, size_t in_features, size_t out_features) {
    layer->in_features = in_features;
    layer->out_features = out_features;

    // Allocate memory
    layer->weights = (elem_t *)calloc(out_features * in_features, sizeof(elem_t));
    layer->bias = (elem_t *)calloc(out_features, sizeof(elem_t));
    layer->grad_weights = (elem_t *)calloc(out_features * in_features, sizeof(elem_t));
    layer->grad_bias = (elem_t *)calloc(out_features, sizeof(elem_t));
    layer->input_cache = NULL;  // Allocated during forward
    layer->output_cache = NULL;

    // Xavier initialization (simplified for fixed-point)
    // std = sqrt(2 / (in + out))
    int scale = (int)(FIXED_SCALE * 0.5 / sqrt(in_features + out_features));
    for (size_t i = 0; i < out_features * in_features; i++) {
        // Simple pseudo-random initialization
        layer->weights[i] = (elem_t)((rand() % (2 * scale)) - scale);
    }
}

/**
 * @brief Free linear layer memory
 */
static void linear_free(LinearLayer *layer) {
    free(layer->weights);
    free(layer->bias);
    free(layer->grad_weights);
    free(layer->grad_bias);
    if (layer->input_cache) free(layer->input_cache);
    if (layer->output_cache) free(layer->output_cache);
}

// ============================================================
// Forward Pass
// ============================================================

/**
 * @brief Linear layer forward: y = x * W^T + b
 *
 * @param layer The linear layer
 * @param output Output tensor [batch_size x out_features]
 * @param input Input tensor [batch_size x in_features]
 * @param batch_size Number of samples
 */
static void linear_forward(LinearLayer *layer, elem_t *output,
                           const elem_t *input, size_t batch_size) {
    size_t in_f = layer->in_features;
    size_t out_f = layer->out_features;

    // Cache input for backward pass
    if (layer->input_cache) free(layer->input_cache);
    layer->input_cache = (elem_t *)malloc(batch_size * in_f * sizeof(elem_t));
    memcpy(layer->input_cache, input, batch_size * in_f * sizeof(elem_t));

    // Compute y = x * W^T + b
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_f; o++) {
            int32_t sum = 0;
            for (size_t i = 0; i < in_f; i++) {
                sum += (int32_t)input[b * in_f + i] * (int32_t)layer->weights[o * in_f + i];
            }
            output[b * out_f + o] = (elem_t)(sum >> FIXED_SHIFT) + layer->bias[o];
        }
    }
}

/**
 * @brief ReLU forward: y = max(0, x)
 */
static void relu_forward(elem_t *output, const elem_t *input,
                         ReLUCache *cache, size_t size) {
    if (cache->mask) free(cache->mask);
    cache->mask = (elem_t *)malloc(size * sizeof(elem_t));
    cache->size = size;

    for (size_t i = 0; i < size; i++) {
        if (input[i] > 0) {
            output[i] = input[i];
            cache->mask[i] = 1;
        } else {
            output[i] = 0;
            cache->mask[i] = 0;
        }
    }
}

/**
 * @brief Softmax forward (for output layer)
 * Computes softmax along the last dimension
 */
static void softmax_forward(elem_t *output, const elem_t *input,
                            size_t batch_size, size_t num_classes) {
    for (size_t b = 0; b < batch_size; b++) {
        // Find max for numerical stability
        elem_t max_val = input[b * num_classes];
        for (size_t c = 1; c < num_classes; c++) {
            if (input[b * num_classes + c] > max_val) {
                max_val = input[b * num_classes + c];
            }
        }

        // Compute exp and sum
        int32_t sum = 0;
        for (size_t c = 0; c < num_classes; c++) {
            // Approximate exp using Taylor series or lookup table
            // For simplicity, use linear approximation in valid range
            elem_t x = input[b * num_classes + c] - max_val;
            // exp(x) ≈ 1 + x + x^2/2 for small x
            // In fixed-point: exp ≈ FIXED_SCALE + x + x^2/(2*FIXED_SCALE)
            int32_t exp_val = FIXED_SCALE;
            if (x > -4 * FIXED_SCALE) {  // Prevent underflow
                exp_val = FIXED_SCALE + x + (((int32_t)x * x) >> (FIXED_SHIFT + 1));
                if (exp_val < 1) exp_val = 1;
            } else {
                exp_val = 1;  // Very small
            }
            output[b * num_classes + c] = (elem_t)exp_val;
            sum += exp_val;
        }

        // Normalize
        for (size_t c = 0; c < num_classes; c++) {
            output[b * num_classes + c] = fixed_div(
                (elem_t)((int32_t)output[b * num_classes + c] * FIXED_SCALE),
                (elem_t)sum
            );
        }
    }
}

// ============================================================
// Backward Pass
// ============================================================

/**
 * @brief Compute cross-entropy loss gradient (softmax output vs one-hot target)
 *
 * For softmax + cross-entropy, the gradient simplifies to:
 * dL/dz = softmax(z) - y_onehot
 *
 * @param grad_output Output gradient [batch_size x num_classes]
 * @param pred Predicted probabilities (softmax output)
 * @param target Target labels (class indices, not one-hot)
 * @param batch_size Number of samples
 * @param num_classes Number of classes
 * @return Average loss value
 */
static elem_t cross_entropy_backward(elem_t *grad_output, const elem_t *pred,
                                     const elem_t *target, size_t batch_size,
                                     size_t num_classes) {
    int32_t total_loss = 0;

    for (size_t b = 0; b < batch_size; b++) {
        int target_class = (int)target[b];

        for (size_t c = 0; c < num_classes; c++) {
            if (c == target_class) {
                // Gradient: pred - 1.0 (scaled)
                grad_output[b * num_classes + c] = pred[b * num_classes + c] - FIXED_SCALE;
                // Loss: -log(pred[target])
                // Approximate: loss = -log(pred) ≈ (1 - pred) for pred close to 1
                total_loss += FIXED_SCALE - pred[b * num_classes + c];
            } else {
                // Gradient: pred - 0.0
                grad_output[b * num_classes + c] = pred[b * num_classes + c];
            }
        }
    }

    return (elem_t)(total_loss / batch_size);
}

/**
 * @brief MSE loss gradient
 * dL/dy = 2 * (pred - target) / n
 */
static elem_t mse_backward(elem_t *grad_output, const elem_t *pred,
                           const elem_t *target_onehot, size_t batch_size,
                           size_t num_classes) {
    int32_t total_loss = 0;

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < num_classes; c++) {
            elem_t diff = pred[b * num_classes + c] - target_onehot[b * num_classes + c];
            grad_output[b * num_classes + c] = (2 * diff) / batch_size;
            total_loss += fixed_mul(diff, diff);
        }
    }

    return (elem_t)(total_loss / batch_size);
}

/**
 * @brief Linear layer backward
 *
 * Given dL/dy (grad_output), compute:
 * - dL/dW = (dL/dy)^T * x
 * - dL/db = sum(dL/dy, axis=0)
 * - dL/dx = dL/dy * W (for previous layer)
 */
static void linear_backward(LinearLayer *layer, elem_t *grad_input,
                            const elem_t *grad_output, size_t batch_size) {
    size_t in_f = layer->in_features;
    size_t out_f = layer->out_features;

    // Zero gradients
    memset(layer->grad_weights, 0, out_f * in_f * sizeof(elem_t));
    memset(layer->grad_bias, 0, out_f * sizeof(elem_t));

    // Compute gradients
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_f; o++) {
            elem_t g = grad_output[b * out_f + o];

            // dL/db += dL/dy
            layer->grad_bias[o] += g / batch_size;

            // dL/dW += dL/dy * x^T
            for (size_t i = 0; i < in_f; i++) {
                layer->grad_weights[o * in_f + i] +=
                    fixed_mul(g, layer->input_cache[b * in_f + i]) / batch_size;
            }
        }
    }

    // Compute grad_input = dL/dy * W (if needed for previous layer)
    if (grad_input != NULL) {
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t i = 0; i < in_f; i++) {
                int32_t sum = 0;
                for (size_t o = 0; o < out_f; o++) {
                    sum += (int32_t)grad_output[b * out_f + o] *
                           (int32_t)layer->weights[o * in_f + i];
                }
                grad_input[b * in_f + i] = (elem_t)(sum >> FIXED_SHIFT);
            }
        }
    }
}

/**
 * @brief ReLU backward
 * dL/dx = dL/dy * (x > 0 ? 1 : 0)
 */
static void relu_backward(elem_t *grad_input, const elem_t *grad_output,
                          const ReLUCache *cache) {
    for (size_t i = 0; i < cache->size; i++) {
        grad_input[i] = cache->mask[i] ? grad_output[i] : 0;
    }
}

// ============================================================
// Weight Update (Optimizer)
// ============================================================

/**
 * @brief SGD weight update
 * W = W - lr * dL/dW
 */
static void sgd_update(LinearLayer *layer, elem_t learning_rate) {
    size_t size = layer->out_features * layer->in_features;

    for (size_t i = 0; i < size; i++) {
        layer->weights[i] -= fixed_mul(learning_rate, layer->grad_weights[i]);
    }

    for (size_t i = 0; i < layer->out_features; i++) {
        layer->bias[i] -= fixed_mul(learning_rate, layer->grad_bias[i]);
    }
}

// ============================================================
// Simple MLP with Backpropagation
// ============================================================

typedef struct {
    LinearLayer fc1;
    LinearLayer fc2;
    ReLUCache relu_cache;
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
} SimpleMLP;

/**
 * @brief Initialize a simple 2-layer MLP
 */
static void mlp_init(SimpleMLP *mlp, size_t input_size, size_t hidden_size,
                     size_t output_size) {
    mlp->input_size = input_size;
    mlp->hidden_size = hidden_size;
    mlp->output_size = output_size;

    linear_init(&mlp->fc1, input_size, hidden_size);
    linear_init(&mlp->fc2, hidden_size, output_size);
    mlp->relu_cache.mask = NULL;
}

/**
 * @brief Free MLP memory
 */
static void mlp_free(SimpleMLP *mlp) {
    linear_free(&mlp->fc1);
    linear_free(&mlp->fc2);
    if (mlp->relu_cache.mask) free(mlp->relu_cache.mask);
}

/**
 * @brief MLP forward pass
 *
 * x -> Linear1 -> ReLU -> Linear2 -> Softmax -> output
 */
static void mlp_forward(SimpleMLP *mlp, elem_t *output, const elem_t *input,
                        size_t batch_size, elem_t *hidden_out, elem_t *relu_out) {
    // FC1
    linear_forward(&mlp->fc1, hidden_out, input, batch_size);

    // ReLU
    relu_forward(relu_out, hidden_out, &mlp->relu_cache,
                 batch_size * mlp->hidden_size);

    // FC2
    elem_t *logits = (elem_t *)malloc(batch_size * mlp->output_size * sizeof(elem_t));
    linear_forward(&mlp->fc2, logits, relu_out, batch_size);

    // Softmax
    softmax_forward(output, logits, batch_size, mlp->output_size);

    free(logits);
}

/**
 * @brief MLP backward pass and weight update
 */
static elem_t mlp_backward(SimpleMLP *mlp, const elem_t *pred, const elem_t *target,
                           size_t batch_size, elem_t learning_rate) {
    // Allocate gradient buffers
    elem_t *grad_softmax = (elem_t *)malloc(batch_size * mlp->output_size * sizeof(elem_t));
    elem_t *grad_fc2 = (elem_t *)malloc(batch_size * mlp->hidden_size * sizeof(elem_t));
    elem_t *grad_relu = (elem_t *)malloc(batch_size * mlp->hidden_size * sizeof(elem_t));

    // Compute loss and output gradient
    elem_t loss = cross_entropy_backward(grad_softmax, pred, target,
                                         batch_size, mlp->output_size);

    // Backward through FC2
    linear_backward(&mlp->fc2, grad_fc2, grad_softmax, batch_size);

    // Backward through ReLU
    relu_backward(grad_relu, grad_fc2, &mlp->relu_cache);

    // Backward through FC1
    linear_backward(&mlp->fc1, NULL, grad_relu, batch_size);

    // Update weights
    sgd_update(&mlp->fc2, learning_rate);
    sgd_update(&mlp->fc1, learning_rate);

    // Clean up
    free(grad_softmax);
    free(grad_fc2);
    free(grad_relu);

    return loss;
}

/**
 * @brief Train MLP for one epoch
 */
static elem_t mlp_train_epoch(SimpleMLP *mlp, const elem_t *X, const elem_t *Y,
                              size_t num_samples, size_t batch_size,
                              elem_t learning_rate) {
    elem_t total_loss = 0;
    size_t num_batches = num_samples / batch_size;

    // Allocate buffers
    elem_t *hidden_out = (elem_t *)malloc(batch_size * mlp->hidden_size * sizeof(elem_t));
    elem_t *relu_out = (elem_t *)malloc(batch_size * mlp->hidden_size * sizeof(elem_t));
    elem_t *output = (elem_t *)malloc(batch_size * mlp->output_size * sizeof(elem_t));

    for (size_t b = 0; b < num_batches; b++) {
        const elem_t *batch_X = X + b * batch_size * mlp->input_size;
        const elem_t *batch_Y = Y + b * batch_size;

        // Forward
        mlp_forward(mlp, output, batch_X, batch_size, hidden_out, relu_out);

        // Backward and update
        elem_t loss = mlp_backward(mlp, output, batch_Y, batch_size, learning_rate);
        total_loss += loss;
    }

    free(hidden_out);
    free(relu_out);
    free(output);

    return total_loss / num_batches;
}

/**
 * @brief Evaluate MLP accuracy
 */
static float mlp_evaluate(SimpleMLP *mlp, const elem_t *X, const elem_t *Y,
                          size_t num_samples) {
    elem_t *hidden_out = (elem_t *)malloc(num_samples * mlp->hidden_size * sizeof(elem_t));
    elem_t *relu_out = (elem_t *)malloc(num_samples * mlp->hidden_size * sizeof(elem_t));
    elem_t *output = (elem_t *)malloc(num_samples * mlp->output_size * sizeof(elem_t));

    mlp_forward(mlp, output, X, num_samples, hidden_out, relu_out);

    int correct = 0;
    for (size_t i = 0; i < num_samples; i++) {
        // Find argmax
        elem_t max_val = output[i * mlp->output_size];
        int pred = 0;
        for (size_t c = 1; c < mlp->output_size; c++) {
            if (output[i * mlp->output_size + c] > max_val) {
                max_val = output[i * mlp->output_size + c];
                pred = c;
            }
        }
        if (pred == (int)Y[i]) {
            correct++;
        }
    }

    free(hidden_out);
    free(relu_out);
    free(output);

    return (float)correct / num_samples;
}

#endif // FIONA_BACKPROP_H
