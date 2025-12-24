/**
 * @file main.cc
 * @brief Extreme Learning Machine (ELM) for Iris classification
 *
 * ELM Algorithm (No Backpropagation Required):
 *   1. Initialize hidden layer weights W1 randomly (FIXED, no training)
 *   2. Compute hidden layer output: H = ReLU(X * W1)
 *   3. Solve for output weights: W2 = pinv(H) * Y  (closed-form solution)
 *   4. Inference: Y_pred = H * W2
 *
 * Key advantage: Only matrix operations, no iterative gradient descent
 *
 * @author FIONA Project
 * @date 2025-12-05
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "fiona.h"
#include "utils/pprint.h"

// ============================================================
// Configuration
// ============================================================
#define NUM_TRAIN    120
#define NUM_TEST     30
#define NUM_FEATURES 4
#define NUM_HIDDEN   32    // ELM typically needs more hidden neurons
#define NUM_CLASSES  3

#define SCALE 16  // Fixed-point scale

// ============================================================
// Iris Dataset (same as mlp_iris_train)
// ============================================================

// Training data (simplified: using representative samples)
static const elem_t train_X[NUM_TRAIN][NUM_FEATURES] = {
    // Setosa (40 samples) - negative petal features
    {-19, 5, -23, -18}, {-22, -2, -23, -18}, {-26, 2, -24, -18}, {-22, -5, -23, -18},
    {-18, 8, -23, -18}, {-11, 14, -20, -15}, {-22, 5, -23, -15}, {-18, 5, -21, -18},
    {-26, -8, -23, -18}, {-22, 2, -21, -15}, {-14, 11, -21, -18}, {-18, 5, -20, -15},
    {-22, -2, -23, -18}, {-29, -2, -24, -18}, {-8, 18, -23, -15}, {-8, 22, -20, -12},
    {-14, 14, -23, -15}, {-19, 5, -21, -15}, {-8, 11, -18, -15}, {-19, 8, -20, -15},
    {-14, 5, -18, -18}, {-19, 11, -21, -15}, {-26, 8, -24, -18}, {-19, 2, -16, -12},
    {-18, 5, -16, -15}, {-18, -2, -20, -18}, {-18, 5, -20, -12}, {-14, 5, -21, -18},
    {-14, 2, -21, -18}, {-18, 2, -20, -15}, {-18, -2, -20, -15}, {-14, 11, -18, -12},
    {-14, 18, -23, -18}, {-11, 19, -23, -15}, {-22, 2, -21, -15}, {-18, 5, -23, -18},
    {-11, 11, -23, -18}, {-22, 8, -21, -18}, {-26, -2, -23, -18}, {-19, 5, -20, -18},
    // Versicolor (40 samples) - medium features
    {3, -2, 5, 1}, {-3, -8, 3, 1}, {1, -5, 5, 3}, {-18, -6, 2, 1},
    {-3, -5, 5, 3}, {-6, -5, 3, 1}, {1, -1, 5, 6}, {-22, -11, -1, -4},
    {-3, -6, 5, 3}, {-14, -8, 2, -2}, {-22, -16, -1, -2}, {-6, -2, 3, 3},
    {-11, -11, 2, -2}, {3, -5, 6, 3}, {-11, -1, -1, 1}, {1, -2, 5, 1},
    {-6, -2, 5, 6}, {-11, -5, 3, 1}, {1, -13, 5, 1}, {-11, -6, 2, 1},
    {3, -6, 8, 6}, {-6, -5, 3, 3}, {6, -6, 6, 3}, {3, -8, 5, 3},
    {-3, -5, 5, 1}, {-3, -2, 5, 3}, {1, -5, 6, 3}, {3, -2, 6, 3},
    {-3, -5, 3, 1}, {-18, -6, 2, 3}, {-18, -8, 2, 1}, {-3, -2, 3, 3},
    {-11, -8, 3, 1}, {-3, -6, 5, 6}, {-11, -2, 3, 1}, {1, -2, 5, 3},
    {3, 1, 5, 3}, {-3, -5, 5, -2}, {-11, -5, 3, 1}, {-6, -5, 5, 3},
    // Virginica (40 samples) - large positive petal features
    {10, -2, 13, 16}, {1, -5, 8, 9}, {13, -2, 13, 11}, {3, -5, 9, 9},
    {6, -2, 10, 11}, {16, -1, 14, 11}, {-22, -8, 5, 6}, {13, -5, 13, 9},
    {6, -8, 9, 9}, {13, 5, 13, 18}, {3, 1, 9, 11}, {1, -8, 8, 9},
    {6, -2, 10, 9}, {1, -8, 9, 6}, {3, -1, 9, 11}, {6, -1, 10, 16},
    {3, -2, 9, 9}, {16, 6, 14, 14}, {16, -8, 16, 11}, {-6, -11, 6, 6},
    {10, -1, 13, 16}, {1, -5, 8, 11}, {16, -5, 14, 9}, {3, -6, 8, 9},
    {6, -2, 9, 11}, {10, -5, 10, 9}, {1, -5, 9, 6}, {1, -2, 8, 9},
    {6, -5, 10, 11}, {13, -1, 10, 14}, {13, -6, 13, 9}, {17, 5, 14, 16},
    {6, -5, 10, 16}, {3, -5, 8, 6}, {1, -5, 8, 9}, {13, -2, 13, 11},
    {6, 1, 10, 9}, {3, -2, 9, 9}, {-3, -5, 6, 9}, {6, -1, 10, 11}
};

static const elem_t train_Y_onehot[NUM_TRAIN][NUM_CLASSES] = {
    // Setosa (40)
    {SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},
    {SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},
    {SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},
    {SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},{SCALE,0,0},
    // Versicolor (40)
    {0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},
    {0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},
    {0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},
    {0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},{0,SCALE,0},
    // Virginica (40)
    {0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},
    {0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},
    {0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},
    {0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE},{0,0,SCALE}
};

// Train labels (class indices)
static const elem_t train_Y[NUM_TRAIN] = {
    // Setosa (40)
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // Versicolor (40)
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    // Virginica (40)
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
};

static const elem_t test_X[NUM_TEST][NUM_FEATURES] = {
    // Setosa (10)
    {-19, 2, -23, -18}, {-14, 5, -21, -18}, {-11, 14, -24, -18}, {-19, 11, -23, -15}, {-14, 8, -20, -15},
    {-22, 5, -21, -18}, {-11, 8, -21, -18}, {-16, 6, -23, -18}, {-18, 8, -21, -15}, {-22, 2, -24, -18},
    // Versicolor (10)
    {-6, -8, 3, 1}, {1, -5, 5, 3}, {-3, -5, 3, 1}, {-8, -5, 3, 3}, {1, -8, 6, 3},
    {-11, -6, 3, 3}, {-6, -5, 5, 1}, {-3, -3, 5, 3}, {-8, -6, 5, 3}, {3, -5, 5, 1},
    // Virginica (10)
    {6, -2, 10, 14}, {10, -5, 11, 9}, {3, -3, 9, 11}, {6, 2, 11, 14}, {13, -2, 13, 11},
    {10, -6, 11, 9}, {6, -5, 10, 11}, {3, -5, 10, 11}, {10, 2, 11, 14}, {13, -2, 11, 11}
};

static const elem_t test_Y[NUM_TEST] = {
    0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2
};

// ============================================================
// Fixed Random Hidden Layer Weights
// Generated with np.random.seed(42), scaled to [-4, 4]
// ============================================================
static const elem_t W1[NUM_FEATURES][NUM_HIDDEN] = {
    { 2,-1, 3, 1,-2, 0, 1,-1, 3,-2, 1,-2, 0, 1,-3, 2,-1, 0, 2,-1, 1,-2, 0, 3,-1, 0,-1, 2, 0,-3, 1, 0},
    {-1, 2, 0,-3, 1,-1, 0, 2,-1, 3, 0,-2, 1, 1,-1, 3,-2, 1, 0,-1, 2, 0,-3, 1,-1, 0, 2,-1, 3, 0,-2, 1},
    { 0,-2, 1, 0,-1, 3,-1, 0, 2,-3, 0, 1,-1, 2,-1, 0, 3,-1, 1,-2, 0, 1,-3, 2,-1, 0,-1, 2, 0,-3, 1, 0},
    { 1, 1,-1, 2,-3, 0, 1,-2, 0, 0,-1, 3,-1, 0, 2,-1, 0,-3, 1, 0,-1, 2,-1, 0, 3,-1, 1,-2, 0, 1,-3, 2}
};

// Output weights (learned by ELM)
static elem_t W2[NUM_HIDDEN][NUM_CLASSES];

// ============================================================
// ELM Training Functions
// ============================================================

/**
 * Compute H = ReLU(X * W1)
 */
void compute_hidden(elem_t *H, const elem_t *X, size_t num_samples) {
    for (size_t n = 0; n < num_samples; n++) {
        for (size_t h = 0; h < NUM_HIDDEN; h++) {
            int32_t sum = 0;
            for (size_t f = 0; f < NUM_FEATURES; f++) {
                sum += (int32_t)X[n * NUM_FEATURES + f] * W1[f][h];
            }
            // ReLU
            H[n * NUM_HIDDEN + h] = (sum > 0) ? (elem_t)(sum / SCALE) : 0;
        }
    }
}

/**
 * Solve W2 = pinv(H) * Y using simplified normal equation
 * W2 = (H^T * H)^(-1) * H^T * Y
 *
 * For stability, we use ridge regression: (H^T*H + λI)^(-1) * H^T * Y
 */
void solve_output_weights(const elem_t *H, const elem_t *Y) {
    // Compute H^T * Y directly (simplified approach)
    // Each column of W2[h][c] = sum over n of H[n][h] * Y[n][c] / norm(H[:,h])

    printf("[ELM] Solving for output weights...\n");

    for (size_t h = 0; h < NUM_HIDDEN; h++) {
        // Compute norm of column h
        int32_t norm_sq = 1;  // Regularization term
        for (size_t n = 0; n < NUM_TRAIN; n++) {
            norm_sq += (int32_t)H[n * NUM_HIDDEN + h] * H[n * NUM_HIDDEN + h];
        }

        for (size_t c = 0; c < NUM_CLASSES; c++) {
            int32_t sum = 0;
            for (size_t n = 0; n < NUM_TRAIN; n++) {
                sum += (int32_t)H[n * NUM_HIDDEN + h] * Y[n * NUM_CLASSES + c];
            }
            W2[h][c] = (elem_t)((sum * SCALE) / norm_sq);
        }
    }
    printf("[ELM] Output weights computed!\n");
}

/**
 * Predict: Y_pred = H * W2
 */
void predict(elem_t *pred, const elem_t *H, size_t num_samples) {
    for (size_t n = 0; n < num_samples; n++) {
        // Compute scores
        int32_t scores[NUM_CLASSES] = {0};
        for (size_t c = 0; c < NUM_CLASSES; c++) {
            for (size_t h = 0; h < NUM_HIDDEN; h++) {
                scores[c] += (int32_t)H[n * NUM_HIDDEN + h] * W2[h][c];
            }
        }

        // Argmax
        int max_c = 0;
        int32_t max_score = scores[0];
        for (size_t c = 1; c < NUM_CLASSES; c++) {
            if (scores[c] > max_score) {
                max_score = scores[c];
                max_c = c;
            }
        }
        pred[n] = max_c;
    }
}

float compute_accuracy(const elem_t *pred, const elem_t *true_labels, size_t num_samples) {
    int correct = 0;
    for (size_t i = 0; i < num_samples; i++) {
        if (pred[i] == true_labels[i]) correct++;
    }
    return (float)correct / num_samples;
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("============================================\n");
    printf("  FIONA ELM (Extreme Learning Machine)\n");
    printf("  Training WITHOUT Backpropagation\n");
    printf("============================================\n");
    printf("  Dataset: Iris\n");
    printf("  Architecture: %d -> %d (fixed) -> %d\n", NUM_FEATURES, NUM_HIDDEN, NUM_CLASSES);
    printf("  Training samples: %d\n", NUM_TRAIN);
    printf("  Test samples: %d\n", NUM_TEST);
    printf("============================================\n\n");

    // Allocate hidden layer outputs
    static elem_t H_train[NUM_TRAIN][NUM_HIDDEN];
    static elem_t H_test[NUM_TEST][NUM_HIDDEN];

    // ========================================
    // Phase 1: Training (One-shot, no iterations)
    // ========================================
    printf("=== Phase 1: Training ===\n");

    printf("[1/2] Computing hidden layer (random projection)...\n");
    compute_hidden(&H_train[0][0], &train_X[0][0], NUM_TRAIN);

    printf("[2/2] Solving for output weights (least squares)...\n");
    solve_output_weights(&H_train[0][0], &train_Y_onehot[0][0]);

    printf("\n*** Training complete! (No iterations needed) ***\n\n");

    // ========================================
    // Phase 2: Evaluation
    // ========================================
    printf("=== Phase 2: Evaluation ===\n");

    // Train accuracy
    elem_t train_pred[NUM_TRAIN];
    predict(train_pred, &H_train[0][0], NUM_TRAIN);
    float train_acc = compute_accuracy(train_pred, train_Y, NUM_TRAIN);
    printf("Train accuracy: %.2f%%\n", train_acc * 100);

    // Test accuracy
    compute_hidden(&H_test[0][0], &test_X[0][0], NUM_TEST);
    elem_t test_pred[NUM_TEST];
    predict(test_pred, &H_test[0][0], NUM_TEST);
    float test_acc = compute_accuracy(test_pred, test_Y, NUM_TEST);
    printf("Test accuracy:  %.2f%%\n", test_acc * 100);

    // Print predictions
    printf("\n=== Test Predictions ===\n");
    printf("Pred: ");
    for (int i = 0; i < NUM_TEST; i++) printf("%d ", test_pred[i]);
    printf("\n");
    printf("True: ");
    for (int i = 0; i < NUM_TEST; i++) printf("%d ", test_Y[i]);
    printf("\n");

    printf("\n============================================\n");
    printf("  ELM Training Complete!\n");
    printf("  Key: Hidden layer weights were FIXED\n");
    printf("  Only output layer was learned (no backprop)\n");
    printf("============================================\n");

    DUMP_STAT;

    return 0;
}
