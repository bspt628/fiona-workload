/**
 * @file main.cc
 * @brief MLP Training on Iris Dataset with Backpropagation
 *
 * This demonstrates on-device training using traditional
 * gradient descent with backpropagation.
 *
 * Training Pipeline:
 *   1. Forward pass: X -> FC1 -> ReLU -> FC2 -> Softmax -> Y_pred
 *   2. Loss computation: Cross-Entropy(Y_pred, Y_true)
 *   3. Backward pass: Compute gradients dL/dW for each layer
 *   4. Weight update: W = W - lr * dL/dW
 *
 * @author FIONA Project
 * @date 2025-12-05
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include the backprop library
#include "../../lib/nn/backprop.h"

// ============================================================
// Iris Dataset Configuration
// ============================================================
#define NUM_TRAIN    120
#define NUM_TEST     30
#define NUM_FEATURES 4
#define NUM_CLASSES  3

// Training hyperparameters
#define HIDDEN_SIZE  16
#define NUM_EPOCHS   50
#define BATCH_SIZE   10
#define LEARNING_RATE (FIXED_SCALE / 5)  // 0.2

// ============================================================
// Iris Dataset (Standardized and Quantized)
// Values scaled by FIXED_SCALE (256)
// ============================================================

// Training data: 120 samples (40 per class)
static const elem_t iris_train_X[NUM_TRAIN][NUM_FEATURES] = {
    // Class 0: Setosa (40 samples)
    {-307, 77, -366, -294}, {-358, -26, -366, -294}, {-409, 26, -392, -294},
    {-358, -77, -366, -294}, {-281, 128, -366, -294}, {-179, 230, -315, -243},
    {-358, 77, -366, -243}, {-281, 77, -341, -294}, {-409, -128, -366, -294},
    {-358, 26, -341, -243}, {-230, 179, -341, -294}, {-281, 77, -315, -243},
    {-358, -26, -366, -294}, {-460, -26, -392, -294}, {-128, 281, -366, -243},
    {-128, 358, -315, -192}, {-230, 230, -366, -243}, {-307, 77, -341, -243},
    {-128, 179, -290, -243}, {-307, 128, -315, -243}, {-230, 77, -290, -294},
    {-307, 179, -341, -243}, {-409, 128, -392, -294}, {-307, 26, -264, -192},
    {-281, 77, -264, -243}, {-281, -26, -315, -294}, {-281, 77, -315, -192},
    {-230, 77, -341, -294}, {-230, 26, -341, -294}, {-281, 26, -315, -243},
    {-281, -26, -315, -243}, {-230, 179, -290, -192}, {-230, 281, -366, -294},
    {-179, 307, -366, -243}, {-358, 26, -341, -243}, {-281, 77, -366, -294},
    {-179, 179, -366, -294}, {-358, 128, -341, -294}, {-409, -26, -366, -294},
    {-307, 77, -315, -294},
    // Class 1: Versicolor (40 samples)
    {51, -26, 77, 13}, {-51, -128, 51, 13}, {13, -77, 77, 51}, {-281, -102, 26, 13},
    {-51, -77, 77, 51}, {-102, -77, 51, 13}, {13, -13, 77, 90}, {-358, -179, -13, -64},
    {-51, -102, 77, 51}, {-230, -128, 26, -26}, {-358, -256, -13, -26}, {-102, -26, 51, 51},
    {-179, -179, 26, -26}, {51, -77, 102, 51}, {-179, -13, -13, 13}, {13, -26, 77, 13},
    {-102, -26, 77, 90}, {-179, -77, 51, 13}, {13, -204, 77, 13}, {-179, -102, 26, 13},
    {51, -102, 128, 90}, {-102, -77, 51, 51}, {102, -102, 102, 51}, {51, -128, 77, 51},
    {-51, -77, 77, 13}, {-51, -26, 77, 51}, {13, -77, 102, 51}, {51, -26, 102, 51},
    {-51, -77, 51, 13}, {-281, -102, 26, 51}, {-281, -128, 26, 13}, {-51, -26, 51, 51},
    {-179, -128, 51, 13}, {-51, -102, 77, 90}, {-179, -26, 51, 13}, {13, -26, 77, 51},
    {51, 13, 77, 51}, {-51, -77, 77, -26}, {-179, -77, 51, 13}, {-102, -77, 77, 51},
    // Class 2: Virginica (40 samples)
    {153, -26, 204, 256}, {13, -77, 128, 141}, {204, -26, 204, 179}, {51, -77, 141, 141},
    {102, -26, 166, 179}, {256, -13, 230, 179}, {-358, -128, 77, 90}, {204, -77, 204, 141},
    {102, -128, 141, 141}, {204, 77, 204, 294}, {51, 13, 141, 179}, {13, -128, 128, 141},
    {102, -26, 166, 141}, {13, -128, 141, 90}, {51, -13, 141, 179}, {102, -13, 166, 256},
    {51, -26, 141, 141}, {256, 102, 230, 217}, {256, -128, 256, 179}, {-102, -179, 102, 90},
    {153, -13, 204, 256}, {13, -77, 128, 179}, {256, -77, 230, 141}, {51, -102, 128, 141},
    {102, -26, 141, 179}, {153, -77, 166, 141}, {13, -77, 141, 90}, {13, -26, 128, 141},
    {102, -77, 166, 179}, {204, -13, 166, 217}, {204, -102, 204, 141}, {307, 77, 230, 256},
    {102, -77, 166, 256}, {51, -77, 128, 90}, {13, -77, 128, 141}, {204, -26, 204, 179},
    {102, 13, 166, 141}, {51, -26, 141, 141}, {-51, -77, 102, 141}, {102, -13, 166, 179}
};

// Training labels (class index: 0, 1, or 2)
static const elem_t iris_train_Y[NUM_TRAIN] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
};

// Test data: 30 samples (10 per class)
static const elem_t iris_test_X[NUM_TEST][NUM_FEATURES] = {
    // Class 0: Setosa (10 samples)
    {-307, 26, -366, -294}, {-230, 77, -341, -294}, {-179, 230, -392, -294},
    {-307, 179, -366, -243}, {-230, 128, -315, -243}, {-358, 77, -341, -294},
    {-179, 128, -341, -294}, {-256, 102, -366, -294}, {-281, 128, -341, -243},
    {-358, 26, -392, -294},
    // Class 1: Versicolor (10 samples)
    {-102, -128, 51, 13}, {13, -77, 77, 51}, {-51, -77, 51, 13},
    {-128, -77, 51, 51}, {13, -128, 102, 51}, {-179, -102, 51, 51},
    {-102, -77, 77, 13}, {-51, -51, 77, 51}, {-128, -102, 77, 51},
    {51, -77, 77, 13},
    // Class 2: Virginica (10 samples)
    {102, -26, 166, 217}, {153, -77, 179, 141}, {51, -51, 141, 179},
    {102, 26, 179, 217}, {204, -26, 204, 179}, {153, -102, 179, 141},
    {102, -77, 166, 179}, {51, -77, 153, 179}, {153, 26, 179, 217},
    {204, -26, 179, 179}
};

// Test labels
static const elem_t iris_test_Y[NUM_TEST] = {
    0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2
};

// ============================================================
// Main Training Program
// ============================================================

int main() {
    printf("============================================\n");
    printf("  FIONA MLP Training with Backpropagation\n");
    printf("============================================\n");
    printf("  Dataset: Iris\n");
    printf("  Architecture: %d -> %d -> %d\n", NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES);
    printf("  Training samples: %d\n", NUM_TRAIN);
    printf("  Test samples: %d\n", NUM_TEST);
    printf("  Epochs: %d\n", NUM_EPOCHS);
    printf("  Batch size: %d\n", BATCH_SIZE);
    printf("  Learning rate: %.2f\n", (float)LEARNING_RATE / FIXED_SCALE);
    printf("============================================\n\n");

    // Initialize random seed
    srand(42);

    // Initialize MLP
    SimpleMLP mlp;
    mlp_init(&mlp, NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES);

    printf("=== Initial Evaluation ===\n");
    float train_acc = mlp_evaluate(&mlp, &iris_train_X[0][0], iris_train_Y, NUM_TRAIN);
    float test_acc = mlp_evaluate(&mlp, &iris_test_X[0][0], iris_test_Y, NUM_TEST);
    printf("Train accuracy: %.2f%%\n", train_acc * 100);
    printf("Test accuracy:  %.2f%%\n\n", test_acc * 100);

    // Training loop
    printf("=== Training ===\n");
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        elem_t loss = mlp_train_epoch(&mlp, &iris_train_X[0][0], iris_train_Y,
                                      NUM_TRAIN, BATCH_SIZE, LEARNING_RATE);

        // Evaluate every 10 epochs
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            train_acc = mlp_evaluate(&mlp, &iris_train_X[0][0], iris_train_Y, NUM_TRAIN);
            test_acc = mlp_evaluate(&mlp, &iris_test_X[0][0], iris_test_Y, NUM_TEST);
            printf("Epoch %3d: loss=%.4f, train_acc=%.2f%%, test_acc=%.2f%%\n",
                   epoch + 1, (float)loss / FIXED_SCALE, train_acc * 100, test_acc * 100);
        }
    }

    // Final evaluation
    printf("\n=== Final Evaluation ===\n");
    train_acc = mlp_evaluate(&mlp, &iris_train_X[0][0], iris_train_Y, NUM_TRAIN);
    test_acc = mlp_evaluate(&mlp, &iris_test_X[0][0], iris_test_Y, NUM_TEST);
    printf("Train accuracy: %.2f%%\n", train_acc * 100);
    printf("Test accuracy:  %.2f%%\n", test_acc * 100);

    // Print predictions
    printf("\n=== Test Predictions ===\n");
    elem_t *hidden_out = (elem_t *)malloc(NUM_TEST * HIDDEN_SIZE * sizeof(elem_t));
    elem_t *relu_out = (elem_t *)malloc(NUM_TEST * HIDDEN_SIZE * sizeof(elem_t));
    elem_t *output = (elem_t *)malloc(NUM_TEST * NUM_CLASSES * sizeof(elem_t));

    mlp_forward(&mlp, output, &iris_test_X[0][0], NUM_TEST, hidden_out, relu_out);

    printf("Pred: ");
    for (int i = 0; i < NUM_TEST; i++) {
        elem_t max_val = output[i * NUM_CLASSES];
        int pred = 0;
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (output[i * NUM_CLASSES + c] > max_val) {
                max_val = output[i * NUM_CLASSES + c];
                pred = c;
            }
        }
        printf("%d ", pred);
    }
    printf("\n");

    printf("True: ");
    for (int i = 0; i < NUM_TEST; i++) {
        printf("%d ", iris_test_Y[i]);
    }
    printf("\n");

    // Clean up
    free(hidden_out);
    free(relu_out);
    free(output);
    mlp_free(&mlp);

    printf("\n============================================\n");
    printf("  Training Complete!\n");
    printf("============================================\n");

    return 0;
}
