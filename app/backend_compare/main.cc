/**
 * Backend Comparison Benchmark
 *
 * This application runs the same workload on different backends
 * (FIONA, CPU, Gemmini) and measures performance for comparison.
 *
 * Usage:
 *   # Build and run with FIONA backend
 *   make BACKEND=fiona build-fiona/backend_compare.elf
 *   spike --extension=fiona pk build-fiona/backend_compare.elf
 *
 *   # Build and run with CPU baseline
 *   make BACKEND=cpu build-cpu/backend_compare.elf
 *   spike pk build-cpu/backend_compare.elf
 *
 *   # Compare the cycle counts from both runs
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fiona.h"
#include "utils/perf.h"

//=============================================================================
// Test Parameters
//=============================================================================

#define TEST_VLEN 8          // Vector length for basic tests
#define TEST_ROWS 16         // Matrix rows
#define TEST_COLS 16         // Matrix columns
#define TEST_ITERATIONS 100  // Number of iterations for timing

//=============================================================================
// Test Data (static allocation)
//=============================================================================

static elem_t vec_a[TEST_COLS];
static elem_t vec_b[TEST_COLS];
static elem_t vec_c[TEST_COLS];
static elem_t mat_a[TEST_ROWS * TEST_COLS];
static elem_t mat_b[TEST_ROWS * TEST_COLS];
static elem_t result_vec[TEST_ROWS];
static elem_t result_mat[TEST_ROWS * TEST_COLS];

//=============================================================================
// Initialize Test Data
//=============================================================================

void init_test_data(void) {
    // Initialize vectors with simple patterns
    for (size_t i = 0; i < TEST_COLS; i++) {
        vec_a[i] = (elem_t)(i + 1);
        vec_b[i] = (elem_t)(TEST_COLS - i);
    }

    // Initialize matrices
    for (size_t i = 0; i < TEST_ROWS; i++) {
        for (size_t j = 0; j < TEST_COLS; j++) {
            mat_a[i * TEST_COLS + j] = (elem_t)((i + j) % 128);
            mat_b[i * TEST_COLS + j] = (elem_t)((i * j) % 128);
        }
    }

    // Clear results
    memset(vec_c, 0, sizeof(vec_c));
    memset(result_vec, 0, sizeof(result_vec));
    memset(result_mat, 0, sizeof(result_mat));
}

//=============================================================================
// Benchmark Functions
//=============================================================================

void benchmark_vector_add(int iterations) {
    PERF_START("vector_add");
    for (int iter = 0; iter < iterations; iter++) {
        tiled_vector_add(vec_c, vec_a, vec_b, TEST_COLS);
    }
    PERF_END();
}

void benchmark_vector_mul_scalar(int iterations) {
    PERF_START("vector_mul_scalar");
    for (int iter = 0; iter < iterations; iter++) {
        tiled_vector_mul_scalar(vec_c, vec_a, 2, TEST_COLS);
    }
    PERF_END();
}

void benchmark_dot_product(int iterations) {
    PERF_START("dot_product");
    for (int iter = 0; iter < iterations; iter++) {
        elem_t dot_result;
        tiled_dotprod(&dot_result, vec_a, vec_b, TEST_COLS);
    }
    PERF_END();
}

void benchmark_mvm(int iterations) {
    PERF_START("mvm");
    for (int iter = 0; iter < iterations; iter++) {
        tiled_mvm(result_vec, mat_a, vec_a, TEST_ROWS, TEST_COLS);
    }
    PERF_END();
}

void benchmark_relu(int iterations) {
    PERF_START("relu");
    for (int iter = 0; iter < iterations; iter++) {
        tiled_vector_relu(vec_c, vec_a, TEST_COLS);
    }
    PERF_END();
}

void benchmark_linear(int iterations) {
    PERF_START("linear");
    for (int iter = 0; iter < iterations; iter++) {
        nn_linear(result_vec, mat_a, vec_a, TEST_COLS, TEST_ROWS, 1);
    }
    PERF_END();
}

//=============================================================================
// Main
//=============================================================================

int main(void) {
    printf("\n");
    printf("================================================\n");
    printf("  Backend Comparison Benchmark\n");
    printf("================================================\n");
    printf("Backend: %s\n", BACKEND_NAME);
    printf("Description: %s\n", BACKEND_DESC);
    printf("------------------------------------------------\n");
    printf("Test Parameters:\n");
    printf("  Vector Length: %d\n", TEST_COLS);
    printf("  Matrix Size: %d x %d\n", TEST_ROWS, TEST_COLS);
    printf("  Iterations: %d\n", TEST_ITERATIONS);
    printf("================================================\n\n");

    // Initialize
    PERF_INIT();
    init_test_data();

    // Run benchmarks
    printf("Running benchmarks...\n\n");

    benchmark_vector_add(TEST_ITERATIONS);
    benchmark_vector_mul_scalar(TEST_ITERATIONS);
    benchmark_dot_product(TEST_ITERATIONS);
    benchmark_mvm(TEST_ITERATIONS);
    benchmark_relu(TEST_ITERATIONS);
    benchmark_linear(TEST_ITERATIONS);

    // Report results
    PERF_REPORT();

    // Verification (print some results to ensure correctness)
    printf("\n");
    printf("Verification (sample results):\n");
    printf("  vec_c[0] = %d\n", vec_c[0]);
    printf("  result_vec[0] = %d\n", result_vec[0]);

    printf("\n");
    printf("================================================\n");
    printf("  Benchmark Complete\n");
    printf("================================================\n");
    printf("\n");
    printf("To compare backends:\n");
    printf("  1. Run with BACKEND=fiona and note TOTAL cycles\n");
    printf("  2. Run with BACKEND=cpu and note TOTAL cycles\n");
    printf("  3. Speedup = CPU_cycles / FIONA_cycles\n");
    printf("\n");

    return 0;
}
