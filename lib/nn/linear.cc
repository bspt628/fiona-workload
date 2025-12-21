#include "nn/common.h"
#include "nn/linear.h"

/******************** Linear ********************/
void nn_linear(elem_t *y, const elem_t *w, const elem_t *x, size_t feature_in, size_t feature_out, size_t batch_size) {
    // @w: feature_out * feature_in
    // @x: batch_size * feature_in
    if(y == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: nn_linear().\n");
        printf("[HINT] elem_t y[batch_size=%d][feature_out=%d];\n", batch_size, feature_out);
        exit(-1);
    }
    tiled_matmul_transpose(y, w, x, feature_out, feature_in, batch_size);
}

void nn_linear(elem_t *y, const elem_t *w, const elem_t *x, const elem_t *b, size_t feature_in, size_t feature_out, size_t batch_size) {
    // @w: feature_out * feature_in
    // @x: batch_size * feature_in
    // @b: feature_out
    if(y == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: nn_linear().\n");
        printf("[HINT] elem_t y[batch_size=%d][feature_out=%d];\n", batch_size, feature_out);
        exit(-1);
    }
    tiled_matmul_transpose(y, w, x, feature_out, feature_in, batch_size);
    tiled_matrix_vector_add(y, y, b, batch_size, feature_out);
}

/******************** Stride-aware Linear for VLSU 64-byte alignment ********************/
void nn_linear_strided(elem_t *y, const elem_t *w, const elem_t *x,
                       size_t feature_in, size_t feature_out, size_t batch_size,
                       size_t stride_w, size_t stride_x) {
    // @w: logical size feature_out * feature_in, physical row stride = stride_w
    // @x: logical size batch_size * feature_in, physical row stride = stride_x
    // stride_w, stride_x should be multiples of EU_VEC_ELEM (32) for 64-byte alignment
    if(y == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: nn_linear_strided().\n");
        printf("[HINT] elem_t y[batch_size=%d][feature_out=%d];\n", batch_size, feature_out);
        exit(-1);
    }
    tiled_matmul_transpose_strided(y, w, x, feature_out, feature_in, batch_size, stride_w, stride_x);
}

void nn_linear_full_strided(elem_t *y, const elem_t *w, const elem_t *x,
                            size_t feature_in, size_t feature_out, size_t batch_size,
                            size_t stride_w, size_t stride_x, size_t stride_y) {
    // @w: logical size feature_out * feature_in, physical row stride = stride_w
    // @x: logical size batch_size * feature_in, physical row stride = stride_x
    // @y: logical size batch_size * feature_out, physical row stride = stride_y
    // All strides should be multiples of EU_VEC_ELEM (32) for 64-byte alignment
    if(y == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: nn_linear_full_strided().\n");
        printf("[HINT] elem_t y[batch_size=%d][stride_y=%d];\n", batch_size, stride_y);
        exit(-1);
    }
    tiled_matmul_transpose_full_strided(y, w, x, feature_out, feature_in, batch_size, stride_w, stride_x, stride_y);
}
