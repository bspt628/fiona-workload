#ifndef FIONA_NN_LINEAR_H
#define FIONA_NN_LINEAR_H

#include "math/all.h"

void nn_linear(elem_t *y, const elem_t *w, const elem_t *x, size_t feature_in, size_t feature_out, size_t batch_size);
void nn_linear(elem_t *y, const elem_t *w, const elem_t *x, const elem_t *b, size_t feature_in, size_t feature_out, size_t batch_size);

// Stride-aware version for VLSU 64-byte alignment
// stride_w, stride_x: row stride in elements (should be multiple of EU_VEC_ELEM=32 for 64-byte alignment)
void nn_linear_strided(elem_t *y, const elem_t *w, const elem_t *x,
                       size_t feature_in, size_t feature_out, size_t batch_size,
                       size_t stride_w, size_t stride_x);

// Full stride-aware version with output stride support
// stride_w, stride_x, stride_y: row strides for weight, input, and output arrays
void nn_linear_full_strided(elem_t *y, const elem_t *w, const elem_t *x,
                            size_t feature_in, size_t feature_out, size_t batch_size,
                            size_t stride_w, size_t stride_x, size_t stride_y);

#endif /* FIONA_NN_LINEAR_H */