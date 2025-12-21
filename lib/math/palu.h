#ifndef FIONA_PHOTONIC_ALU_H
#define FIONA_PHOTONIC_ALU_H

#include "base/config.h"
#include "base/instr.h"

void fit_dotprod(elem_t *retval, const elem_t *vec1, const elem_t *vec2, size_t vlen);
void tiled_dotprod(elem_t *retval, const elem_t *vec1, const elem_t *vec2, size_t vlen);
void tiled_mvm(elem_t *retval, const elem_t *mat, const elem_t *vec, size_t rows, size_t cols);
void tiled_matmul_transpose(elem_t *retval, const elem_t *mat1, const elem_t *mat2_T, size_t I, size_t J, size_t K);

// Stride-aware versions for VLSU 64-byte alignment
// tile_stride: distance between 8-element tiles in memory (should be EU_VEC_ELEM=32 for 64-byte alignment)
void tiled_dotprod_strided(elem_t *retval, const elem_t *vec1, const elem_t *vec2, size_t vlen, size_t tile_stride);
// row_stride: distance between rows in memory (should be multiple of EU_VEC_ELEM for 64-byte aligned rows)
void tiled_mvm_strided(elem_t *retval, const elem_t *mat, const elem_t *vec, size_t rows, size_t cols, size_t row_stride);
void tiled_matmul_transpose_strided(elem_t *retval, const elem_t *mat1, const elem_t *mat2_T,
                                     size_t I, size_t J, size_t K, size_t stride1, size_t stride2);

// Full stride-aware version with output stride support
void tiled_matmul_transpose_full_strided(elem_t *retval, const elem_t *mat1, const elem_t *mat2_T,
                                          size_t I, size_t J, size_t K,
                                          size_t stride1, size_t stride2, size_t stride_out);

#endif /* FIONA_PHOTONIC_ALU_H */