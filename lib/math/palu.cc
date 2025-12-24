#include "math/palu.h"
#include <cstring>

void fit_dotprod(elem_t *retval, const elem_t *vec1, const elem_t *vec2, size_t vlen) {
    assert(vlen <= PU_MRR_RING);
    SET_VLEN(vlen);
    LOAD_V(1, vec1);  // @x1: [v]
    LOAD_V(2, vec2);  // @x2: [v]
    DOTP(*retval, 1, 2);   // @retval: [s]
}

void tiled_dotprod(elem_t *retval, const elem_t *vec1, const elem_t *vec2, size_t vlen) {
    if(vlen <= PU_MRR_RING) {
        fit_dotprod(retval, vec1, vec2, vlen);
    } else {
        size_t remainder = vlen % PU_MRR_RING;
        size_t loop_num = vlen / PU_MRR_RING;
        elem_t sum = 0, val = 0;
        for(size_t i = 0; i < loop_num; ++i) {
            size_t index = i * PU_MRR_RING;
            fit_dotprod(&val, &vec1[index], &vec2[index], PU_MRR_RING);
            sum += val;
        }
        if(remainder > 0) {
            size_t offset = vlen - remainder;
            fit_dotprod(&val, &vec1[offset], &vec2[offset], remainder);
            sum += val;
        }
        *retval = sum;
    }
}

// Temporary aligned buffers for MVM tile operations
static elem_t mvm_tile_mat[32 * 32] __attribute__((aligned(64)));
static elem_t mvm_tile_vec[32] __attribute__((aligned(64)));
static elem_t mvm_tile_out[32] __attribute__((aligned(64)));

// Helper: execute single tile MVM using MVM instruction
static void fit_mvm_tile(elem_t *out, const elem_t *tile_mat, const elem_t *tile_vec, size_t tile_size) {
    SET_VLEN(tile_size);
    SET_MAT(tile_mat);      // Load matrix (tile_size x tile_size), stride=1 by default
    LOAD_V(1, tile_vec);    // Load input vector
    MVM(0, 1);              // Execute MVM -> calls Python mvm() once
    STORE_V(0, out);        // Store result
}

void tiled_mvm(elem_t *retval, const elem_t *mat, const elem_t *vec, size_t rows, size_t cols) {
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_mvm().\n");
        printf("[HINT] elem_t retval[rows=%d];\n", rows);
        exit(-1);
    }

    // Initialize output to zero
    for(size_t i = 0; i < rows; ++i) {
        retval[i] = 0;
    }

    const size_t TILE_SIZE = 32;  // Max MVM tile size

    // Tile over output dimension (rows)
    for(size_t row_tile = 0; row_tile < rows; row_tile += TILE_SIZE) {
        size_t tile_rows = (row_tile + TILE_SIZE <= rows) ? TILE_SIZE : (rows - row_tile);

        // Tile over input dimension (cols) - results are accumulated
        for(size_t col_tile = 0; col_tile < cols; col_tile += TILE_SIZE) {
            size_t tile_cols = (col_tile + TILE_SIZE <= cols) ? TILE_SIZE : (cols - col_tile);

            // Determine tile size for MVM (must be square for SET_MAT)
            size_t tile_size = (tile_rows > tile_cols) ? tile_rows : tile_cols;

            // Prepare tile matrix (tile_size x tile_size, zero-padded)
            memset(mvm_tile_mat, 0, TILE_SIZE * TILE_SIZE * sizeof(elem_t));
            for(size_t i = 0; i < tile_rows; i++) {
                for(size_t j = 0; j < tile_cols; j++) {
                    mvm_tile_mat[i * tile_size + j] = mat[(row_tile + i) * cols + (col_tile + j)];
                }
            }

            // Prepare tile vector (zero-padded)
            memset(mvm_tile_vec, 0, TILE_SIZE * sizeof(elem_t));
            for(size_t j = 0; j < tile_cols; j++) {
                mvm_tile_vec[j] = vec[col_tile + j];
            }

            // Execute tile MVM
            memset(mvm_tile_out, 0, TILE_SIZE * sizeof(elem_t));
            fit_mvm_tile(mvm_tile_out, mvm_tile_mat, mvm_tile_vec, tile_size);

            // Accumulate results
            for(size_t i = 0; i < tile_rows; i++) {
                retval[row_tile + i] += mvm_tile_out[i];
            }
        }
    }
}

void tiled_matmul_transpose(elem_t *retval, const elem_t *mat1, const elem_t *mat2_T, size_t I, size_t J, size_t K) {
    // @mat1: size I*J
    // @mat2: size J*K --> mat2_T: size K*J
    // @retval: size K*I (column-major)
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matmul_transpose().\n");
        printf("[HINT] elem_t retval[K=%d][I=%d];\n", K, I);
        exit(-1);
    }
    for(size_t k = 0; k < K; ++k) {
        tiled_mvm(&retval[k * I], &mat1[0], &mat2_T[k * J], I, J);
    }
}

/******************** Stride-aware versions for VLSU 64-byte alignment ********************/

void tiled_dotprod_strided(elem_t *retval, const elem_t *vec1, const elem_t *vec2, size_t vlen, size_t tile_stride) {
    // tile_stride: memory distance between 8-element tiles (should be EU_VEC_ELEM=32 for 64-byte alignment)
    if(vlen <= PU_MRR_RING) {
        fit_dotprod(retval, vec1, vec2, vlen);
    } else {
        size_t remainder = vlen % PU_MRR_RING;
        size_t loop_num = vlen / PU_MRR_RING;
        elem_t sum = 0, val = 0;
        for(size_t i = 0; i < loop_num; ++i) {
            // Use tile_stride for address calculation (not PU_MRR_RING)
            size_t offset = i * tile_stride;
            fit_dotprod(&val, &vec1[offset], &vec2[offset], PU_MRR_RING);
            sum += val;
        }
        if(remainder > 0) {
            size_t offset = loop_num * tile_stride;
            fit_dotprod(&val, &vec1[offset], &vec2[offset], remainder);
            sum += val;
        }
        *retval = sum;
    }
}

void tiled_mvm_strided(elem_t *retval, const elem_t *mat, const elem_t *vec, size_t rows, size_t cols, size_t row_stride) {
    // row_stride: memory distance between rows (should be multiple of EU_VEC_ELEM for 64-byte alignment)
    // cols: actual number of elements per row (for computation)
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_mvm_strided().\n");
        printf("[HINT] elem_t retval[rows=%d];\n", rows);
        exit(-1);
    }
    // Calculate tile_stride for within-row tiling
    // For 64-byte alignment, tiles should be EU_VEC_ELEM apart
    size_t tile_stride = EU_VEC_ELEM;
    for(size_t i = 0; i < rows; ++i) {
        tiled_dotprod_strided(&retval[i], &mat[i * row_stride], &vec[0], cols, tile_stride);
    }
}

void tiled_matmul_transpose_strided(elem_t *retval, const elem_t *mat1, const elem_t *mat2_T,
                                     size_t I, size_t J, size_t K, size_t stride1, size_t stride2) {
    // @mat1: logical size I*J, physical row stride = stride1
    // @mat2_T: logical size K*J, physical row stride = stride2
    // @retval: size K*I (column-major)
    // stride1, stride2: memory distance between rows for mat1 and mat2_T
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matmul_transpose_strided().\n");
        printf("[HINT] elem_t retval[K=%d][I=%d];\n", K, I);
        exit(-1);
    }
    for(size_t k = 0; k < K; ++k) {
        tiled_mvm_strided(&retval[k * I], &mat1[0], &mat2_T[k * stride2], I, J, stride1);
    }
}

void tiled_matmul_transpose_full_strided(elem_t *retval, const elem_t *mat1, const elem_t *mat2_T,
                                          size_t I, size_t J, size_t K,
                                          size_t stride1, size_t stride2, size_t stride_out) {
    // @mat1: logical size I*J, physical row stride = stride1
    // @mat2_T: logical size K*J, physical row stride = stride2
    // @retval: logical size K*I, physical row stride = stride_out
    // stride1, stride2: memory distance between rows for mat1 and mat2_T
    // stride_out: memory distance between output rows (for padded output arrays)
    if(retval == nullptr) {
        printf("[ERROR] please allocate memory for retval before call func: tiled_matmul_transpose_full_strided().\n");
        printf("[HINT] elem_t retval[K=%d][stride_out=%d];\n", K, stride_out);
        exit(-1);
    }
    for(size_t k = 0; k < K; ++k) {
        // Output to retval[k * stride_out] instead of retval[k * I]
        tiled_mvm_strided(&retval[k * stride_out], &mat1[0], &mat2_T[k * stride2], I, J, stride1);
    }
}

