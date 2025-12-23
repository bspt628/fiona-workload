#ifndef FIONA_BACKEND_CPU_OPS_H
#define FIONA_BACKEND_CPU_OPS_H

/**
 * CPU Backend Operations
 *
 * Pure software implementations of FIONA accelerator operations.
 * Used as baseline for performance comparison.
 */

#include "base/config.h"
#include <string.h>

//=============================================================================
// CPU Backend - Simulated Vector Registers
//=============================================================================

// Simulated vector register file (mirrors FIONA's register structure)
#define CPU_NUM_VREGS 32
extern elem_t cpu_vregs[CPU_NUM_VREGS][EU_VEC_ELEM];
extern size_t cpu_vlen;  // Current vector length

//=============================================================================
// CPU Backend - Instruction Emulation Macros
//=============================================================================

// These macros emulate FIONA RoCC instructions in pure software

#define SET_VLEN(len) do { cpu_vlen = (len); } while(0)

#define LOAD_V(vregnum, src) do { \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        cpu_vregs[vregnum][_i] = (src)[_i]; \
    } \
} while(0)

#define STORE_V(vregnum, dst) do { \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        (dst)[_i] = cpu_vregs[vregnum][_i]; \
    } \
} while(0)

// Vector-Vector operations
#define ADD_V(vd, v1, v2) do { \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        cpu_vregs[vd][_i] = cpu_vregs[v1][_i] + cpu_vregs[v2][_i]; \
    } \
} while(0)

#define SUB_V(vd, v1, v2) do { \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        cpu_vregs[vd][_i] = cpu_vregs[v1][_i] - cpu_vregs[v2][_i]; \
    } \
} while(0)

// Vector-Scalar operations
#define ADD_VS(vd, scalar, v2) do { \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        cpu_vregs[vd][_i] = (elem_t)(scalar) + cpu_vregs[v2][_i]; \
    } \
} while(0)

#define SUB_VS(vd, scalar, v2) do { \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        cpu_vregs[vd][_i] = (elem_t)(scalar) - cpu_vregs[v2][_i]; \
    } \
} while(0)

#define MUL_VS(vd, scalar, v2) do { \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        cpu_vregs[vd][_i] = (elem_t)(scalar) * cpu_vregs[v2][_i]; \
    } \
} while(0)

#define DIV_VS(vd, scalar, v2) do { \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        cpu_vregs[vd][_i] = cpu_vregs[v2][_i] / (elem_t)(scalar); \
    } \
} while(0)

// Nonlinear operations
#define RELU_V(vd, v1) do { \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        cpu_vregs[vd][_i] = (cpu_vregs[v1][_i] > 0) ? cpu_vregs[v1][_i] : 0; \
    } \
} while(0)

// Approximated tanh for int16 (scaled by 2^14 = 16384)
#define TANH_V(vd, v1) do { \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        int32_t x = cpu_vregs[v1][_i]; \
        /* Simple piecewise linear approximation */ \
        if (x > 16384) cpu_vregs[vd][_i] = 16384; \
        else if (x < -16384) cpu_vregs[vd][_i] = -16384; \
        else cpu_vregs[vd][_i] = (elem_t)x; \
    } \
} while(0)

// Approximated sigmoid for int16 (scaled by 2^14)
#define SIGMOID_V(vd, v1) do { \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        int32_t x = cpu_vregs[v1][_i]; \
        /* Piecewise linear approximation of sigmoid */ \
        if (x > 8192) cpu_vregs[vd][_i] = 16384; \
        else if (x < -8192) cpu_vregs[vd][_i] = 0; \
        else cpu_vregs[vd][_i] = (elem_t)(8192 + x); \
    } \
} while(0)

// Dot product
// Note: FIONA uses photonic_models for scaling. CPU baseline uses simple sum.
// The result is truncated to elem_t (int16_t) range.
#define DOTP(rd, v1, v2) do { \
    int32_t _sum = 0; \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        _sum += (int32_t)cpu_vregs[v1][_i] * (int32_t)cpu_vregs[v2][_i]; \
    } \
    /* Scale to match FIONA's photonic model output range */ \
    /* Adjust CPU_DOTP_SCALE if needed for accurate comparison */ \
    (rd) = (elem_t)(_sum >> CPU_DOTP_SCALE); \
} while(0)

// Scaling factor for DOTP (adjustable for matching FIONA behavior)
#ifndef CPU_DOTP_SCALE
#define CPU_DOTP_SCALE 8
#endif

// Matrix-Vector Multiply (simplified - uses existing infrastructure)
#define SET_MAT(r1) do { /* CPU backend doesn't need this */ } while(0)
#define SET_STRIDE(r1) do { /* CPU backend doesn't need this */ } while(0)
#define MVM(vd, v1) do { /* Handled by tiled_mvm */ } while(0)

// Shuffle and reduce operations
#define SHUFFLE_V(vd, v1, v2) do { \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        size_t idx = cpu_vregs[v2][_i] % cpu_vlen; \
        cpu_vregs[vd][_i] = cpu_vregs[v1][idx]; \
    } \
} while(0)

#define MAX_V(rd, v1) do { \
    elem_t _max = cpu_vregs[v1][0]; \
    for (size_t _i = 1; _i < cpu_vlen; _i++) { \
        if (cpu_vregs[v1][_i] > _max) _max = cpu_vregs[v1][_i]; \
    } \
    (rd) = _max; \
} while(0)

#define MIN_V(rd, v1) do { \
    elem_t _min = cpu_vregs[v1][0]; \
    for (size_t _i = 1; _i < cpu_vlen; _i++) { \
        if (cpu_vregs[v1][_i] < _min) _min = cpu_vregs[v1][_i]; \
    } \
    (rd) = _min; \
} while(0)

// PReLU
#define PRELU_V(vd, alpha, v2) do { \
    for (size_t _i = 0; _i < cpu_vlen; _i++) { \
        if (cpu_vregs[v2][_i] > 0) { \
            cpu_vregs[vd][_i] = cpu_vregs[v2][_i]; \
        } else { \
            cpu_vregs[vd][_i] = (elem_t)(((int32_t)(alpha) * cpu_vregs[v2][_i]) >> 8); \
        } \
    } \
} while(0)

// FP32 operations (placeholder - CPU uses same logic)
#define LOAD_V_FP32(vregnum, src) LOAD_V(vregnum, src)
#define STORE_V_FP32(vregnum, dst) STORE_V(vregnum, dst)
#define MVM_FP32(vd, v1) MVM(vd, v1)
#define SET_VLEN_FP32(r1) SET_VLEN(r1)
#define SET_MAT_FP32(r1) SET_MAT(r1)

// Statistics dump (no-op for CPU)
#define DUMP_STAT do { } while(0)

#endif /* FIONA_BACKEND_CPU_OPS_H */
