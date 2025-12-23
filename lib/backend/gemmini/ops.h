#ifndef FIONA_BACKEND_GEMMINI_OPS_H
#define FIONA_BACKEND_GEMMINI_OPS_H

/**
 * Gemmini Backend Operations (Placeholder)
 *
 * This file will contain Gemmini systolic array operations
 * once the Gemmini submodule is integrated.
 *
 * Gemmini is a systolic array accelerator from UC Berkeley:
 * https://github.com/ucb-bar/gemmini
 *
 * TODO: Implement the following after Gemmini integration:
 *   1. Include gemmini.h from Gemmini software library
 *   2. Map FIONA operations to Gemmini primitives:
 *      - DOTP  -> gemmini_mvin + gemmini_mvout (vector dot product)
 *      - MVM   -> gemmini_matmul (matrix-vector multiply)
 *      - ADD_V -> gemmini_mvin + gemmini_mvout (element-wise add)
 *   3. Handle data layout differences (systolic vs. photonic)
 */

#include "base/config.h"

#ifdef BACKEND_GEMMINI

#error "Gemmini backend not yet implemented. Please use BACKEND_FIONA or BACKEND_CPU."

// Placeholder: These will be replaced with actual Gemmini calls
// #include "gemmini.h"

// Configuration for Gemmini (typical values, adjust based on actual config)
#define GEMMINI_DIM 16          // Systolic array dimension
#define GEMMINI_ACC_SCALE 1     // Accumulator scale

// Placeholder macros - to be implemented
#define SET_VLEN(len) do { /* gemmini_set_config(...) */ } while(0)
#define LOAD_V(vregnum, src) do { /* gemmini_mvin(...) */ } while(0)
#define STORE_V(vregnum, dst) do { /* gemmini_mvout(...) */ } while(0)
#define ADD_V(vd, v1, v2) do { /* gemmini element-wise add */ } while(0)
#define SUB_V(vd, v1, v2) do { /* gemmini element-wise sub */ } while(0)
#define DOTP(rd, v1, v2) do { /* gemmini dot product */ } while(0)
#define MVM(vd, v1) do { /* gemmini_matmul(...) */ } while(0)
#define RELU_V(vd, v1) do { /* gemmini with ReLU activation */ } while(0)

#endif /* BACKEND_GEMMINI */

#endif /* FIONA_BACKEND_GEMMINI_OPS_H */
