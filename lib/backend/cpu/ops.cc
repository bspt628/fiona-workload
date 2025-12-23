/**
 * CPU Backend - Global State
 *
 * Simulated vector register file for CPU baseline.
 */

#include "base/config.h"

#ifdef BACKEND_CPU

// Must match the definition in ops.h
#ifndef CPU_NUM_VREGS
#define CPU_NUM_VREGS 32
#endif

// Simulated vector register file
elem_t cpu_vregs[CPU_NUM_VREGS][EU_VEC_ELEM];

// Current vector length
size_t cpu_vlen = EU_VEC_ELEM;

#endif /* BACKEND_CPU */
