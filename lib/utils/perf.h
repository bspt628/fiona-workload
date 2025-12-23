#ifndef FIONA_UTILS_PERF_H
#define FIONA_UTILS_PERF_H

/**
 * Performance Measurement Infrastructure
 *
 * Provides cycle counting and performance metrics for comparing
 * different backends (FIONA, CPU, Gemmini).
 *
 * Usage:
 *   PERF_INIT();
 *   PERF_START("matmul");
 *   // ... computation ...
 *   PERF_END();
 *   PERF_REPORT();
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "backend/backend.h"

//=============================================================================
// Cycle Counter Access (RISC-V CSR)
//=============================================================================

static inline uint64_t rdcycle(void) {
#if defined(__riscv)
    uint64_t cycle;
    asm volatile ("rdcycle %0" : "=r"(cycle));
    return cycle;
#else
    // Fallback for non-RISC-V (e.g., host compilation)
    return 0;
#endif
}

static inline uint64_t rdinstret(void) {
#if defined(__riscv)
    uint64_t instret;
    asm volatile ("rdinstret %0" : "=r"(instret));
    return instret;
#else
    return 0;
#endif
}

//=============================================================================
// Performance Measurement State
//=============================================================================

#define PERF_MAX_REGIONS 32
#define PERF_NAME_LEN 32

typedef struct {
    char name[PERF_NAME_LEN];
    uint64_t start_cycle;
    uint64_t end_cycle;
    uint64_t start_instret;
    uint64_t end_instret;
    uint64_t total_cycles;
    uint64_t total_instret;
    uint32_t call_count;
} perf_region_t;

typedef struct {
    perf_region_t regions[PERF_MAX_REGIONS];
    int num_regions;
    int current_region;
    uint64_t global_start;
    uint64_t global_end;
} perf_state_t;

// Global performance state
extern perf_state_t perf_state;

//=============================================================================
// Performance Measurement Macros
//=============================================================================

#define PERF_INIT() do { \
    perf_state.num_regions = 0; \
    perf_state.current_region = -1; \
    perf_state.global_start = rdcycle(); \
    printf("========================================\n"); \
    printf("Performance Measurement Initialized\n"); \
    printf("Backend: %s\n", BACKEND_NAME); \
    printf("========================================\n"); \
} while(0)

#define PERF_START(name) do { \
    int _idx = -1; \
    for (int _i = 0; _i < perf_state.num_regions; _i++) { \
        if (strcmp(perf_state.regions[_i].name, name) == 0) { \
            _idx = _i; \
            break; \
        } \
    } \
    if (_idx < 0 && perf_state.num_regions < PERF_MAX_REGIONS) { \
        _idx = perf_state.num_regions++; \
        strncpy(perf_state.regions[_idx].name, name, PERF_NAME_LEN-1); \
        perf_state.regions[_idx].total_cycles = 0; \
        perf_state.regions[_idx].total_instret = 0; \
        perf_state.regions[_idx].call_count = 0; \
    } \
    if (_idx >= 0) { \
        perf_state.current_region = _idx; \
        perf_state.regions[_idx].start_cycle = rdcycle(); \
        perf_state.regions[_idx].start_instret = rdinstret(); \
    } \
} while(0)

#define PERF_END() do { \
    int _idx = perf_state.current_region; \
    if (_idx >= 0) { \
        perf_state.regions[_idx].end_cycle = rdcycle(); \
        perf_state.regions[_idx].end_instret = rdinstret(); \
        perf_state.regions[_idx].total_cycles += \
            perf_state.regions[_idx].end_cycle - perf_state.regions[_idx].start_cycle; \
        perf_state.regions[_idx].total_instret += \
            perf_state.regions[_idx].end_instret - perf_state.regions[_idx].start_instret; \
        perf_state.regions[_idx].call_count++; \
        perf_state.current_region = -1; \
    } \
} while(0)

#define PERF_REPORT() do { \
    perf_state.global_end = rdcycle(); \
    uint64_t _total = perf_state.global_end - perf_state.global_start; \
    printf("\n"); \
    printf("========================================\n"); \
    printf("Performance Report [%s]\n", BACKEND_NAME); \
    printf("========================================\n"); \
    printf("%-20s %12s %12s %8s %10s\n", "Region", "Cycles", "Instrs", "Calls", "IPC"); \
    printf("------------------------------------------------------------\n"); \
    for (int _i = 0; _i < perf_state.num_regions; _i++) { \
        perf_region_t *_r = &perf_state.regions[_i]; \
        double _ipc = (_r->total_cycles > 0) ? \
            (double)_r->total_instret / _r->total_cycles : 0.0; \
        printf("%-20s %12lu %12lu %8u %10.2f\n", \
            _r->name, _r->total_cycles, _r->total_instret, _r->call_count, _ipc); \
    } \
    printf("------------------------------------------------------------\n"); \
    printf("%-20s %12lu\n", "TOTAL", _total); \
    printf("========================================\n"); \
} while(0)

// Simple one-shot timing macro
#define PERF_MEASURE(name, code) do { \
    PERF_START(name); \
    { code; } \
    PERF_END(); \
} while(0)

//=============================================================================
// Comparison Helper
//=============================================================================

// For printing comparison results
#define PERF_COMPARE_HEADER() do { \
    printf("\n"); \
    printf("========================================\n"); \
    printf("Backend Comparison\n"); \
    printf("========================================\n"); \
    printf("Run the same workload with different backends:\n"); \
    printf("  1. make BACKEND=fiona <app>\n"); \
    printf("  2. make BACKEND=cpu <app>\n"); \
    printf("  3. Compare the TOTAL cycles from each run\n"); \
    printf("========================================\n"); \
} while(0)

#endif /* FIONA_UTILS_PERF_H */
