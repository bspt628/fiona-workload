/**
 * Performance Measurement Infrastructure - Global State
 */

#include "utils/perf.h"

// Global performance state
perf_state_t perf_state = {
    .num_regions = 0,
    .current_region = -1,
    .global_start = 0,
    .global_end = 0
};
