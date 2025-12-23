#ifndef FIONA_BACKEND_H
#define FIONA_BACKEND_H

/**
 * Backend Abstraction Layer for FIONA Workloads
 *
 * Supported backends:
 *   - BACKEND_FIONA   : FIONA photonic accelerator (default)
 *   - BACKEND_CPU     : Pure CPU baseline (no accelerator)
 *   - BACKEND_GEMMINI : Gemmini systolic array (future)
 *
 * Usage:
 *   make BACKEND=cpu      # Build with CPU baseline
 *   make BACKEND=fiona    # Build with FIONA accelerator (default)
 *   make BACKEND=gemmini  # Build with Gemmini (future)
 */

//=============================================================================
// Backend Selection
//=============================================================================

// Default to FIONA if no backend specified
#if !defined(BACKEND_FIONA) && !defined(BACKEND_CPU) && !defined(BACKEND_GEMMINI)
#define BACKEND_FIONA
#endif

// Validate only one backend is selected
#if (defined(BACKEND_FIONA) + defined(BACKEND_CPU) + defined(BACKEND_GEMMINI)) > 1
#error "Only one backend can be selected at a time"
#endif

//=============================================================================
// Backend Information
//=============================================================================

#if defined(BACKEND_FIONA)
#define BACKEND_NAME "FIONA"
#define BACKEND_DESC "FIONA Photonic Accelerator"
#elif defined(BACKEND_CPU)
#define BACKEND_NAME "CPU"
#define BACKEND_DESC "CPU Baseline (Pure Software)"
#elif defined(BACKEND_GEMMINI)
#define BACKEND_NAME "Gemmini"
#define BACKEND_DESC "Gemmini Systolic Array Accelerator"
#endif

//=============================================================================
// Backend-specific includes
//=============================================================================

#include "base/config.h"

#if defined(BACKEND_FIONA)
// FIONA backend uses RoCC instructions
#include "base/instr.h"
#elif defined(BACKEND_CPU)
// CPU backend uses pure software implementations
#include "backend/cpu/ops.h"
#elif defined(BACKEND_GEMMINI)
// Gemmini backend (future)
#include "backend/gemmini/ops.h"
#endif

//=============================================================================
// Backend initialization (optional)
//=============================================================================

static inline void backend_init(void) {
#if defined(BACKEND_FIONA)
    // FIONA: no special initialization needed
#elif defined(BACKEND_CPU)
    // CPU: no special initialization needed
#elif defined(BACKEND_GEMMINI)
    // Gemmini: initialize systolic array
    // gemmini_init();
#endif
}

static inline void backend_print_info(void) {
    printf("========================================\n");
    printf("Backend: %s\n", BACKEND_NAME);
    printf("Description: %s\n", BACKEND_DESC);
    printf("========================================\n");
}

#endif /* FIONA_BACKEND_H */
