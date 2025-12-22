// Test BSS size threshold for bare-metal
#include <stdio.h>

// Configurable BSS size via compile-time define
#ifndef BSS_KB
#define BSS_KB 10
#endif

// Create array of specified size
static float test_array[BSS_KB * 256];  // 256 floats = 1KB

int main() {
    printf("BSS Size Test: %d KB\n", BSS_KB);

    // Touch array to prevent optimization
    test_array[0] = 1.0f;
    test_array[BSS_KB * 256 - 1] = 2.0f;

    printf("  array[0] = %d\n", (int)(test_array[0] * 1000));
    printf("  array[last] = %d\n", (int)(test_array[BSS_KB * 256 - 1] * 1000));

    printf("BSS Size Test Complete!\n");
    return 0;
}
