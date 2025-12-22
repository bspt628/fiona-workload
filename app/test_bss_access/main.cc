// Test float access for bare-metal
#include <stdio.h>
#include <math.h>

// Force the variable to not be optimized away
float test_float_result;

int main() {
    printf("Float Test Start\n");

    // Call math function first
    float x = sinf(1.0f);
    printf("sinf called\n");

    // Stack float (non-volatile)
    float test_float = 3.14f;
    test_float_result = test_float;  // Use it to prevent optimization
    printf("  test_float set\n");

    printf("Float Test Complete!\n");
    return 0;
}
