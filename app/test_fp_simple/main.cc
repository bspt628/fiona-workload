// Simple floating point test for bare-metal
#include <stdio.h>
#include <math.h>

int main() {
    printf("FP Test Start\n");

    // Basic floating point operations
    float a = 3.14159f;
    float b = 2.71828f;
    float c = a + b;
    printf("%.2f + %.2f = %.2f\n", a, b, c);

    // sin/cos test
    float angle = 1.0f;
    float s = sinf(angle);
    float cs = cosf(angle);
    printf("sin(1) = %.4f, cos(1) = %.4f\n", s, cs);

    // pow test
    float p = powf(2.0f, 10.0f);
    printf("2^10 = %.1f\n", p);

    printf("FP Test Complete\n");
    return 0;
}
