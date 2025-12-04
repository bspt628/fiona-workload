#include "fiona.h"
#include "utils/pprint.h"

int main() {
    printf("------------- FIONA ReLU Test -------------\n");
    
    // Test input: mix of positive and negative values
    static const elem_t test_input[10] = {-90, -132, 220, -110, -20, 212, 228, -6, 2, -14};
    elem_t test_output[10];
    
    // Expected output: [0, 0, 220, 0, 0, 212, 228, 0, 2, 0]
    static const elem_t expected_output[10] = {0, 0, 220, 0, 0, 212, 228, 0, 2, 0};
    
    printf("[test] Input:    ");
    print_vec(test_input, 10);
    
    // Apply ReLU using FIONA instruction
    SET_VLEN(10);
    LOAD_V(1, test_input);
    RELU_V(2, 1);
    STORE_V(2, test_output);
    
    printf("[test] Output:   ");
    print_vec(test_output, 10);
    
    printf("[test] Expected: ");
    print_vec(expected_output, 10);
    
    // Check if output matches expected
    elem_t all_correct = 1;
    for(size_t i = 0; i < 10; ++i) {
        if(test_output[i] != expected_output[i]) {
            all_correct = 0;
            printf("[ERROR] Mismatch at index %d: got %d, expected %d\n", i, test_output[i], expected_output[i]);
        }
    }
    
    if(all_correct) {
        printf("[SUCCESS] ReLU test passed!\n");
    } else {
        printf("[FAILED] ReLU test failed!\n");
    }
    
    return 0;
}
