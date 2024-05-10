#include <hip/hip_runtime.h>

#include <stdexcept>
#include <stdio.h>
#include <vector>

#include "transpose_kernel.h"

// Launch the kernel.
//
// Modify this function to compile, load, and launch the transpose
// kernel via hipRTC.
void do_transpose(float* input_d, float* output_d, size_t N1, size_t N2)
{
    dim3 gridDim{1};
    dim3 blockDim{64, 16};
    transpose_kernel<<<gridDim, blockDim>>>(input_d, output_d, N1, N2);
}

int main()
{
    // array of length N, which is N1 * N2;
    static const size_t LEN_N1 = 11;
    static const size_t LEN_N2 = 20;
    static const size_t LEN_N  = LEN_N1 * LEN_N2;

    // host array
    std::vector<float> input_h(LEN_N);
    for(size_t i = 0; i < LEN_N; ++i)
        input_h[i] = i;

    // print the array
    puts("input:");
    for(size_t colIdx = 0; colIdx < LEN_N2; ++colIdx)
    {
        for(size_t rowIdx = 0; rowIdx < LEN_N1; ++rowIdx)
        {
            printf("%.1f ", static_cast<double>(input_h[rowIdx + colIdx * LEN_N1]));
        }
        printf("\n");
    }

    // copy to device
    float* input_d = nullptr;
    if(hipMalloc(&input_d, sizeof(float) * LEN_N) != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    if(hipMemcpy(input_d, input_h.data(), sizeof(float) * LEN_N, hipMemcpyHostToDevice)
       != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    // allocate output
    float* output_d = nullptr;
    if(hipMalloc(&output_d, sizeof(float) * LEN_N) != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    // transpose the data
    do_transpose(input_d, output_d, LEN_N1, LEN_N2);

    // copy output back to host
    std::vector<float> output_h(LEN_N);
    if(hipMemcpy(output_h.data(), output_d, sizeof(float) * LEN_N, hipMemcpyDeviceToHost)
       != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    // print output
    puts("\noutput:");
    for(size_t colIdx = 0; colIdx < LEN_N1; ++colIdx)
    {
        for(size_t rowIdx = 0; rowIdx < LEN_N2; ++rowIdx)
        {
            printf("%.1f ", static_cast<double>(output_h[rowIdx + colIdx * LEN_N2]));
        }
        printf("\n");
    }

    (void)hipFree(input_d);
    (void)hipFree(output_d);

    return 0;
}
