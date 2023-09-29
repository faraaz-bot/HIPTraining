/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IaS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd)                                                                  \
    do {                                                                                \
        hipError_t error = (cmd);                                                       \
        if (error != hipSuccess)                                                        \
        {                                                                               \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error)          \
                      << ") at line " << __LINE__ << " in file " << __FILE__ << "\n";   \
            exit(-1);                                                                   \
        }                                                                               \
    } while (0)


/**
 * @brief Sums hipBlockDim_x elements of res and store into res[0]
 * 
 * @param res 
 */
__device__ void series_reduce(float* res)
{
    int64_t idx = hipThreadIdx_x;
    int64_t cur_n = hipBlockDim_x;

    while(cur_n > 1)
    {
        float tmp;
        __syncthreads();
        int64_t i = 2 * idx;
        if(i < cur_n)
        {
            tmp = res[i];
            if(i + 1 < cur_n)
                tmp += res[i + 1];
        }
        __syncthreads();

        res[idx] = tmp;
        cur_n = (cur_n + 1) / 2;
    }
}

/**
 * @brief Calculates the first "n" elements of the series sum(1/(x^2)) from x=1...n
 * 
 * @param res an array of device memory with hipBlockDim_x elements
 * @param n the amount of elements of the sum to calculate
 */
__global__ void series_forward(float* res, int n)
{
    const int idx = hipThreadIdx_x;

    // # items in series for each thread
    const int64_t items = (n + hipBlockDim_x - 1) / hipBlockDim_x;

    const int64_t start_range = items * idx + 1;
    const int64_t end_range = items * (idx + 1) + 1;

    float tmp = 0;
    for(int64_t i = start_range; i < end_range; i++)
    {
        if(i <= n && i >= 1)
            tmp += 1.0 / (i * i);
    }

    res[idx] = tmp;

    // res now contains partial sums, reduce these into a single sum
    series_reduce(res);
}

/**
 * @brief Calculates the first "n" elements of the series sum(1/((n-x)^2)) from x=0...n-1
 * 
 * @param res an array of device memory with hipBlockDim_x elements
 * @param n the amount of elements of the sum to calculate
 */
__global__ void series_backward(float* res, int n)
{
    const int idx = hipThreadIdx_x;

    // # items in series for each thread
    const int64_t items = (n + hipBlockDim_x - 1) / hipBlockDim_x;

    const int64_t start_range = items * idx + 1;
    const int64_t end_range = items * (idx + 1) + 1;

    float tmp = 0;
    for(int64_t i = start_range; i < end_range; i++)
    {
        int64_t x = n - i;
        if(x <= n && x >= 1)
            tmp += 1.0 / (x * x);
    }

    res[idx] = tmp;

    // res now contains partial sums, reduce these into a single sum
    series_reduce(res);
}

int main(int argc, char** argv)
{
    if(argc != 3 && argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " num_elements " << " num_threads " << "[optional] csv_output (y,n)\n";
        return 0;
    }

    int n = atoi(argv[1]);
    int threads = atoi(argv[2]);
    if(n <= 0)
    {
        std::cout << "Must have >= 1 element(s).\n";
        return 0;
    }

    if(threads <= 0 || threads > n)
    {
        std::cout << "Must have 1 <= threads <= n\n";
        return 0;
    }

    // Adding some code to just output in csv format for easier parsing in a script
    bool csv_output = false;
    if(argc == 4 && (argv[3][0] == 'y' || argv[3][0] == 'Y'))
        csv_output = true;

    if(!csv_output)
        std::cout << "Computing series using a single thread and " << n << " elements.\n";

    // Double precision reference value
    double reference = M_PI * M_PI / 6.0;

    float* d_res;
    float h_res;

    HIP_CHECK(hipMalloc(&d_res, threads * sizeof(float)));

    int blocks = 1;

    hipLaunchKernelGGL(series_forward, dim3(blocks), dim3(threads), 0, 0,
                       d_res, n);

    // hipLaunchKernelGGL(series_backward, dim3(blocks), dim3(threads), 0, 0,
    //                    d_res, n);

    HIP_CHECK(hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost));

    double rel_error = std::abs((reference - h_res) / reference);
    std::cout << std::fixed << std::setprecision(15);

    if(!csv_output)
    {
        std::cout << "Double precision reference: " << reference << "\n";
        std::cout << "Single precision result: " << h_res << "\n";
        std::cout << "Relative error: " << rel_error << "\n";
    }
    else
    {
        std::cout << "threads,elements,f64_reference,f32_result,relative_error\n";
        std::cout << threads << "," << n << "," << reference << "," << h_res << "," << rel_error << "\n";
    }

    HIP_CHECK(hipFree(d_res));

    return 0;
}
