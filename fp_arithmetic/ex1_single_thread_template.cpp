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

__global__ void series_forward(/*params*/)
{
    // TODO: Using a single thread, calculate the series for n elements
    // sum(1/(x^2)) for x = 1...n
}

__global__ void series_backward(/*params*/)
{
    // TODO: For #3, calculate the summation in reverse order
    // sum(1/(n-x)^2) for x = 0...n-1
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " num_elements\n";
        return -1;
    }

    int n = atoi(argv[1]);
    if(n <= 0)
    {
        std::cout << "Must have >= 1 element(s).\n";
        return -2;
    }

    std::cout << "Computing series using a single thread and " << n << " elements.\n";

    // Double precision reference value
    double reference = M_PI * M_PI / 6.0;

    // TODO: Allocate device memory

    int threads = 1;
    int blocks = 1;

    // TODO: Launch kernel, copy result back to host for comparison, calculate relative error, cleanup.

    return 0;
}
