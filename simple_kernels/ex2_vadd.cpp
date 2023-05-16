#include <hip/hip_runtime.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

__global__ void vecAdd(float* a, const float* b)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] += b[idx];
}

// Fill the array with some values
void fillArray(std::vector<float>& v)
{
    for(int i = 0; i < v.size(); ++i)
    {
        v[i] = i; //sin(i);
    }
}

int main()
{
    std::cout << "HIP vector addition example\n";

    const int N = 16;

    std::vector<float> vala(N);
    fillArray(vala);

    std::vector<float> valb(N);
    fillArray(valb);
    const size_t valbytes = vala.size() * sizeof(decltype(vala)::value_type);

    hipError_t rt = hipSuccess;

    float* d_a = nullptr;
    assert(hipMalloc(&d_a, valbytes) == hipSuccess);
    assert(hipMemcpy(d_a, vala.data(), valbytes, hipMemcpyHostToDevice) == hipSuccess);

    float* d_b = nullptr;
    assert(hipMalloc(&d_b, valbytes) == hipSuccess);
    assert(hipMemcpy(d_b, valb.data(), valbytes, hipMemcpyHostToDevice) == hipSuccess);

    int blockSize = N;
    int blocks    = 1;
    vecAdd<<<dim3(blocks), dim3(blockSize)>>>(d_a, d_b);
    rt = hipGetLastError();
    assert(rt == hipSuccess);

    assert(hipMemcpy(vala.data(), d_a, valbytes, hipMemcpyDeviceToHost) == hipSuccess);

    for(const auto& val: vala)
        std::cout << val << " ";
    std::cout << "\n";

    // Release device memory
    assert(hipFree(d_a) == hipSuccess);
    assert(hipFree(d_b) == hipSuccess);

    return 0;
}
