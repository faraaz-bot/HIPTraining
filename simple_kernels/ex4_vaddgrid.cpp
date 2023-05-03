#include <hip/hip_runtime.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// Computes ceil(numerator/divisor) for integer types.
template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
intT1 ceildiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}

__global__ void vecAdd(float* a, const float* b)
{
    for(int bidx = 0; bidx < blockDim.x; ++bidx)
    {
        const int idx = bidx * blockDim.x + threadIdx.x;
        a[idx] += b[idx];
    }
}

// Print the array to stdout
void printVec(const std::vector<float> v)
{
    for(int i = 0; i < v.size(); ++i)
    {
        std::cout << v[i] << " ";
    }
    std::cout << "\n";
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

    const int N         = 256;
    const int blockSize = 256;
    assert(N % blockSize == 0);

    std::vector<float> vala(N);
    fillArray(vala);

    std::vector<float> valb(N);
    fillArray(valb);
    const size_t valbytes = vala.size() * sizeof(decltype(vala)::value_type);

    float* d_a;
    assert(hipMalloc(&d_a, valbytes) == hipSuccess);
    assert(hipMemcpy(d_a, vala.data(), valbytes, hipMemcpyHostToDevice) == hipSuccess);

    float* d_b;
    assert(hipMalloc(&d_b, valbytes) == hipSuccess);
    assert(hipMemcpy(d_b, valb.data(), valbytes, hipMemcpyHostToDevice) == hipSuccess);

    const int blocks = ceildiv(N, blockSize);
    vecAdd<<<dim3(blocks), dim3(blockSize)>>>(d_a, d_b);

    assert(hipMemcpy(vala.data(), d_a, valbytes, hipMemcpyDeviceToHost) == hipSuccess);

    printVec(vala);

    // Release device memory
    assert(hipFree(d_a) == hipSuccess);
    assert(hipFree(d_b) == hipSuccess);

    return 0;
}
