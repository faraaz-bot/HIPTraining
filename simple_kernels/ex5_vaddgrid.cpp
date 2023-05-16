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

__global__ void vecAddBlock(float* a, const float* b, const int N, const int batch)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < batch; ++i)
    {
        const int pos = i * blockDim.x * gridDim.x + idx;
        if(pos < N)
        {
            a[pos] += b[pos];
        }
    }
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

    const int N     = 16;
    const int batch = 4;
    std::cout << "N: " << N << "\n";
    std::cout << "batch: " << batch << "\n";
    assert(N % batch == 0);

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

    const int blockSize = 32;
    const int blocks    = ceildiv(N / 4, blockSize);
    std::cout << "blockSize: " << blockSize << "\n";
    std::cout << "blocks: " << blocks << "\n";
    vecAddBlock<<<dim3(blocks), dim3(blockSize)>>> (d_a, d_b, N, batch);

    assert(hipMemcpy(vala.data(), d_a, valbytes, hipMemcpyDeviceToHost) == hipSuccess);

    for(const auto& val: vala)
        std::cout << val << " ";
    std::cout << "\n";
    
    // Release device memory
    assert(hipFree(d_a) == hipSuccess);
    assert(hipFree(d_b) == hipSuccess);

    return 0;
}
