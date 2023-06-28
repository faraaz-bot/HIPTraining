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
    // Solution
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        for(int i = 0; i < batch; ++i)
        {
            const int pos = idx * batch + i;
            if(pos < N)
            {
                a[pos] += b[pos];
            }
        }
    }
}

int main()
{
    std::cout << "HIP vector addition example\n";

    const int N     = 16;
    const int batch = 4;
    std::cout << "N: " << N << "\n";
    std::cout << "batch: " << batch << "\n";
    if(N % batch != 0)
    {
        throw std::runtime_error("N must be divisible by batch");
    }

    std::vector<float> vala(N);
    for(int i = 0; i < vala.size(); ++i)
    {
        vala[i] = i; // or whatever you want to fill it with
    }

    std::vector<float> valb(N);
    for(int i = 0; i < valb.size(); ++i)
    {
        valb[i] = i; // or whatever you want to fill it with
    }

    // Solution
    {
        const size_t valbytes = vala.size() * sizeof(decltype(vala)::value_type);

        float* d_a = nullptr;
        if(hipMalloc(&d_a, valbytes) != hipSuccess)
        {
            throw std::runtime_error("hipMalloc failed");
        }
        if(hipMemcpy(d_a, vala.data(), valbytes, hipMemcpyHostToDevice) != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed");
        }
        
        float* d_b = nullptr;
        if(hipMalloc(&d_b, valbytes) != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed");
        }
        if(hipMemcpy(d_b, valb.data(), valbytes, hipMemcpyHostToDevice) != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed");
        }

        const int blockSize = 32;
        const int blocks    = ceildiv(N / 4, blockSize);
        std::cout << "blockSize: " << blockSize << "\n";
        std::cout << "blocks: " << blocks << "\n";

        vecAddBlock<<<dim3(blocks), dim3(blockSize)>>> (d_a, d_b, N, batch);
        if(hipGetLastError() != hipSuccess)
        {
            throw std::runtime_error("kernel execution failed");
        }

        if(hipMemcpy(vala.data(), d_a, valbytes, hipMemcpyDeviceToHost) != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed");
        }


        // Release device memory
        if(hipFree(d_a) != hipSuccess)
        {
            throw std::runtime_error("hipFree failed");
        }
        if(hipFree(d_b) != hipSuccess)
        {
            throw std::runtime_error("hipFree failed");
        }
    
    }
    
    for(const auto& val: vala)
        std::cout << val << " ";
    std::cout << "\n";

    
    return 0;
}
