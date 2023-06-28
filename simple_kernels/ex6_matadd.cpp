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

// Kernel for adding one 2D array to another
__global__ void matAdd(float* a, const float* b, const int Nx, const int Ny)
{
    // Solution
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;
        if(idx < Nx && idy < Ny)
        {
            const int pos = idx + Nx * idy;
            a[pos] += b[pos];
        }
    }
}

// Helper functions for filling and showing matrices
void fillMatrix(std::vector<float>& mat, const int m, const int n)
{
    assert(mat.size() == n * m);
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < m; ++j)
        {
            const int idx = i * m + j;
            mat[idx]      = i + j;
        }
    }
}
void showMatrix(const std::vector<float> mat, const int m, const int n)
{
    assert(mat.size() == n * m);
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < m; ++j)
        {
            const int idx = i * m + j;
            std::cout << mat[idx] << " ";
        }
        std::cout << "\n";
    }
}

int main()
{
    std::cout << "HIP vector addition example\n";

    // Problem dimensions:
    const int N = 5;
    const int M = 4;

    std::cout << "input a:\n";
    std::vector<float> vala(M * N);
    fillMatrix(vala, N, M);
    showMatrix(vala, N, M);

    std::cout << "input b:\n";
    std::vector<float> valb(M * N);
    fillMatrix(valb, N, M);
    showMatrix(valb, N, M);

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

        matAdd<<<dim3(32, 32), dim3(ceildiv(M, 32), ceildiv(N, 32))>>>(d_a, d_b, N, M);
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

    std::cout << "output:\n";
    showMatrix(vala, N, M);
    
    return 0;
}
