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
        // Put your solution here.
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

    std::vector<float> valout(vala.size());
    
    // Solution
    {
        // Put your solution here.
    }

    float maxerr = 0.0;
    for(int i = 0; i < valout.size(); ++i) {
        float diff = std::abs(vala[i] + valb[i] - valout[i]);
        if(diff > maxerr)
            maxerr = diff;
    }
    std::cout << "max error: " << maxerr << "\n";

    
    return 0;
}
