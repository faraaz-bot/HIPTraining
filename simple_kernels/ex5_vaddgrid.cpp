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
        // Put your solution here.
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
        // Put your solution here.
    }

    for(const auto& val: vala)
        std::cout << val << " ";
    std::cout << "\n";
    
    return 0;
}
