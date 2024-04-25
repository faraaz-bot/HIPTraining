#include <hip/hip_runtime.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sstream>

// Computes ceil(numerator/divisor) for integer types.
template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
intT1 ceildiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}

__global__ void vecAdd(float* a, const float* b, const int N)
{
    // Solution
    {
        // Put your solution here.
    }
}

int main()
{
    std::cout << "HIP vector addition example\n";

    const int N = 16;

    std::cout << "N: " << N << "\n";
    
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

    std::vector<float> valout(N, 0.0);
    
    // Solution:
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
