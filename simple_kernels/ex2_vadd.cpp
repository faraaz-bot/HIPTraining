#include <hip/hip_runtime.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

__global__ void vecAdd(float* a, const float* b)
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
    
    // Solution:
    {
        // Put your solution here.
    }
    
    for(const auto& val: vala)
        std::cout << val << " ";
    std::cout << "\n";


    return 0;
}
