#include <hip/hip_runtime.h>
#include <iostream>

__global__ void kfail(float* a)
{
    // Solution:
    {
        // Put your solution here.
    }
}

int main()
{
    std::cout << "HIP error handling example\n";
    std::cout << std::endl;

    
    std::cout << "Let's try allocating too much memory:\n";

    // Solution:
    {
        // Put your solution here.
    }


    std::cout << "Ok, let's launch a kernel which shouldn't succeed:\n";

    // Solution:
    {
        // Put your solution here.
    }
    
    return 0;
}
