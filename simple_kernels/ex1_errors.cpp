#include <hip/hip_runtime.h>
#include <iostream>

__global__ void ksegfault(float* a)
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

    
    void* d_p  = nullptr;
    // Exercise: allocate an unreasonable amount of memory (like 2^31
    // bytes) in d_p and detect the error code.  Use peek and get last
    // error to detect error codes.  Determine the behaviour when
    // hipGetLastError is called more than once.

    // Solution:
    {
        // Put your solution here.
    }
    
    // Release device memory
    if(d_p != nullptr)
    {
        if(hipFree(d_p) != hipSuccess)
            std::cout << "error freeing d_p\n";
    }

    std::cout << "Ok, let's launch a kernel which shouldn't succeed:\n";

    // The easiest way to get a failed kernel is to try and store a
    // value at memory address 0 (ie the nullptr).
    
    float* d_a = nullptr;

    // Solution:
    {
        // Put your solution here.
    }
    
    return 0;
}
