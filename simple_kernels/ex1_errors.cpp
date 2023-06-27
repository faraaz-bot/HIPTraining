#include <hip/hip_runtime.h>
#include <iostream>

__global__ void ksegfault(float* a)
{
    // Solution:
    {
        *a = 1.0;
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
        size_t toomuch = (size_t)1 << 31;

        hipMalloc(&d_p, toomuch);
        if(hipPeekAtLastError() != hipSuccess)
        {
            std::cout << "\tWell, that didn't work.\n";
        }
        if(hipGetLastError() != hipSuccess)
        {
            std::cout << "\tStill not good.\n";
        }
        if(hipGetLastError() == hipSuccess)
        {
            std::cout << "\tLooks like everything is fine now!\n";
        }
        std::cout << std::endl;
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
        // We don't allocate the memory so that we can observe the error
        // behaviour:
        //assert(hipMalloc(&d_a, sizeof(float)) == hipSuccess);
    
        // NB: this actually aborts before we get to the return-code
        // logic.
        ksegfault<<<dim3(1), dim3(1)>>>(d_a);
        hipDeviceSynchronize();
        if(hipGetLastError() != hipSuccess)
        {
            std::cout << "\tError running ksegfault!\n";
        }
        else
        {
            std::cout << "\tksegfault is totally fine!\n";
        }
        
        // Release device memory
        if(d_a != nullptr)
        {
            assert(hipFree(d_a) == hipSuccess);
        }
    }
    
    return 0;
}
