#include <hip/hip_runtime.h>
#include <iostream>

__global__ void ksegfault(float* a)
{
    a[-1] = 1.0;
}

int main()
{
    std::cout << "HIP error handling example\n";
    std::cout << std::endl;

    std::cout << "Let's try allocating too much memory:\n";
    size_t toomuch = 1 << 31;
    void*  d_p     = NULL;
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

    std::cout << "Ok, let's launch a kernel which goes out-of-bounds:\n";

    float* d_a;
    assert(hipMalloc(&d_a, sizeof(float)) == hipSuccess);

    // NB: this actually aborts before we get to the return-code
    // logic.
    hipError_t rt = hipSuccess;
    ksegfault<<<dim3(1), dim3(1)>>>(d_a);
    hipDeviceSynchronize();
    rt = hipGetLastError();
    assert(rt == hipSuccess);

    // Release device memory
    assert(hipFree(d_a) == hipSuccess);

    return 0;
}
