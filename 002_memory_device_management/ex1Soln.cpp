#include <hip/hip_runtime.h>
#include <iostream>

/*
    Compile:
    hipcc helloworld.cpp -o helloworld

    Run:
    ./helloworld
*/

__global__ void gpuHelloWorld()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello world from thread %d \n",tid);
}

void cpuHelloWorld()
{
    std::cout << "Hello world from the CPU!" << std::endl;
}

int main()
{
    cpuHelloWorld();
    gpuHelloWorld <<<1,16>>>();
    return 0;
}