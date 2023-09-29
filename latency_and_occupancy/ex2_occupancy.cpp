#include<iostream>

#include <hip/hip_runtime.h>

__global__ void vecAdd(float* a, const float* b, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        a[idx] += b[idx];
    }
}

int main()
{

    {
        size_t lds_bytes = 1 << 12;
        dim3 blockDim(N, 1, 1);
        int max_blocks_per_sm = 0;
        
        const auto ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, vecAdd, blockDim.x * blockDim.y * blockDim.z, lds_bytes);
        if(ret != hipSuccess)  {
            throw std::runtime_error("kernel execution failed");
        }
        std::cout << "occupancy: " << max_blocks_per_sm << "\n";
    }

    // TODO: add timing and launch stuff.
    // TODO: pass stuff via command-line?
    
    return 0;

    
}
