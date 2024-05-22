#pragma once

#define HIP_CHECK(stat)                                                        \
    {                                                                          \
        if(stat != hipSuccess)                                                 \
        {                                                                      \
            std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
        }                                                                      \
    }

// Fill array with data on host (demonstrate host nodes)
// Copy data to device array
// Scale this array on device
// Find maximum value in array on device

__launch_bounds__(256) __global__ void scale(int m, float* data, float value)
{
    int gid = threadIdx.x + 256 * blockIdx.x;

    if (gid < m) data[gid] *= value;
}

__launch_bounds__(256) __global__ void add(int m, float* c, const float* a, const float* b)
{
    int gid = threadIdx.x + 256 * blockIdx.x;

    if (gid < m) c[gid] = a[gid] + b[gid];
}


__launch_bounds__(256) __global__ void subtract(int m, float* c, const float* a, const float* b)
{
    int gid = threadIdx.x + 256 * blockIdx.x;

    if (gid < m) c[gid] = a[gid] - b[gid];
}
