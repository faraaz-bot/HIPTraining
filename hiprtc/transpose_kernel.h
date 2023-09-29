#include <hip/hip_runtime.h>

template <typename Tfloat>
__global__ __launch_bounds__(1024) void transpose_kernel(const Tfloat* __restrict__ input,
                                                         Tfloat* __restrict__ output,
                                                         unsigned int length0,
                                                         unsigned int length1)
{
    __shared__ Tfloat lds[64][64];
    unsigned int      tileBlockIdx_y = blockIdx.y;
    unsigned int      tileBlockIdx_x = blockIdx.x;
    unsigned int      tile_x_index   = threadIdx.x;
    unsigned int      tile_y_index   = threadIdx.y;

#pragma unroll
    for(unsigned int i = 0; i < 4; ++i)
    {
        auto logical_row = 64 * tileBlockIdx_y + tile_y_index + i * 16;
        auto idx0        = 64 * tileBlockIdx_x + tile_x_index;
        auto idx1        = logical_row;
        if(idx0 >= length0 || idx1 >= length1)
        {
            break;
        }

        auto   global_read_idx = idx0 + idx1 * length0;
        Tfloat elem;
        elem                                     = input[global_read_idx];
        lds[tile_x_index][i * 16 + tile_y_index] = elem;
    }
    __syncthreads();
    Tfloat val[4];
    // reallocate threads to write along fastest dim (length1) and
    // read transposed from LDS
    tile_x_index = threadIdx.y;
    tile_y_index = threadIdx.x;
#pragma unroll
    for(unsigned int i = 0; i < 4; ++i)
    {
        val[i] = lds[tile_x_index + i * 16][tile_y_index];
    }
#pragma unroll
    for(unsigned int i = 0; i < 4; ++i)
    {
        auto logical_col = 64 * tileBlockIdx_x + tile_x_index + i * 16;
        auto logical_row = 64 * tileBlockIdx_y + tile_y_index;
        auto idx0        = logical_col;
        auto idx1        = logical_row;
        if(idx0 >= length0 || idx1 >= length1)
        {
            break;
        }

        auto global_write_idx    = idx0 * length1 + idx1;
        output[global_write_idx] = val[i];
    }
}
