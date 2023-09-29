#include<iostream>
#include<vector>

#include <hip/hip_runtime.h>


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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        a[idx] += b[idx];
    }
}

int main()
{

    const int N = 1<<15;

    {
        size_t lds_bytes = 0;//1 << 2;

        int blockSize = 512;
        int gridSize = ceildiv(N, blockSize);
        
        dim3 gridDim(gridSize);
        dim3 blockDim(blockSize);
        
        auto ret = hipOccupancyMaxPotentialBlockSize(
            &gridSize, // int* gridSize
            &blockSize, // int* blockSize,
            vecAdd, //hipFunction_t f,
            lds_bytes, //size_t dynSharedMemPerBlk,
            1024 //int blockSizeLimit 
            );

        // https://docs.amd.com/projects/HIP/en/docs-5.0.0/doxygen/html/group___occupancy.html#ga59c488f35b0ba4b4938ba16e1a7ed7ec
        
        // const auto ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        //     &max_blocks_per_sm, vecAdd, blockDim.x * blockDim.y * blockDim.z, lds_bytes);
        if(ret != hipSuccess)  {
            std::stringstream ss;
            ss << "hipOccupancyMaxActiveBlocksPerMultiprocessorkernel failed with code ";
            ss << ret;
            ss << " " << hipGetErrorName(ret);
            ss << " which means: " << hipGetErrorString(ret);
            throw std::runtime_error(ss.str());
        }
        std::cout << "Launch parameters for maximum occupancy: gridSize "
                  << gridSize << "\tblockSize " << blockSize << "\n";

        int numBlocks = 0;
        ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks, // int* numBlocks,
            vecAdd,     // const void* f,
            blockSize,  // int blockSize,
            lds_bytes   //size_t dynSharedMemPerBlk 
            );
        if(ret != hipSuccess)  {
            throw std::runtime_error("hipOccupancyMaxActiveBlocksPerMultiprocessor failed");
        }
        std::cout << "Occupancy (numBlocks): " << numBlocks << "\n";

        std::vector<float> vala(N);
        std::vector<float> valb(N);

        const size_t valbytes = vala.size() * sizeof(decltype(vala)::value_type);
        float* d_a = nullptr;
        float* d_b = nullptr;
        ret = hipMalloc(&d_a, valbytes);
        if(ret != hipSuccess)  {
            throw std::runtime_error("hipMalloc failed");
        }
        ret = hipMalloc(&d_b, valbytes);
        if(ret != hipSuccess)  {
            throw std::runtime_error("hipMalloc failed");
        }
        if(hipMemcpy(d_a, vala.data(), valbytes, hipMemcpyHostToDevice) != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed");
        }
        if(hipMemcpy(d_b, valb.data(), valbytes, hipMemcpyHostToDevice) != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed");
        }

        vecAdd<<<gridDim, blockDim, lds_bytes >>>(d_a, d_b, N);

        // FIXME: copy data back, check results, get performance, check return code.
    }

    
    // TODO: add timing and launch stuff.
    // TODO: pass stuff via command-line?
    
    return 0;

    
}
