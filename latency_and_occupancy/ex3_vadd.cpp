#include<iostream>
#include<vector>

#include <hip/hip_runtime.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;


// Computes ceil(numerator/divisor) for integer types.
template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
intT1 ceildiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}

__global__ void vecAdd(float * const a, const float * const  b, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        a[idx] += b[idx];
    }
}

int main(int argc, char *argv[])
{
    // Command-line specified arguments
    int N;
    int lds_bytes = 0;
    int blockSize = 0;
    
    po::options_description opdesc("rocfft rider command line options");
    opdesc.add_options()("h", "produces this help message")
        ("N", po::value<int>(&N)->default_value(1), "Problem size")
        ("L", po::value<int>(&lds_bytes)->default_value(0), "LDS bytes")
        ("B", po::value<int>(&blockSize)->default_value(512), "Thread block size");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);
    
    if(vm.count("h"))
    {
        std::cout << opdesc << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "N: " << N << "\n";
    std::cout << "LDS bytes: " << lds_bytes << "\n";
    std::cout << "Thread block size: " << blockSize << "\n";
    
    dim3 gridDim(blockSize);
    dim3 blockDim(ceildiv(N, blockSize));

    // Print the occupancy:
    int numBlocks = 0;
    auto ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(
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
    for(int i = 0; i < N; ++i) {
        vala[i] = 1.0 / (1.0 + i);
        valb[i] = 2.0 / (1.0 + i);
    }
    
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


    // Create HIP events for timing
    hipEvent_t startEvent, endEvent;
    hipEventCreate(&startEvent);
    hipEventCreate(&endEvent);
    
    // Record start event
    hipEventRecord(startEvent, 0);

    // Launch the kernel
    vecAdd<<<gridDim, blockDim, lds_bytes >>>(d_a, d_b, N);
        
    // Record end event
    hipEventRecord(endEvent, 0);
    hipEventSynchronize(endEvent);

    if(hipGetLastError() != hipSuccess)
    {
        std::cerr << "\tError running kernel!\n";
    }
    
    // Calculate and print kernel execution time
    float kernelTime;
    hipEventElapsedTime(&kernelTime, startEvent, endEvent);
    std::cout << "Kernel execution time: " << kernelTime / 10.0 << " ms\n";

    // Clean up: free GPU memory:
    ret = hipFree(d_a);
    if(ret != hipSuccess) {
        throw std::runtime_error("hipFree failed.");
    }
    ret = hipFree(d_b);
    if(ret != hipSuccess) {
        throw std::runtime_error("hipFree failed.");
    }

    
    return 0;

    
}
