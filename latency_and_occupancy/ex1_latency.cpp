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

__global__ void emptykernel()
{
    // NB: this kernel contains no bugs.
}

int main(int argc, char *argv[])
{
    // Command-line specified arguments
    int N;
    int lds_bytes;
    int blockSize;
    
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
    int gridSize = ceildiv(N, blockSize);
    std::cout << "Grid size:         " << gridSize << "\n";

    // Solution:
    {
        dim3 gridDim(gridSize);
        dim3 blockDim(blockSize);
        
        // Create HIP events for timing
        hipEvent_t startEvent, endEvent;
        hipEventCreate(&startEvent);
        hipEventCreate(&endEvent);
  
        // Record start event
        hipEventRecord(startEvent, 0);

        // Launch the kernel
        emptykernel<<<gridDim, blockDim >>>();
    
        hipDeviceSynchronize();
        if(hipGetLastError() != hipSuccess) {
            std::cerr << "Error in kernel launch\n";
        }
    
        // Record end event
        hipEventRecord(endEvent, 0);
        hipEventSynchronize(endEvent);
    
        // Calculate and print kernel execution time
        float kernelTime;
        hipEventElapsedTime(&kernelTime, startEvent, endEvent);
        std::cout << "Kernel execution time: " << kernelTime << " ms\n";
    }
    
    return EXIT_SUCCESS;
}
