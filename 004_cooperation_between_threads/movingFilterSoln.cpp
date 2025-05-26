#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <hip/hip_runtime.h>

/* Simple solution.
 *
 * Window size K is attenuated at the beginning of the signal.
 *
 * Each thread:
 * 1. computes a single output value,
 * 2. loads K input values.
 *
 * Global bandwidth is roughly:
 *
 *     number_of_threads * window_size
 *
 */
__global__ void movingFilterSimple(float const* inData, float* outData, int K, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= N)
        return;

    int windowSize = (index < K) ? (index + 1) : (K);

    float sum = inData[index];
    for(int i = 1; i < windowSize; i++)
        sum += inData[index - i];

    outData[index] = sum / windowSize;
}

/* LDS solution.
 *
 * Window size K is attenuated at the beginning of the signal.
 *
 * Each thread:
 * 1. computes a single output value,
 * 2. loads one input value (except for the first thread in each block).
 *
 * Global bandwidth is roughly:
 *
 *     number_of_threads + number_of_blocks * window_size
 *
 */
__global__ void movingFilterShared(float const* inData, float* outData, int K, int N)
{
    extern __shared__ float sharedData[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= N)
        return;

    int sharedIndex = threadIdx.x + K;
    int windowSize  = (index < K) ? (index + 1) : (K);

    if(threadIdx.x == 0)
    {
        for(int i = 0; i < K; i++)
        {
            if(index - i > 0)
                sharedData[sharedIndex - i - 1] = inData[index - i - 1];
        }
    }
    sharedData[sharedIndex] = inData[index];
    __syncthreads();

    float sum = sharedData[sharedIndex];
    for(int i = 1; i < windowSize; i++)
        sum += sharedData[sharedIndex - i];

    outData[index] = sum / windowSize;
}

/* Block specialized LDS solution.
 *
 * Window size K is attenuated at the beginning of the signal.
 *
 * Each thread:
 * 1. computes a single output value,
 * 2. loads one or two input values.
 *
 * Global bandwidth is roughly:
 *
 *     number_of_threads + number_of_blocks * window_size
 *
 * Block 0 is the only block that contains threads that will work on
 * data where K needs to be attenuated.  In all other Blocks, this is
 * not a concern.
 */
__global__ void blockSpecializedSmemMovingFilter(float const* inData, float* outData, int K, int N)
{
    extern __shared__ float sharedData[];

    if(blockIdx.x == 0)
    {
        int index = threadIdx.x;
        if(index >= N)
            return;

        sharedData[index] = inData[index];
        __syncthreads();

        int windowSize = (index < K) ? (index + 1) : (K);

        float sum = sharedData[index];
        for(int i = 1; i < windowSize; i++)
            sum += sharedData[index - i];
        outData[index] = sum / windowSize;
    }
    else
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= N)
            return;

        int windowSize = (index < K) ? (index + 1) : (K);

        if(threadIdx.x < K)
        {
            sharedData[threadIdx.x] = inData[index - K];
        }

        int sharedIndex         = threadIdx.x + K;
        sharedData[sharedIndex] = inData[index];
        __syncthreads();

        float sum = sharedData[sharedIndex];
        for(int i = 1; i < windowSize; i++)
            sum += sharedData[sharedIndex - i];

        outData[index] = sum / windowSize;
    }
}

void loadData(std::string file, std::vector<float>& data)
{
    std::ifstream f;
    f.open(file);
    for(int i = 0; i < data.size(); i++)
    {
        f >> data[i];
    }
    f.close();
}

void writeData(std::string file, std::vector<float>& data)
{
    std::ofstream     myFile;
    std::stringstream outputBuffer;

    myFile.open(file);
    for(int i = 0; i < data.size(); i++)
    {
        outputBuffer << data[i] << '\n';
    }
    myFile << outputBuffer.str() << std::flush;
    myFile.close();
}

int main(int argc, char* argv[])
{
    int k = 5;
    if(argc > 1)
    {
        k = std::stoi(argv[1]);
    }

    // Initialization
    int N = 1048576;

    std::vector<float> rawSignal(N);
    std::vector<float> filteredSignal(N);

    float* dRaw;
    float* dFiltered;

    size_t arraySize = N * sizeof(float);

    // Allocate and load
    hipMalloc(&dRaw, arraySize);
    hipMalloc(&dFiltered, arraySize);

    loadData("raw_signal.dat", rawSignal);
    hipMemcpy(dRaw, rawSignal.data(), arraySize, hipMemcpyHostToDevice);
    hipMemset(dFiltered, 0, arraySize);

    // Kernel parameters
    dim3 dimBlock(1024);
    dim3 dimGrid(N / dimBlock.x);
    int  smemBytes = (dimBlock.x + k) * sizeof(float);

    // Timers
    hipEvent_t start, stop;
    float      ms;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Basic moving filter
    hipMemset(dFiltered, 0, arraySize);
    hipEventRecord(start);
    movingFilterSimple<<<dimGrid, dimBlock>>>(dRaw, dFiltered, k, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms, start, stop);
    std::cout << "Basic time = " << ms << std::endl;
    hipMemcpy(filteredSignal.data(), dFiltered, arraySize, hipMemcpyDeviceToHost);
    writeData("filtered_signal.dat", filteredSignal);

    // LDS moving filter
    hipMemset(dFiltered, 0, arraySize);
    hipEventRecord(start);
    movingFilterShared<<<dimGrid, dimBlock, smemBytes>>>(dRaw, dFiltered, k, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms, start, stop);
    std::cout << "Smem time = " << ms << std::endl;
    hipMemcpy(filteredSignal.data(), dFiltered, arraySize, hipMemcpyDeviceToHost);
    writeData("filtered_signal_smem.dat", filteredSignal);

    // Block specialized LDS moving filter
    hipMemset(dFiltered, 0, arraySize);
    hipEventRecord(start);
    blockSpecializedSmemMovingFilter<<<dimGrid, dimBlock, smemBytes>>>(dRaw, dFiltered, k, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms, start, stop);
    std::cout << "Block specialized smem time = " << ms << std::endl;
    hipMemcpy(filteredSignal.data(), dFiltered, arraySize, hipMemcpyDeviceToHost);
    writeData("filtered_signal_bsmem.dat", filteredSignal);

    // Free
    hipFree(dRaw);
    hipFree(dFiltered);
}