#include <hip/hip_runtime.h>
#include <iostream>

#define BLOCK_SIZE 512 // Define your desired block size

__global__ void reduce(float *g_data, float *g_odata, int len)
{
  //=====================Step4: First add during load=========================================//
  __shared__ float sdata[BLOCK_SIZE];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  //load data
  // ??

  // do reduction in shared mem
  // ??

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

int main()
{
  // Input data initialization
  float inputSize = 1024 * 1024 * 1024; /* Define the size of input data */
  ;
  float *h_input = new float[inputSize];
  for (int i = 0; i < inputSize; ++i)
  {
    h_input[i] = 1; // Assign appropriate values
  }

  // Allocate memory on the device for input and output data
  float *d_input;
  float *d_output;
  int numBlocks = (inputSize + (BLOCK_SIZE << 1) - 1) / (BLOCK_SIZE << 1); // Calculate the number of blocks

  hipMalloc((void **)&d_input, inputSize * sizeof(float));
  hipMalloc((void **)&d_output, numBlocks * sizeof(float));

  // Copy input data to the device
  hipMemcpy(d_input, h_input, inputSize * sizeof(float), hipMemcpyHostToDevice);

  // Create HIP events for timing
  hipEvent_t startEvent, endEvent;
  hipEventCreate(&startEvent);
  hipEventCreate(&endEvent);

  // Record start event
  hipEventRecord(startEvent, 0);

  // Launch the HIP kernel
  for (auto i = 0; i < 10; i++)
    reduce<<<dim3(numBlocks), dim3(BLOCK_SIZE), 0, 0>>>(d_input, d_output, inputSize);
  // Record end event
  hipEventRecord(endEvent, 0);
  hipEventSynchronize(endEvent);

  // Calculate and print kernel execution time
  float kernelTime;
  hipEventElapsedTime(&kernelTime, startEvent, endEvent);
  std::cout << "Kernel execution time: " << kernelTime / 10.0 << " milliseconds" << std::endl;

  // Copy the result back to the host
  float *h_output = new float[numBlocks];
  hipMemcpy(h_output, d_output, numBlocks * sizeof(float), hipMemcpyDeviceToHost);

  // Perform verification
  float expectedSum = inputSize;

  float computedSum = 0.0f;
  for (int i = 0; i < numBlocks; ++i)
  {
    computedSum += h_output[i];
  }

  if (abs(expectedSum - computedSum) < 1e-5)
  {
    std::cout << "Verification passed. Expected sum: " << expectedSum << " Computed sum: " << computedSum << std::endl;
  }
  else
  {
    std::cerr << "Verification failed. Expected sum: " << expectedSum << " Computed sum: " << computedSum << std::endl;
  }

  // Clean up allocated memory on the device and host
  hipFree(d_input);
  hipFree(d_output);
  delete[] h_input;
  delete[] h_output;

  return 0;
}