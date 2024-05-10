#include <hip/hip_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

/* Kernel example of a bad implementation of histogram.
   Inputs: Int array d_bins, Int array d_in, Int BIN_COUNT
   Ouput: Int array d_bins
*/
__global__ void FauxHisto(unsigned int *dBins, const int *dIn,
                          const int binCount) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int myItem = dIn[myId];
  int myBin = myItem % binCount;
  dBins[myBin]++;
}

/* Kernel, example of a simple but unoptimized implementation of histogram.
   Inputs: Int array d_bins, Int array d_in, Int BIN_COUNT
   Ouput: Int array d_bins
*/
__global__ void SimpleHisto(unsigned int *dBins, const int *dIn,
                            const int binCount) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int myItem = dIn[myId];
  int myBin = myItem % binCount;
  atomicAdd(&(dBins[myBin]), 1);
}

/*
    Driver function which compares implementations of histogram.
*/
int main() {
  using uint = unsigned int;
  uint *dBins;
  int *dIn;
  const int arraySize = 65536;
  const int arrayBytes = arraySize * sizeof(int);
  const int binCount = 16;
  const int binBytes = binCount * sizeof(uint);

  std::vector<uint> bins(binCount, 0);
  std::vector<int> in(arraySize);
  hipMalloc(&dIn, arrayBytes);
  hipMalloc(&dBins, binBytes);

  for (int i = 0; i < arraySize; i++)
    in[i] = i;

  hipMemcpy(dIn, in.data(), arrayBytes, hipMemcpyHostToDevice);
  hipMemcpy(dBins, bins.data(), binBytes, hipMemcpyHostToDevice);
  FauxHisto<<<arraySize / binCount, binCount>>>(dBins, dIn, binCount);
  hipMemcpy(bins.data(), dBins, binBytes, hipMemcpyDeviceToHost);
  std::cout << "First Try Histogram =" << std::endl;
  for (int i = 0; i < binCount; i++)
    std::cout << "Bin " << i << " = " << bins[i] << std::endl;

  hipMemset(dBins, 0, sizeof(uint) * binCount); 
  SimpleHisto<<<arraySize / binCount, binCount>>>(dBins, dIn, binCount);
  hipMemcpy(bins.data(), dBins, binBytes, hipMemcpyDeviceToHost);
  std::cout << "Real Histogram =" << std::endl;
  for (int i = 0; i < binCount; i++)
    std::cout << "Bin " << i << " = " << bins[i] << std::endl;

  hipFree(dBins);
  hipFree(dIn);
}
