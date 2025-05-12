#include <hip/hip_runtime.h>

int main() {
  //Allocate a device array that requires twice the number of 
  //bytes that are in device global memory, and query 
  //the HIP return status and error string
  /*** Insert code here ***/
  hipDeviceProp_t deviceProperties;
  hipGetDeviceProperties(&deviceProperties, 0);
  size_t memSize = 2 * deviceProperties.totalGlobalMem;
  int* oversizeDeviceArray;
  hipError_t ret;
  ret = hipMalloc(&oversizeDeviceArray, memSize);
  printf("hipMalloc error: %s\n", hipGetErrorString(ret));
  // Query the last HIP error
  /*** Insert code here ***/
  ret = hipGetLastError();
  printf("Last HIP error: %s\n", hipGetErrorString(ret));


  // Verify that the error has been reset
  /*** Insert code here ***/
  ret = hipGetLastError();
  printf("Last HIP error t: %s\n", hipGetErrorString(ret));

  return 0;
}
