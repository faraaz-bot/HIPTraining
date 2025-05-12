#include <hip/hip_runtime.h>

int main() {
  // Allocate a device array that requires twice the number of bytes that are in device global memory, and query the HIP return status and error string
  
  
  // Query the last HIP error
  ret = hipGetLastError();
  printf("Last HIP error: %s\n", hipGetErrorString(ret));

  // Verify that the error has been reset
  ret = hipGetLastError();
  printf("Last HIP error after reset: %s\n", hipGetErrorString(ret));

  return 0;
}