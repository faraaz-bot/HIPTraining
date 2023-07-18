#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <malloc.h>

void compareArrays(int* hostArray, int* deviceArray, int numElems) {
  for (auto i=0; i<numElems; i++) {
    assert(hostArray[i] == deviceArray[i]);
  }
}

int main() {
  // Allocate an array on host and initialize it
  int numElems = 256;
  int hostArray[numElems];
  for (auto i=0; i<numElems; i++) {
    hostArray[i] = i;
  }

  // Allocate an array on device and copy host array to device
  /*** Insert code here ***/

  // What happens if you pass in a pointer instead of a double pointer?
  /*** Insert code here ***/

  // Compare the values of the entire array on host and device and assert if values don't match
  /*** Insert code here ***/

  // Free the device arrays
  /*** Insert code here ***/
  return 0;
}

