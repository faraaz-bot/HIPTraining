#include <hip/hip_runtime.h>
#include <malloc.h>
using namespace std;

void compareArrays(int* hostArray0, int* deviceArray, int numElems) {
  for (auto i=0; i<numElems; i++) {
    assert(hostArray0[i] == deviceArray[i]);
  }
}

int main() {
  // Allocate hostArray0 and initialize it
  int numElems = 256;
  int hostArray0[numElems];
  for (auto i=0; i<numElems; i++) {
    hostArray0[i] = i;
  }

  // Set current device to 0
  /*** Insert code here ***/

  // Confirm that current device is 0
  /*** Insert code here ***/

  // Allocate an array on device 0 and copy hostArray0 to deviceArray0
  /*** Insert code here ***/

  // Allocate an array on device 1 and copy deviceArray0 to deviceArray1
  /*** Insert code here ***/

  // Compare the values of the entire deviceArray0 and deviceArray1 and assert if values don't match
  /*** Insert code here ***/

  // Allocate hostArray1
  int hostArray1[numElems];

  // Copy deviceArray1 to hostArray1
  /*** Insert code here ***/

  // Compare the values of the entire hostArray0 and hostArray1 and assert if values don't match
  /*** Insert code here ***/

  // Free the host and device arrays, as applicable
  /*** Insert code here ***/

  // Set current device to (deviceCount + 1) and capture the error
  /*** Insert code here ***/

  // Allocate a host array and try to free it using HIP and capture the error
  /*** Insert code here ***/

  return 0;
}

