#include <iostream>
#include <hip/hip_runtime.h>
using namespace std;

void compareArrays(int* hostArray, int* deviceArray, int numElems);
{
for(auto int i = 0, i < numElems; i++){
  assert(hostArray[i] == deviceArray[i]);
  }
  printf("The arrays are equal");
}

int main(){
  //Allocate hostArray0 and initialize it.

  int numElems = 256;
  int arraySize = sizeof(int) * numElems;
  int hostArray0[numElems];

  for(auto int i = 0; i < numElems, i++){
    hostArray0[i] = i;
  }

  // Set current device to 0
  hipSetDevice(0);

  // Confirm that current device is 0
  int device;
  hipGetDevice(&device);
  printf("This device ID is: %d", device);

  // Allocate an array on device 0 and copy hostArray0 to deviceArray0
  int* deviceArray0;
  hipMalloc(&deviceArray0, arraySize);
  hipMemcpy(deviceArray0, hostArray0, arraySize, hipMemcpyHostToDevice);
  
  // Allocate hostArray1

  int hostArray1[numElems];

  // Copy deviceArray1 to hostArray1
  hipMemcpy(hostArray1, deviceArray1, arraySize, hipMemcpyDeviceToHost);

  // Free the host and device arrays, as applicable
  hipFree(deviceArray0);
  hipFree(deviceArray1);

  // Set current device to (deviceCount + 1) and capture the error
  int count;
  hipGetDeviceCount(&count);
  hipError_t error = hipSetDevice(count + 1);
  cout << "Error: " << hipGetErrorString(error) << endl;

  // Allocate a host array and try to free it using HIP and capture the error
  int hostArray2[numElems];
  error = hipFree(&hostArray2);
  cout << "Error: " << hipGetErrorString(error) << endl;
  
  int* hostArray3 = new int[numElems];
  error = hipFree(hostArray3);
  cout << "Error: " << hipGetErrorString(error) << endl;
  delete[] hostArray3;
  
return 0;
}