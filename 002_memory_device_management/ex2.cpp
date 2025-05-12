#include <iostream>
#include <hip/hip_runtime.h>
using namespace std;

//send in addressof host Array, device array
void compareArrays(int* hostArray, int* deviceArray, int numElems){
for(auto i = 0; i < numElems; i++){
    assert(hostArray[i] == deviceArray[i]);
  }
  printf("Arrays are equal! \n");
}

int main(){
//Allocate an array on host and initialize it
int numElems = 256;
int hostArray[numElems];

for(auto i = 0; i < numElems; i++){
  hostArray[i] = i;
}

//Allocate an array on device and copy host array to device
//make array on device, then copy
int* deviceArray;
hipMalloc(&deviceArray, sizeof(int) * numElems);


hipError_t error = hipMemcpy(deviceArray, hostArray, sizeof(int) * numElems, hipMemcpyHostToDevice);
assert(error == hipSuccess);

  // Allocate memory to copy the device data back to the host
int hostResult[numElems];
error = hipMemcpy(hostResult, deviceArray, sizeof(int) * numElems, hipMemcpyDeviceToHost);
assert(error == hipSuccess);

//Compare hostArray and hostResult
compareArrays(hostArray, hostResult, numElems);

assert(hipFree(deviceArray)== hipSuccess);
return 0;
}
