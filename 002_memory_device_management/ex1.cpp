#include <iostream>
#include <hip/hip_runtime.h>
using namespace std;

__global__ void helloFromGPU(){

int tid = blockIdx.x * blockDim.x + threadIdx.x;
printf("Hello from GPU, thread number: %d \n", tid);

}

void helloFromCPU(){
cout<<"Hello from CPU! " << endl;
}

int main(){
helloFromGPU<<<1,16>>>();
helloFromCPU();
return 0;
}