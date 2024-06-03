#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

__global__
void deviceMatrixAdd(float* c, const float* a, const float* b, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    c[idx] = a[idx] + b[idx];
}

void vectorAdd(std::vector<float>& c, const std::vector<float>& a, const std::vector<float>& b)
{
    for (int i = 0; i < c.size(); ++i)
        c[i] = a[i] + b[i];
}

bool compareVectors(const std::vector<float>& ref, const std::vector<float>& test)
{
    for (int i = 0; i < ref.size(); ++i)
    {
        if (ref[i] != test[i])
        {
            std::cout << "Mismatch at index " << i << ": " << ref[i] << " != " << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

void fillArray(std::vector<float>& v)
{
    for (int i = 0; i < v.size(); ++i)
        v[i] = i;
}

int ceilDiv(int num, int div)
{
    return (num + div - 1) / div;
}

int main()
{
    std::cout << "HIP matrix addition example" << std::endl;

    const int width = 1000000;
    const int height = 100;
    const int SIZE = width * height;

    std::vector<float> hostA(SIZE);
    fillArray(hostA);
    std::vector<float> hostB(SIZE);
    fillArray(hostB);

    float* devA = nullptr;
    assert(hipMalloc(&devA, SIZE * sizeof(float)) == hipSuccess);
    assert(hipMemcpy(devA, hostA.data(), SIZE * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);

    float* devB = nullptr;
    assert(hipMalloc(&devB, SIZE * sizeof(float)) == hipSuccess);
    assert(hipMemcpy(devB, hostB.data(), SIZE * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);

    float* devC = nullptr;
    assert(hipMalloc(&devC, SIZE * sizeof(float)) == hipSuccess);

    dim3 blockSize(8, 8);
    dim3 blocks(ceilDiv(width, blockSize.x), ceilDiv(height, blockSize.y));
    deviceMatrixAdd<<<blocks, blockSize>>>(devC, devA, devB, width, height);
    hipDeviceSynchronize();

    std::vector<float> refC(SIZE);
    vectorAdd(refC, hostA, hostB);
    std::vector<float> testC(SIZE);
    assert(hipMemcpy(testC.data(), devC, SIZE * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);

    if (compareVectors(refC, testC))
        std::cout << "Test passed" << std::endl;

    return 0;
}
