#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <chrono>

#include <hip/hip_runtime.h>

#include <stdio.h>

#include "kernels.h"

int main()
{
    int m = 100;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    std::vector<float> hdata(m);

    // Fill data
    std::cout << "Initial data" << std::endl;
    for (int i = 0; i < m; i++)
    {
        hdata[i] = (float)i;
        std::cout << hdata[i] << " ";
    }
    std::cout << "" << std::endl;

    float* ddata = nullptr;
    HIP_CHECK(hipMallocAsync((void**)&ddata, sizeof(float) * m, stream));
    HIP_CHECK(hipMemcpyAsync(ddata, hdata.data(), sizeof(float) * m, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Step 1: Create a graph 1
    // Step 2: Use begin and end stream capture to add scale kernels to graph 1
    // Step 3: Instantiate graph 1 into an executable graph
    // Step 4: Create a second graph 2
    // Step 5: Use begin and end stream capture to add scale kernels to graph 2, this time 
    //         using a different scale factor from what was used in the scaling kernels added 
    //         to graph 1 above
    // Step 6: Launch executable graph made from graph 1

    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> hresults(m, 0);
    HIP_CHECK(hipMemcpy(hresults.data(), ddata, sizeof(float) * m, hipMemcpyDeviceToHost));

    std::cout << "Results" << std::endl;
    for (size_t i = 0; i < hresults.size(); i++)
    {
        std::cout << hresults[i] << " ";
    }
    std::cout << "" << std::endl;

    // Step 7: Try and apply changes found in graph 2 to the executable graph.
    // Step 8: Launch executable graph (this time with changes applied from graph 2)

    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipMemcpy(hresults.data(), ddata, sizeof(float) * m, hipMemcpyDeviceToHost));

    std::cout << "Updated results" << std::endl;
    for (size_t i = 0; i < hresults.size(); i++)
    {
        std::cout << hresults[i] << " ";
    }
    std::cout << "" << std::endl;

    // Step 9: Free graph resources

    // Free device array
    HIP_CHECK(hipFree(ddata));

    return 0;
}
