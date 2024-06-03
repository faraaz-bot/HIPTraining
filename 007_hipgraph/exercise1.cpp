#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <chrono>

#include <hip/hip_runtime.h>

#include <stdio.h>

#include "kernels.h"

double run_graph(int m, int iter, bool use_hip_graph)
{
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    std::vector<float> hdata(3 * m);

    // Fill data
    std::cout << "Initial data" << std::endl;
    for (int i = 0; i < m; i++)
    {
        hdata[i + 0 * m] = 0.0f;
        hdata[i + 1 * m] = 1.0f;
        hdata[i + 2 * m] = 2.0f;
    }
    std::cout << "" << std::endl;

    float* ddata = nullptr;
    HIP_CHECK(hipMallocAsync((void**)&ddata, sizeof(float) * 3 * m, stream));

    float valueA = 1.0f;
    float valueB = 2.0f;
    float valueC = 3.0f;

    // 0 0 0 0 1 1 1 1 2 2 2 2
    // 0 0 0 0 2 2 2 2 6 6 6 6
    // 0 0 0 0 8 8 8 8 6 6 6 6
    // -8 -8 -8 -8 8 8 8 8 6 6 6 6

    hipGraph_t graph;
    hipGraphExec_t instance;

    float* dataA = ddata;
    float* dataB = ddata + m;
    float* dataC = ddata + 2 * m;

    std::cout << "Number of iterations: " << iter << " m: " << m << " using hipgraph? " << use_hip_graph << std::endl;

    auto wcts = std::chrono::system_clock::now();
    if (use_hip_graph)
    {
        std::cout << "Using hipgraph" << std::endl;

        bool graphCreated = false;
        for (int i = 0; i < iter; i++)
        {
            if (!graphCreated)
            {
		// Step 1: Create a hipGraph
		// Step 2: Create graph nodes (memcpy, scale, scale, scale, add, and subtract)
		//         and add them to the graph with proper dependencies to match the exercise graph
		// Step 3: Instaniate executable graph

                graphCreated = true;
            }

	    // Step 4: Launch graph
            
            HIP_CHECK(hipStreamSynchronize(stream));
        }
    }
    else
    {
        std::cout << "Using regular kernel launch" << std::endl;
        for (int i = 0; i < iter; i++)
        {
            HIP_CHECK(hipMemcpyAsync(ddata, hdata.data(), sizeof(float) * 3 * m, hipMemcpyHostToDevice, stream));
	    hipLaunchKernelGGL(scale, dim3((m - 1) / 256 + 1), dim3(256), 0, stream, m, dataA, valueA);
	    hipLaunchKernelGGL(scale, dim3((m - 1) / 256 + 1), dim3(256), 0, stream, m, dataB, valueB);
	    hipLaunchKernelGGL(scale, dim3((m - 1) / 256 + 1), dim3(256), 0, stream, m, dataC, valueC);
	    hipLaunchKernelGGL(add, dim3((m - 1) / 256 + 1), dim3(256), 0, stream, m, dataB, dataB, dataC);
	    hipLaunchKernelGGL(subtract, dim3((m - 1) / 256 + 1), dim3(256), 0, stream, m, dataA, dataA, dataB);
            HIP_CHECK(hipStreamSynchronize(stream));
        }
    }
    std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
    std::cout << "Finished in " << wctduration.count() << " seconds [Wall Clock]" << std::endl;

    /*std::vector<float> hresults(3 * m);
    HIP_CHECK(hipMemcpy(hresults.data(), ddata, sizeof(float) * 3 * m, hipMemcpyDeviceToHost));

    std::cout << "hresults" << std::endl;
    for (size_t i = 0; i < hresults.size(); i++)
    {
        std::cout << hresults[i] << " ";
    }
    std::cout << "" << std::endl;*/

    if (use_hip_graph)
    {
	// Step 5: Free graph resources
    }

    // Free device array
    HIP_CHECK(hipFree(ddata));

    return 0.0;
}

int main()
{
    run_graph(1000, 1000, true);
    run_graph(1000, 1000, false);

    return 0;
}
