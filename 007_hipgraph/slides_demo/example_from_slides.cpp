#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
#include <ctime>
#include <chrono>
#include <fstream>

#include <hip/hip_runtime.h>

#include <stdio.h>

#include "../kernels.h"

double run(int m, int iter, bool use_hip_graph)
{
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    std::vector<float> hdata(m);

    // Fill data
    std::cout << "Initial data" << std::endl;
    for (int i = 0; i < m; i++)
    {
        hdata[i] = 0.1f * (i % 113);
        //std::cout << hdata[i] << " ";
    }
    std::cout << "" << std::endl;

    float* ddata = nullptr;
    HIP_CHECK(hipMallocAsync((void**)&ddata, sizeof(float) * m, stream));
    HIP_CHECK(hipMemcpyAsync(ddata, hdata.data(), sizeof(float) * m, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    hipGraph_t graph;
    hipGraphExec_t instance;

    std::cout << "iter: " << iter << std::endl;

    auto wcts = std::chrono::system_clock::now();
    if(use_hip_graph)
    {
        std::cout << "Using hipgraph" << std::endl;

        bool graphCreated = false;
        for (int i = 0; i < iter; i++)
        {
            if (!graphCreated)
            {
                // Create graph
                HIP_CHECK(hipGraphCreate(&graph, 0));

                // Begin capture
                HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

                for (int j = 0; j < 10; j++)
                {
                    hipLaunchKernelGGL(scale, dim3((m - 1) / 256 + 1), dim3(256), 0, stream, m, ddata, 2.0f);
                    hipLaunchKernelGGL(scale, dim3((m - 1) / 256 + 1), dim3(256), 0, stream, m, ddata, 0.5f);
                }

                // End capture
                HIP_CHECK(hipStreamEndCapture(stream, &graph));

                // Instantiate executable graph
                HIP_CHECK(hipGraphInstantiate(&instance, graph, NULL, NULL, 0));
                graphCreated = true;
            }

            // Launch graph
            HIP_CHECK(hipGraphLaunch(instance, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        }
    }
    else
    {
        std::cout << "Using regular kernel launch" << std::endl;
        for (int i = 0; i < iter; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                hipLaunchKernelGGL(scale, dim3((m - 1) / 256 + 1), dim3(256), 0, stream, m, ddata, 2.0f);
                hipLaunchKernelGGL(scale, dim3((m - 1) / 256 + 1), dim3(256), 0, stream, m, ddata, 0.5f);
            }
            HIP_CHECK(hipStreamSynchronize(stream));
        }
    }
    std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
    std::cout << "Finished in " << wctduration.count() << " seconds [Wall Clock]" << std::endl;



    std::vector<float> hresults(m, 0);
    HIP_CHECK(hipMemcpy(hresults.data(), ddata, sizeof(float) * m, hipMemcpyDeviceToHost));

    //std::cout << "Results" << std::endl;
    //for (size_t i = 0; i < hresults.size(); i++)
    //{
    //    std::cout << hresults[i] << " ";
    //}
    //std::cout << "" << std::endl;

    if(use_hip_graph)
    {
        // Free graph resources
        HIP_CHECK(hipGraphDestroy(graph));
        HIP_CHECK(hipGraphExecDestroy(instance));
    }

    // Free device array
    HIP_CHECK(hipFree(ddata));

    return wctduration.count();
}


int main()
{
    int m = 1000000;
    std::string filename = "without_hipgraph_gfx942_" + std::to_string(m) + ".data";
    std::ofstream file(filename);
    if(file.is_open())
    {
        std::array<int, 20> sizes = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000};
        for (size_t i = 0; i < sizes.size(); i++)
        {
            file << std::to_string(sizes[i]);
            if(i != sizes.size() - 1)
            {
                file << ",";
            }
            
        }
        file << "\n";
        
        for (size_t i = 0; i < sizes.size(); i++)
        {
            file << std::to_string(run(m, sizes[i], false));
            if(i != sizes.size() - 1)
            {
                file << ",";
            }
        }
        file << "\n";

        file.close();
    }
    else
    {
        std::cout << "Could not open file" << std::endl;
    }

    std::string filename_hipgraph = "with_hipgraph_gfx942_" + std::to_string(m) + ".data";
    std::ofstream file_hipgraph(filename_hipgraph);
    if(file_hipgraph.is_open())
    {
        std::array<int, 20> sizes = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000};
        for (size_t i = 0; i < sizes.size(); i++)
        {
            file_hipgraph << std::to_string(sizes[i]);
            if(i != sizes.size() - 1)
            {
                file_hipgraph << ",";
            }
            
        }
        file_hipgraph << "\n";
        
        for (size_t i = 0; i < sizes.size(); i++)
        {
            file_hipgraph << std::to_string(run(m, sizes[i], true));
            if(i != sizes.size() - 1)
            {
                file_hipgraph << ",";
            }
        }
        file_hipgraph << "\n";

        file_hipgraph.close();
    }
    else
    {
        std::cout << "Could not open file" << std::endl;
    }
}

