#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <hip/hip_runtime_api.h>

#include "car.hpp"
#include "common.hpp"

// Total number of cars to be constructed
#define NCARS 20

int main(int argc, char* argv[])
{
    // Array of cars
    Car cars[NCARS];

    HIP_CALL(hipDeviceSynchronize());
    auto tick = std::chrono::steady_clock::now();

    /***************************************
     Modify this function below as required
    ***************************************/

    // Implement pipelining
    // While the car assembly for a single car is in strictly sequential
    // order, multiple cars can be assembled at the same time.

    // Verify your assembly by visualizing profiler data generated with
    // rocprof

    // Car assembly
    for(int i = 0; i < NCARS; ++i)
    {
        AssembleFrame(&cars[i]);
        InsertEngine(&cars[i]);
        PaintAndInstallBody(&cars[i]);
        InstallWheelsAndTires(&cars[i]);
        ShipToCustomer(&cars[i]);
    }

    /***************************************
     Modify this function above as required
    ***************************************/

    // Measure time the host path is blocked by device instructions
    auto tack = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(tack - tick);
    std::cout << "Host was blocked for " << time.count() << " ms" << std::endl;

    // Measure total time until all computations are completed
    HIP_CALL(hipDeviceSynchronize());
    auto sync = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(sync - tick);
    std::cout << "Total time to compute: " << time.count() << " ms" << std::endl;

    // Verify all parts were built
    for(int i = 0; i < NCARS; ++i)
    {
        if(!(*(cars[i].frame)))   std::cerr << "Could not assemble frame for car " << i << std::endl;
        if(!(*(cars[i].engine)))  std::cerr << "Could not insert engine for car " << i << std::endl;
        if(!(*(cars[i].body)))    std::cerr << "Could not paint and install body for car " << i << std::endl;
        if(!(*(cars[i].wheels)))  std::cerr << "Could not install wheels and tires for car " << i << std::endl;
        if(!(*(cars[i].shipped))) std::cerr << "Could not ship car " << i << " to customer" << std::endl;
    }

    return 0;
}
