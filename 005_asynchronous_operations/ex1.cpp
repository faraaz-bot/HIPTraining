#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <hip/hip_runtime.h>
#include "car1.cpp"
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

    // Further modify your code, such that the host path gets fully unblocked
    // while still pipelining the sequential host only 'InsertEngine' part.

    // Verify your solution by checking the 'Host was blocked for' timer. It
    // should only show very few miliseconds.

    // Prime the pipeline
    AssembleFrame(&cars[0]);

    AssembleFrame(&cars[1]);
    InsertEngine(&cars[0]);

    AssembleFrame(&cars[2]);
    InsertEngine(&cars[1]);
    PaintAndInstallBody(&cars[0]);

    AssembleFrame(&cars[3]);
    InsertEngine(&cars[2]);
    PaintAndInstallBody(&cars[1]);
    InstallWheelsAndTires(&cars[0]);

    // Asynchronous car assembly
    for(int i = 2; i < NCARS - 2; ++i)
    {
        AssembleFrame(&cars[i + 2]);
        InsertEngine(&cars[i + 1]);
        PaintAndInstallBody(&cars[i]);
        InstallWheelsAndTires(&cars[i - 1]);
        ShipToCustomer(&cars[i - 2]);
    }

    // Empty the pipeline
    ShipToCustomer(&cars[NCARS - 4]);

    InstallWheelsAndTires(&cars[NCARS - 3]);
    ShipToCustomer(&cars[NCARS - 3]);

    PaintAndInstallBody(&cars[NCARS - 2]);
    InstallWheelsAndTires(&cars[NCARS - 2]);
    ShipToCustomer(&cars[NCARS - 2]);

    InsertEngine(&cars[NCARS - 1]);
    PaintAndInstallBody(&cars[NCARS - 1]);
    InstallWheelsAndTires(&cars[NCARS - 1]);
    ShipToCustomer(&cars[NCARS - 1]);

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