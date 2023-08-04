#include <hip/hip_runtime.h>
#include <cstdlib>
#include <unistd.h>

#include "car.hpp"
#include "common.hpp"
#include "kernels.hpp"

Car::Car()
{
    // Temporary device memory - this is required to trick the compiler
    // to not optimize out the sleep statements - DO NOT MODIFY
    HIP_CALL(hipMalloc((void**)&tmp, sizeof(bool)));

    frame   = (bool*)malloc(sizeof(bool));
    engine  = (bool*)malloc(sizeof(bool));
    body    = (bool*)malloc(sizeof(bool));
    wheels  = (bool*)malloc(sizeof(bool));
    shipped = (bool*)malloc(sizeof(bool));

    *frame   = false;
    *engine  = false;
    *body    = false;
    *wheels  = false;
    *shipped = false;

    /***************************************
     Modify this function below as required
    ***************************************/


}

Car::~Car()
{
    // Clean temporary device memory - DO NOT MODIFY
    HIP_CALL(hipFree(tmp));

    /***************************************
     Modify this function below as required
    ***************************************/


    /***************************************
     Modify this function above as required
    ***************************************/

    // Clean up host memory
    free(frame);
    free(engine);
    free(body);
    free(wheels);
    free(shipped);
}

// This function assembles the frame of the car
void AssembleFrame(Car* car)
{
    /*********************************
     Modify this function as required
    *********************************/

    // Run kernel
    kernelAssembleFrame<<<1, 1>>>(car->tmp);

    // Mark frame flag as processed
    HIP_CALL(hipMemcpy(car->frame, car->tmp, sizeof(bool), hipMemcpyDeviceToHost));
}

// This function inserts the cars engine. It is strictly sequential and thus should be
// processed by the host
void InsertEngine(Car* car)
{
    /*********************************
     Modify this function as required
    *********************************/

    // To insert the engine on the host path, we need to make sure all previous
    // work has been completed
    HIP_CALL(hipDeviceSynchronize());

    // Insert the engine takes 40ms
    usleep(40000);

    *(car->engine) = true;
}

// This function paints and installs body
void PaintAndInstallBody(Car* car)
{
    /*********************************
     Modify this function as required
    *********************************/

    // Run kernel
    kernelPaintAndInstallBody<<<1, 1>>>(car->tmp);

    // Mark body flag as processed
    HIP_CALL(hipMemcpy(car->body, car->tmp, sizeof(bool), hipMemcpyDeviceToHost));
}

// This function installs wheels and tires
void InstallWheelsAndTires(Car* car)
{
    /*********************************
     Modify this function as required
    *********************************/

    // Run kernel
    kernelInstallWheelsAndTires<<<1, 1>>>(car->tmp);

    // Mark wheels flag as processed
    HIP_CALL(hipMemcpy(car->wheels, car->tmp, sizeof(bool), hipMemcpyDeviceToHost));
}

// This function ships the car to customer
void ShipToCustomer(Car* car)
{
    /*********************************
     Modify this function as required
    *********************************/

    // Run kernel
    kernelShipToCustomer<<<1, 1>>>(car->tmp);

    // Mark shipped flag as processed
    HIP_CALL(hipMemcpy(car->shipped, car->tmp, sizeof(bool), hipMemcpyDeviceToHost));
}
