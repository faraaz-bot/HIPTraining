/************************
 DO NOT MODIFY THIS FILE
 DO NOT MODIFY THIS FILE
 DO NOT MODIFY THIS FILE
************************/

#pragma once

// Device function that puts the device to sleep for a specified amount of cycles
__device__ void kernel_sleep(int n)
{
    for(int i = 0; i < n; ++i)
    {
        // Only 1 to 127 are valid arguments
        __builtin_amdgcn_s_sleep(127);
    }
}

// Assemble frame
__global__ void kernelAssembleFrame(bool* done)
{
    // Assemble frame takes a few cycles
    kernel_sleep(5000);

    // Set assemble flag "done"
    *done = 1;
}

// Insert engine
__global__ void kernelInsertEngine(bool* done)
{
    // Insert engine takes a few cycles
    kernel_sleep(5000);

    // Set engine flag "done"
    *done = 1;
}

// Paint and install body
__global__ void kernelPaintAndInstallBody(bool* done)
{
    // Paint and install body takes a few cycles
    kernel_sleep(5000);

    // Set body flag "done"
    *done = 1;
}

// Install wheels and tires
__global__ void kernelInstallWheelsAndTires(bool* done)
{
    // Install wheels and tires takes a few cycles
    kernel_sleep(5000);

    // Set wheels flag "done"
    *done = 1;
}

// Ship to customer
__global__ void kernelShipToCustomer(bool* done)
{
    // Ship to customer takes a few cycles
    kernel_sleep(5000);

    // Set shipped flag "done"
    *done = 1;
}

/************************
 DO NOT MODIFY THIS FILE
 DO NOT MODIFY THIS FILE
 DO NOT MODIFY THIS FILE
************************/
