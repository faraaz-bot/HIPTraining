#pragma once

#include <hip/hip_runtime_api.h>

// Struct to hold car info
struct Car
{
    Car(void);
    ~Car(void);

    // Some device memory - this is required to trick the compiler
    // to not optimize out the sleep statements
    bool* tmp;

    // Flags that store status of each step
    bool* frame;
    bool* engine;
    bool* body;
    bool* wheels;
    bool* shipped;

    /***************************************
     Modify this function below as required
    ***************************************/


};

void AssembleFrame(Car*);
void InsertEngine(Car*);
void PaintAndInstallBody(Car*);
void InstallWheelsAndTires(Car*);
void ShipToCustomer(Car*);
