/************************
 DO NOT MODIFY THIS FILE
 DO NOT MODIFY THIS FILE
 DO NOT MODIFY THIS FILE
************************/

#pragma once

#include <hip/hip_runtime_api.h>
#include <iostream>
#include <cstdlib>

#define HIP_CALL(cmd)                                                                       \
    do {                                                                                    \
        hipError_t error = (cmd);                                                           \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error)              \
                      << ") at line " << __LINE__ << " in file " << __FILE__ << std::endl;  \
            exit(-1);                                                                       \
        }                                                                                   \
    } while(0)

/************************
 DO NOT MODIFY THIS FILE
 DO NOT MODIFY THIS FILE
 DO NOT MODIFY THIS FILE
************************/
