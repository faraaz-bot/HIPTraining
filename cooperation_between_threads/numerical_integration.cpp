#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <hip/hip_runtime.h>

/** The goal of this workshop example is to practice using different types of communication
 *  to solve a problem. In this case, we will want to write a function to transform the input data 
 *  (map operation), construct the areas of the discrete trapezoids for numerical integration (stencil),
 *  and then reduce the areas to retrieve the discrete integral.
 */



/* Device function to apply root(1-x^2), accepts float x, and returns float */
inline __device__ float RootOneMinusX(float x)
{
}

/*  Applies RootOneMinusX to to data generated from xbeg to xbeg + n*dx 
    computes Reimann Rectangles for numerical integral
    and inputs into float pointer f1.
    Inputs: xbeg float, dx float, n int, float array f1.
    Outputs: float array f1.
 */
__global__ void GenTrapezoids(float xBeg, float dx, int n, float *f1)
{
}

/*  Computes the reduction of FP32 array d_in. 
    Ouput is FP32 array d_out. If input array is larger than 1024 floats
    A partial reduction is computed to the 1024th entries of d_out. 
    If less, the full reduction can be found in d_out[0].
    In this application d_in contains the Reimainn rectangles 
    to compute the numerical integral.  
*/
__global__ void reduceKernel(float *dOut, const float *dIn)
{
}

/* Host code to compute the digits of pi.
   Inputs: n int, the resolution of the integral to calculate pi
   Outputs: pi float, the estimate of pi.
*/ 
float MmmPi(int n)
{
    //Initialization
    
    //Allocation    
    
    //Kernel Parameters
    
    //Insert your map + stencil kernels

    //Call your reduction kernel

    //Free memory
    //return
}

/* Driver for the computation of pi. */
int main()
{
	//Call MmmPi and print your answer, 
}

