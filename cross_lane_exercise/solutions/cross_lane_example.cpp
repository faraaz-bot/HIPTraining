/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

//////////////////////////////////////////////////////
/// Welcome to the Cross-Lane Ops coding exercise! ///
//////////////////////////////////////////////////////

// Nomenclature:
//
// - Cross-lane operation: Special instruction or method to share data
// between threads.
//
// - Compute Unit (CU): GPU processing units on which code is scheduled to
// execute.
//
// - Wave (warp): one group of 64 threads, we pretend they act in unison.
// but are differentiated by their thread IDs.
//
// - Thread-Block: a group of 1 or more waves, which operate on the same
// CU and have access to limited shared resources.

// Problem Description:
//
// The following exercise is a intended as an introduction to different
// cross-lane operations, either using LDS memory or special __builtin
// instructions to serve this purpose. Each cross-lane operation is
// implemented differently, with different pros and cons that have been
// summarized in the accompanying slide deck.
//
// The current problem is simple: 
// 1. Our input data is not in the correct order: even and odd elements
// must be swapped, and
// 2. Our data needs to add +1 to each value.
//
// Let's make a few assumptions to begin:
// 1. Data size is dword (4 bytes)
// 2. Each thread will process 1 data element
// 3. Each wave is 64 elements wide (CDNA arch)
// 3. Threadblocks will be 1D in the X dimension
// (e.g. ThreadBlock = (64, 1, 1), where threadIdx.x is the thread's ID)
//
// Goals:
// 1. Write a solution to our problem using LDS.
// 2. Write a solution to our problem using Swizzle.
// 3. Write a solution to our problem using DPP.
// 4. Write a solution to our problem using BPermute.
// 5. Evaluate the performance of each solution.
// 6. Conclude with observational differences of each solution.
//
// Supported Architecture:
// All of the __builtin instructions in Wave64 in this exercise are supported on:
// MI-class cards such as: gfx908 (MI-100) and gfx90A (MI-200).

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <iostream>
#include <numeric>
#include <vector>

#define TRIAL_NUM 100

#define HIP_CHECK(condition)                                                           \
    {                                                                                  \
        hipError_t error = condition;                                                  \
        if(error != hipSuccess)                                                        \
        {                                                                              \
            std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
            exit(error);                                                               \
        }                                                                              \
    }

///////////////
/// Helpers ///
///////////////

template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
__host__ __device__ static constexpr intT1 ceilDiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}

// The following union helps to index the data footprint
// of element type DataT into b32 chunks. This is important
// because most cross-lane operation __builtins operate on
// dword sized elements. This takes care of the type and size
// aliasing required for the __builtin.
template<typename DataT>
union B32Helper
{
    DataT data;
    uint32_t b32[ceilDiv(sizeof(DataT), sizeof(int))];
};

/////////////////////////////////////////////////////
/// Helper functions that wrap backend __builtins ///
/////////////////////////////////////////////////////

// DPP move __builtin wrapper
template <
        uint32_t DppCtrl,
        uint32_t WriteRowMask,
        uint32_t WriteBankMask,
        bool     BoundCtrl,
        typename DataT>
__device__ static inline DataT dpp_impl_b32(DataT src0, DataT src1)
{
    constexpr int dword_count = ceilDiv(sizeof(DataT), sizeof(int));

    auto alias0 = B32Helper<DataT>{ src0 };
    auto alias1 = B32Helper<DataT>{ src1 };
    auto result = B32Helper<DataT>{ static_cast<DataT>(0) };

    // The __builtin operates on dword element sizes (e.g. b32 register elements)
    // If for example the datatype is f64, we must iterate the operation over upper
    // and lower 32b elements.
    #pragma unroll
    for(int i = 0; i < dword_count; i++)
    {
        result.b32[i] = __builtin_amdgcn_update_dpp(
            alias1.b32[i], // fill value 'prev'
            alias0.b32[i], // Src value
            DppCtrl, // DPP control code
            WriteRowMask, // Mask for affected rows (groups of 16 within each register)
            WriteBankMask, // Mask for affected banks (groups of 4 within each row)
            BoundCtrl); // Fill in 0 on invalid indices
    }

    return result.data;
}

// Swizzle __builtin wrapper
template <
        uint32_t SwizzleCtrl,
        typename DataT>
__device__ static inline DataT swizzle_impl_b32(DataT src0)
{
    constexpr int word_count = ceilDiv(sizeof(DataT), sizeof(int));

    auto alias0 = B32Helper<DataT>{ src0 };
    auto result = B32Helper<DataT>{ static_cast<DataT>(0) };

    // The __builtin operates on dword element sizes (e.g. b32 register elements)
    // If for example the datatype is f64, we must iterate the operation over upper
    // and lower 32b elements.
    #pragma unroll
    for(int i = 0; i < word_count; i++)
    {
        result.b32[i] = __builtin_amdgcn_ds_swizzle(
            alias0.b32[i], // Src value
            SwizzleCtrl); // Swizzle control code
    }

    return result.data;
}

// BPermute __builtin wrapper
template <typename DataT>
__device__ static inline DataT bpermute_impl_b32(DataT src0, uint32_t laneId)
{
    constexpr int word_count = ceilDiv(sizeof(DataT), sizeof(int));

    auto alias0 = B32Helper<DataT>{ src0 };
    auto result = B32Helper<DataT>{ static_cast<DataT>(0) };

    // The __builtin operates on dword element sizes (e.g. b32 register elements)
    // If for example the datatype is f64, we must iterate the operation over upper
    // and lower 32b elements.
    #pragma unroll
    for(int i = 0; i < word_count; i++)
    {
        // NOTE: final address is laneId * 4
        result.b32[i] = __builtin_amdgcn_ds_bpermute(
            laneId << 2,    // Lane ID to pull from for current thread 
            alias0.b32[i]); // Src value
    }
    
    return result.data;
}

///////////////////////////////////////////////////////////
/// Test functions that use the backend implementations ///
///////////////////////////////////////////////////////////

template<typename DataT>
__global__ void test_with_lds(const DataT* d_input, DataT* d_output)
{
    extern __shared__ DataT lds[];
    int                 idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure loads / stores are outside trial loop;
    auto tmp = d_input[idx];

#pragma unroll
    for(int i = 0; i < TRIAL_NUM; i++)
    {
        // LDS needs synchronization barriers after writing and after
        // reading.
        lds[threadIdx.x] = tmp + static_cast<DataT>(1);
        __syncthreads();

        // Invert the last bit of thread Id to swap evens / odds.
        // Then add +1
        tmp = lds[threadIdx.x ^ 0x1];
        __syncthreads();
    }

    d_output[idx] = tmp;
}

template<typename DataT>
__global__ void test_with_swizzle(const DataT* d_input, DataT* d_output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure loads / stores are outside trial loop;
    auto tmp = d_input[idx];
    
#pragma unroll
    for(int i = 0; i < TRIAL_NUM; i++)
    {
        // Add one to result after swizzle
        // Swizzle quad perm mode = 0x8000
        // quad_perm:[1,0,3,2] -> 1011001 = 0xB1
        // Swap for Swizzle Ctrl = 0x8000 | 0x00B1
        tmp = swizzle_impl_b32<0x80B1>(tmp) + static_cast<DataT>(1);
    }

    d_output[idx] = tmp;
}


template<typename DataT>
__global__ void test_with_dpp(const DataT* d_input, DataT* d_output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure loads / stores are outside trial loop;
    auto tmp = d_input[idx];

    // Swap for DPP = quad_perm:[1,0,3,2] -> 10110001 = 0xB1
#pragma unroll
    for(int i = 0; i < TRIAL_NUM; i++)
    {
        // Add one to result after dpp
        tmp = dpp_impl_b32<0xB1, 0xF, 0xF, false>(tmp, DataT(0)) + static_cast<DataT>(1);
    }

    d_output[idx] = tmp;
}

template<typename DataT>
__global__ void test_with_bpermute(const DataT* d_input, DataT* d_output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure loads / stores are outside trial loop;
    auto tmp = d_input[idx];

    // Neighbouring index is inversion of LSB
#pragma unroll
    for(int i = 0; i < TRIAL_NUM; i++)
    {
        // Add one to result after bpermute
        tmp = bpermute_impl_b32(tmp, idx ^ 0x1) + static_cast<DataT>(1);
    }

    d_output[idx] = tmp;
}

///////////////////////////////////////////////////////////
/// Validation function on CPU to check the result      ///
///////////////////////////////////////////////////////////

template<typename DataT>
__host__ bool checkResult(const DataT* input, const DataT* output, const uint32_t elementCount)
{
    for(uint32_t i = 0; i < elementCount; i++)
    {
        if(output[i] != (input[i ^ (TRIAL_NUM % 2)] + TRIAL_NUM))
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    using T = float;
    int test_id = 0;

    if(argc > 1)
    {
        test_id = atoi(argv[1]);
    }

    const int blockDim = 64;
    const int gridDim  = 512;
    const int size   = gridDim * blockDim;

    std::vector<T> input(size);
    std::iota(input.begin(), input.end(), 0);
    std::vector<T> output(size, -1);

    T* d_input;
    T* d_output;

    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));

    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));

    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    switch(test_id)
    {
    case 0:
        hipExtLaunchKernelGGL(test_with_lds,
                          dim3(gridDim),
                          dim3(blockDim),
                          blockDim * sizeof(T), // sharedMemBytes
                          0, // stream
                          startEvent, // Event start
                          stopEvent, // event stop
                          0, // flags
                          (const T*)d_input,
                          d_output);
        break;
    case 1:
        hipExtLaunchKernelGGL(test_with_swizzle,
                          dim3(gridDim),
                          dim3(blockDim),
                          0, // sharedMemBytes
                          0, // stream
                          startEvent, // Event start
                          stopEvent, // event stop
                          0, // flags
                          (const T*)d_input,
                          d_output);
        break;
    case 2:
        hipExtLaunchKernelGGL(test_with_dpp,
                          dim3(gridDim),
                          dim3(blockDim),
                          0, // sharedMemBytes
                          0, // stream
                          startEvent, // Event start
                          stopEvent, // event stop
                          0, // flags
                          (const T*)d_input,
                          d_output);
        break;
    case 3:
        hipExtLaunchKernelGGL(test_with_bpermute,
                          dim3(gridDim),
                          dim3(blockDim),
                          0, // sharedMemBytes
                          0, // stream
                          startEvent, // Event start
                          stopEvent, // event stop
                          0, // flags
                          (const T*)d_input,
                          d_output);
        break;
    default:
        std::cout << "Invalid test selection\n";
        CHECK_HIP_ERROR(hipEventDestroy(startEvent));
        CHECK_HIP_ERROR(hipEventDestroy(stopEvent));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        exit(0);
    }

    // Check for errors
    HIP_CHECK(hipPeekAtLastError());

    // Calculate elapsed time
    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    std::cout << "Element count: " << size << " Elapsed Time (ms): " << elapsedTimeMs << std::endl; 

    HIP_CHECK(hipMemcpy(output.data(), d_output, size * sizeof(T), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    bool validate = true;
    bool printOutput = false;

    if(validate)
    {
        std::cout << (checkResult(input.data(), output.data(), size) ? "Success!\n" : "Failed!\n") << std::endl;
    }

    if(printOutput)
    {
        for(size_t i = 0; i < size; i++)
        {
            std::cout << "[" << i << "]" << output[i] << "\t";
        }
        std::cout << std::endl;
    }
    return 0;
}
