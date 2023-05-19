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

////////////////////////////////////////////
/// Welcome to the MFMA coding exercise! ///
////////////////////////////////////////////

// Nomenclature:
//
// - GEMM: (GEneralized Matrix-Matrix Multiplication)
//
// - Compute Unit (CU): GPU processing units on which code is scheduled to
// execute.
//
// - Wave (warp): one group of 64 threads, we pretend they act in unison.
// Remember from GCN arch, that each vector instruction is processed
// in 4 x SIMD16 ops but this is hidden from the programmer. View
// each vector instruction as processed by the whole wave together.
//
// - Thread-Block: a group of 1 or more waves, which operate on the same
// CU and have access to limited shared resources.
//
// - BLOCK_XYZ: This refers to the block size of the block-wise GEMM
// decomposition.
//
// - MxNxK: GEMM problem size
//
// - col_major: column - major data storage in 1D format, where contiguous
// neighbors are column elements.
//
// - row_major: row - major data storage in 1D format, where contiguous
// neighbors are row elements.

// Problem Description:
//
// The following device kernel is a naive implementation
// of block-wise GEMM. Each wave will compute one BLOCK_M x BLOCK_N
// output block of the MxNxK GEMM, generalized as:
// D = alpha * (A x B) + beta * C
//
// In this simplified example, we assume:
// : A matrix is in col-major format       (M x K)
// : B matrix is in row-major format       (K x N)
// : C, D matrices are in col-major format (M x N)
// : C == D in size, data-type and layout
// : No LDS required
//
// Note: This is a simplified implementation to demonstrate functionality in
// context of wave-level GEMM computation, and is not necessarily optimal.
//
// Supported Architecture:
// MFMA instructions are supported on MI-class cards such as:
// gfx908 (MI-100) and gfx90A (MI-200).

#include <iostream>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "common.hpp"

///////////////
/// Helpers ///
///////////////

// The following type helps to explicitly vectorize
// variables. As a programmer, we can view the Rank
// as the number of vector registers that will be used. 
template<typename T, uint32_t Rank>
using VecT = T __attribute__((ext_vector_type(Rank)));

// Helper for vec size
template<typename T, uint32_t Rank>
static constexpr int32_t vectorSize(VecT<T, Rank>const& v)
{
    return Rank;
}

// Vector fill
// Assign a value to each vector register.
template<typename T, uint32_t Rank>
__device__ void fill_frag(VecT<T, Rank>& frag, T value)
{
    for(int i = 0; i < Rank; i++)
    {
        frag[i] = value;
    }
}

/////////////////
/// Constants ///
/////////////////

// Device wave (warp) size (MI-class architecture)
const int WAVE_SIZE = 64;

// It's up the programmer on how to map threads in a thread-block,
// so for this example we will quantize waves of 64 threads in the
// thread-block X dimension.
// 
// Thread block
// : T_BLOCK_X shall be multiple of WAVE_SIZE.
//
// For this example ASSUME:
// - One wave is 64 threads
// - Each wave will compute one BLOCK_M x BLOCK_N output block
// - Each workgroup will compute T_BLOCK_X / WAVE_SIZE x T_BLOCK_Y output blocks
const int T_BLOCK_X = 1 * WAVE_SIZE;
const int T_BLOCK_Y = 1;

// There are many MFMA builtins available in a variety of types and
// block sizes. For simplicity we will use f32 datatype with the following
// block sizes:
const int BLOCK_M = 16;
const int BLOCK_N = 16;
const int BLOCK_K = 4;

// Define some fragment types that match our measurements:
// A = BLOCK_M * BLOCK_K elements
// B = BLOCK_N * BLOCK_K elements
// C = BLOCK_M * BLOCK_N elements
// Note: using our WAVE perspective, this is the number of 
// vector registers REQUIRED to hold this much data.
// Check: for block dimensions of 16 x 16 x 4
// A should be a vector size of 1
// B should be a vector size of 1
// C should be a vector size of 4
using AFragT = VecT<float, BLOCK_M * BLOCK_K / WAVE_SIZE>;
using BFragT = VecT<float, BLOCK_N * BLOCK_K / WAVE_SIZE>;
using AccumFragT = VecT<float, BLOCK_M * BLOCK_N / WAVE_SIZE>;
using CFragT = AccumFragT;

// This MFMA builtin represents the multiply-accumulate function
// for each K-step of the the above block dimensions.
__device__ AccumFragT mfma_16x16x4_f32(AFragT aFrag, BFragT bFrag, AccumFragT accumFrag)
{
    // Nit: builtin appears to prefer scalar instead of vector[1]
    return __builtin_amdgcn_mfma_f32_16x16x4f32(aFrag[0], bFrag[0], accumFrag, 0, 0, 0);
}

// Define a load function for input A blocks:
// Size: (BLOCK_M x BLOCK_K)
// ASSUMPTION:
// - We want contiguous BLOCK_M sized column neighbors in register.
// - Data is in col_major format
// - Vector width is 1
// This means:
// - From A we will load K columns of size BLOCK_M to satisfy our input data
__device__ AFragT load_A_16x4_col_major(float const* input, int ld)
{
    // Here we want to load a 16x4 block of data.
    // Register Mapping:

    // Size              |   BLOCK_M  |   BLOCK_M   |   BLOCK_M   |   BLOCK_M    |
    // Register Element  | 0  ...  15 | 16  ...  31 | 32  ...  47 | 48  ...   63 |
    //                    ____________ _____________ _____________ ______________
    // Reg 0             |     K0     |     K1      |     K2      |     K3       |

    static constexpr uint32_t VW = vectorSize(AFragT{});
    static constexpr uint32_t Dim = BLOCK_M;

    // How would this offset change for a in a different data layout?
    auto indexOffset = threadIdx.x * VW % Dim;
    auto kOffset = threadIdx.x * VW / Dim * ld;
    auto const* fragPtr = (AFragT const*)(input + kOffset + indexOffset);
    return *fragPtr;
}

// Define a load function for input B blocks:
// Size: (BLOCK_K x BLOCK_N)
// ASSUMPTION:
// - We want contiguous BLOCK_N sized row neighbors in register.
// - Data is in row_major format
// - Vector width is 1
// This means:
// - From B we will load K rows of size BLOCK_N to satisfy our input data
__device__ BFragT load_B_16x4_row_major(float const* input, int ld)
{
    // Here we want to load a 16x4 block of data.
    // Register Mapping:

    // Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N   |   BLOCK_N    |
    // Register Element  | 0  ...  15 | 16  ...  31 | 32  ...  47 | 48  ...   63 |
    //                    ____________ _____________ _____________ ______________
    // Reg0              |     K0     |     K1      |     K2      |     K3       |

    static constexpr uint32_t VW = vectorSize(BFragT{});
    static constexpr uint32_t Dim = BLOCK_N;

    // How would this offset change for b in a different data layout?
    auto indexOffset = threadIdx.x * VW % Dim;
    auto kOffset = threadIdx.x * VW / Dim * ld;
    auto const* fragPtr = (BFragT const*)(input + kOffset + indexOffset);
    return *fragPtr;
}

// Define a load & store function for C, which is in a slightly different layout.
// Size: (BLOCK_M x BLOCK_N)
// ASSUMPTION:
// - We want contiguous BLOCK_N sized row neighbors in register.
// - Data is in row_major format
// - Vector width is 4
// This means:
// - From C we will load BLOCK_M rows of size BLOCK_N to satisfy our input data
__device__ CFragT load_C_16x16_col_major(float const* input, int ld)
{
    // Here we want to load a 16x16 block of data.
    // Register Mapping:

    // Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N   |   BLOCK_N    |
    // Register Element  | 0  ...  15 | 16  ...  31 | 32  ...  47 | 48  ...   63 |
    //                    ____________ _____________ _____________ ______________
    // Reg0              |     M0     |     M4      |     M8      |     M12      |
    // Reg1              |     M1     |     M5      |     M9      |     M13      |
    // Reg2              |     M2     |     M6      |     M10     |     M14      |
    // Reg3              |     M3     |     M7      |     M11     |     M15      |

    static constexpr uint32_t VW = vectorSize(CFragT{});
    static constexpr uint32_t Dim = BLOCK_N;

    // How would this offset change for c in a different data layout?
    auto indexOffset = threadIdx.x % Dim * ld;
    auto mOffset = threadIdx.x / Dim * VW ;
    auto* fragPtr = (CFragT*)(input + mOffset + indexOffset);
    return *fragPtr;
}

__device__ void store_C_16x16_col_major(float* output, CFragT cFrag, int ld)
{
    // Here we want to store a 16x16 block of data.
    // Register Mapping:

    // Size              |   BLOCK_N  |   BLOCK_N   |   BLOCK_N   |   BLOCK_N    |
    // Register Element  | 0  ...  15 | 16  ...  31 | 32  ...  47 | 48  ...   63 |
    //                    ____________ _____________ _____________ ______________
    // Reg0              |     M0     |     M4      |     M8      |     M12      |
    // Reg1              |     M1     |     M5      |     M9      |     M13      |
    // Reg2              |     M2     |     M6      |     M10     |     M14      |
    // Reg3              |     M3     |     M7      |     M11     |     M15      |

    static constexpr uint32_t VW = vectorSize(CFragT{});
    static constexpr uint32_t Dim = BLOCK_N;

    // Why are the offsets the same for loading?
    auto indexOffset = threadIdx.x % Dim * ld;
    auto mOffset = threadIdx.x / Dim * VW;
    auto* fragPtr = (CFragT*)(output + mOffset + indexOffset);
    *fragPtr = cFrag;
}

__global__ void sgemm_example_d(uint32_t     m,
                                uint32_t     n,
                                uint32_t     k,
                                float const* a,
                                float const* b,
                                float const* c,
                                float*       d,
                                uint32_t     lda,
                                uint32_t     ldb,
                                uint32_t     ldc,
                                uint32_t     ldd,
                                float        alpha,
                                float        beta)
{
    // Create frags
    auto fragA = AFragT{};
    auto fragB = BFragT{};
    auto fragAcc = AccumFragT{};

    // Start accumulating at 0.
    fill_frag(fragAcc, 0.0f);

    // Get the current wave's 2D grid coordinate
    auto waveGridX = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;
    auto waveGridY = (blockIdx.y * blockDim.y + threadIdx.y);

    // Scale to target C block coords (row, col) for the current wave
    auto cRow = waveGridX * BLOCK_M;
    auto cCol = waveGridY * BLOCK_N;

    // Bounds check
    if(cRow < m && cCol < n)
    {
        ///
        /// Step 1: accumulate A x B by stepping through k dimension
        ///
        for(int i = 0; i < k; i += BLOCK_K)
        {
            // Load the inputs.
            // Flatten 2D coord (row, col) into 1D, knowing:
            // A = col major, BLOCK_M x BLOCK_K
            // B = row major, BLOCK_K x BLOCK_N
            fragA = load_A_16x4_col_major(a + (cRow  + i * lda), lda);
            fragB = load_B_16x4_row_major(b + (i * ldb + cCol), ldb);

            // Matrix multiply-accumulate using MFMA units
            // Accumulation intermediate = BLOCK_M x BLOCK_N
            fragAcc = mfma_16x16x4_f32(fragA, fragB, fragAcc);
        }

        ///
        /// Step 2: Bilinear element-wise mult
        ///  D = alpha * A x B + beta * C
        ///
        auto fragC = load_C_16x16_col_major(c + (cRow + cCol * ldc), ldc);

        for(int i = 0; i < vectorSize(fragC); ++i)
        {
            fragC[i] = alpha * fragAcc[i] + beta * fragC[i];
        }

        ///
        /// Step3: Store final block result
        ///
        store_C_16x16_col_major(d + (cRow  + cCol * ldd), fragC, ldd);
    }
}

__host__ void gemm_test(uint32_t m, uint32_t n, uint32_t k, float alpha, float beta)
{
    // Bounds check
    if((m < (BLOCK_M * T_BLOCK_X / WAVE_SIZE) 
       || n < (BLOCK_N * T_BLOCK_Y) 
       || k < BLOCK_K)
       || (m % BLOCK_M || n % BLOCK_N || k % BLOCK_K))
    {
        std::cout << "Unsupported size!\n";
        return;
    }

    // Leading dimensions
    int lda = m;   // col_major
    int ldb = n;   // row_major
    int ldc = m;   // col_major
    int ldd = ldc; // col_major

    std::cout << "Initializing host data..." << std::endl;

    // Initialize input matrices
    std::vector<float> matrixA(m * k);
    std::vector<float> matrixB(k * n);
    std::vector<float> matrixC(m * n);
    // Fill outputs with NaN to catch contamination
    std::vector<float> matrixD(m * n, std::numeric_limits<float>::signaling_NaN());

    fillRand(matrixA.data(), m, k);
    fillRand(matrixB.data(), k, n);
    fillRand(matrixC.data(), m, n);

#if !NDEBUG
    std::cout << "Matrix A:" << std::endl;
    print<int, col_major>(reinterpret_cast<int*>(matrixA.data()), m, k);
    std::cout << "Matrix B:" << std::endl;
    print<int, row_major>(reinterpret_cast<int*>(matrixB.data()), k, n);
#endif // !NDEBUG

    std::cout << "Initializing device data..." << std::endl;

    // Allocate and copy device memory
    float* d_a;
    float* d_b;
    float* d_c;
    float* d_d;

    const size_t bytesA = matrixA.size() * sizeof(float);
    const size_t bytesB = matrixB.size() * sizeof(float);
    const size_t bytesC = matrixC.size() * sizeof(float);
    const size_t bytesD = matrixD.size() * sizeof(float);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
    CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

    auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
    auto gridDim  = dim3(ceilDiv(m, BLOCK_M * T_BLOCK_X / WAVE_SIZE),
                         ceilDiv(n, BLOCK_N * T_BLOCK_Y));

    std::cout << "Launching GEMM kernel..." << std::endl;
    std::cout << "TBlock(X, Y) = (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;
    std::cout << "GridDim(X, Y) = (" << gridDim.x << ", " << gridDim.y << ")" << std::endl;

    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    hipExtLaunchKernelGGL(sgemm_example_d,
                          gridDim,
                          blockDim,
                          0, // sharedMemBytes
                          0, // stream
                          startEvent, // Event start
                          stopEvent, // event stop
                          0, // flags
                          m,
                          n,
                          k,
                          d_a,
                          d_b,
                          d_c,
                          d_d,
                          lda,
                          ldb,
                          ldc,
                          ldd,
                          alpha,
                          beta);

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // GEMM flops converge to 2 * mnk
    auto gFlops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) * 1.0e-9;
    auto tFlopsPerSec = gFlops / static_cast<double>(elapsedTimeMs);

    // Echo performance
    std::cout << "BlkM, BlkN, BlkK, "
              << "MatM, MatN, MatK, "
              << "alpha, lda, ldb, "
              << "beta, ldc, ldd, "
              << "elapsedMs, Problem Size(GFlops), TFlops/s" << std::endl;

    std::cout << BLOCK_M << ", " << BLOCK_N << ", " << BLOCK_K << ", " << m << ", " << n
              << ", " << k << ", " << alpha << ", " << lda << ", " << ldb << ", " << beta << ", "
              << ldc << ", " << ldd << ", " << elapsedTimeMs << ", " << gFlops << ", "
              << tFlopsPerSec << std::endl;

    std::cout << "Validating result with reference..." << std::endl;

    // Bring kernel result back to host
    CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));

    // Setup and run reference computation
    std::vector<float> matrixD_ref(m * n, std::numeric_limits<float>::signaling_NaN());
    gemm_cpu_h<float, float, float, col_major, row_major, col_major>(m,
                                                                    n,
                                                                    k,
                                                                    matrixA.data(),
                                                                    matrixB.data(),
                                                                    matrixC.data(),
                                                                    matrixD_ref.data(),
                                                                    lda,
                                                                    ldb,
                                                                    ldc,
                                                                    ldd,
                                                                    alpha,
                                                                    beta);

#if !NDEBUG
    std::cout << "Matrix D Reference:" << std::endl;
    print<float, col_major>(matrixD_ref.data(), m, n);
    std::cout << "Matrix D Result:" << std::endl;
    print<float, col_major>(matrixD.data(), m, n);
#endif // !NDEBUG

    auto res = compareEqual<float>(matrixD.data(), matrixD_ref.data(), m * n);

    if(std::get<0>(res) == false)
    {
        std::cout << "FAILED!\n";
    }
    else
    {
        std::cout << "PASSED!\n";
    }

    std::cout << "Max relative error: " << std::get<1>(res) << std::endl;

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "Finished!" << std::endl;
}

int main()
{
    gemm_test(128, 128, 128, 2.1f, 2.1f);
    return 0;
}
