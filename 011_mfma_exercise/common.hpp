#ifndef SAMPLES_COMMON_HPP_
#define SAMPLES_COMMON_HPP_

#include <cmath>
#include <utility>
#include <vector>
#include <cstdlib>
#include <type_traits>
#include <mutex>
#include <limits>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(status)                   \
    if(status != hipSuccess)                      \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(status),        \
                status,                           \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
__host__ static constexpr intT1 ceilDiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}


struct row_major{};
struct col_major{};

// Host matrix data random initialization
template <typename DataT>
__host__ static inline void fillRand(DataT* mat, uint32_t m, uint32_t n)
{
    auto randInit = []() {
        srand(time(0));
        return 0u;
    };

    static auto init = randInit();
#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
        auto rando = rand() % 5u;
        for(int j = 0; j < n; j++)
        {
            // Assign random integer values within 0-64, alternating
            // sign if the value is a multiple of 3
            auto value     = (rando + j) % 5u;
            mat[i * n + j] = ((value % 3u == 0u) && std::is_signed<DataT>::value)
                                 ? -static_cast<DataT>(value)
                                 : static_cast<DataT>(value);
        }
    }
}

template <typename DataT>
__host__ static inline void fillSeq(DataT* mat, uint32_t m, uint32_t n)
{

#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; j++)
        {
            int32_t index = j * m + i;
            mat[index] = *(reinterpret_cast<float*>(&index));
        }
    }
}

// Host GEMM validation
template <typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD = LayoutC>
__host__ void gemm_cpu_h(uint32_t       m,
                         uint32_t       n,
                         uint32_t       k,
                         InputT const*  a,
                         InputT const*  b,
                         OutputT const* c,
                         OutputT*       d,
                         uint32_t       lda,
                         uint32_t       ldb,
                         uint32_t       ldc,
                         uint32_t       ldd,
                         ComputeT       alpha,
                         ComputeT       beta)
{
    auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
    auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

    auto aIndex = std::is_same<LayoutA, row_major>::value ? rowMjr : colMjr;
    auto bIndex = std::is_same<LayoutB, row_major>::value ? rowMjr : colMjr;
    auto cIndex = std::is_same<LayoutC, row_major>::value ? rowMjr : colMjr;
    auto dIndex = std::is_same<LayoutD, row_major>::value ? rowMjr : colMjr;

#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
#pragma omp parallel for
        for(int j = 0; j < n; ++j)
        {
            ComputeT accum = static_cast<ComputeT>(0);
            for(int h = 0; h < k; ++h)
            {
                accum += static_cast<ComputeT>(a[aIndex(i, h, lda)])
                         * static_cast<ComputeT>(b[bIndex(h, j, ldb)]);
            }
            d[dIndex(i, j, ldd)] = static_cast<OutputT>(
                alpha * accum + beta * static_cast<ComputeT>(c[cIndex(i, j, ldc)]));
        }
    }
}

// Element-wise comparison
template <typename DataT>
__host__ std::pair<bool, double>
         compareEqual(DataT const* a, DataT const* b, uint32_t size, double tolerance = 10.0)
{
    bool   retval             = true;
    double max_relative_error = 0.0;

    // Some types don't have direct conversion to double.
    // Convert to float first then to double.
    auto toDouble = [](DataT const& val) { return static_cast<double>(static_cast<float>(val)); };

    bool       isInf = false;
    bool       isNaN = false;
    std::mutex writeMutex;

#pragma omp parallel for
    for(int i = 0; i < size; ++i)
    {
        auto valA = a[i];
        auto valB = b[i];

        auto numerator = fabs(toDouble(valA) - toDouble(valB));
        auto divisor   = fabs(toDouble(valA)) + fabs(toDouble(valB)) + 1.0;

        if(std::isinf(numerator) || std::isinf(divisor))
        {
#pragma omp atomic
            isInf |= true;
        }
        else
        {
            auto relative_error = numerator / divisor;
            if(std::isnan(relative_error))
            {
#pragma omp atomic
                isNaN |= true;
            }
            else if(relative_error > max_relative_error)
            {
                const std::lock_guard<std::mutex> guard(writeMutex);
                // Double check in case of stall
                if(relative_error > max_relative_error)
                {
                    max_relative_error = relative_error;
                }
            }
        }

        if(isInf || isNaN)
        {
            i = size;
        }
    }

    auto eps = toDouble(std::numeric_limits<DataT>::epsilon());
    if(isInf)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DataT>::infinity();
    }
    else if(isNaN)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DataT>::signaling_NaN();
    }
    else if(max_relative_error > (eps * tolerance))
    {
        retval = false;
    }

    return std::make_pair(retval, max_relative_error);
}

template <typename DataT, typename LayoutT>
__host__ static inline void
print(DataT const* mat, uint32_t m, uint32_t n, std::ostream& stream = std::cout)
{
    auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
    auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

    auto index = std::is_same<LayoutT, row_major>::value ? rowMjr : colMjr;
    auto ld    = std::is_same<LayoutT, row_major>::value ? n : m;

    for(int i = 0; i < m; ++i) // row
    {
        stream << "[ ";
        for(int j = 0; j < n; ++j) // col
        {
            // (Row, col)
            stream << mat[index(i, j, ld)] << "\t";
        }
        stream << "]\n";
    }
    stream << "\n";
}

#endif // SAMPLES_COMMON_HPP_