# Environment, configuration, and compilation:

## Set up your cmake environment to detect ROCm:
```sh
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake/
```

Upgrade to a later cmake (something like 3.21 or later; may need to
install libssl-dev on ubuntu).
Grab the source files, and then build and install with:
```sh
cmake -DCMAKE_INSTALL_PREFIX=${HOME}/cmake .. && make && make install
export PATH=${HOME}/cmake/bin:${PATH}
```

When using the hip-programming language in cmake, you can just compile with

## ROCm-backend configuration:
```sh
cmake ..
```

If you're using the older cmake, which doesn't support the hip
language, then you will need to manually specify a C++ compiler which
understands hip:

## ROCm-backend configuration:
```sh
cmake -DCMAKE_CXX_COMPILER=amdclang++ ..
```

One can also use hipcc instead of amdclang++, though hipcc is in the
process of being deprecated.

## CUDA-backend configuration:

Specify nvidia platform support:
```sh
export HIP_PLATFORM=nvidia

cmake -DCMAKE_CXX_COMPILER=nvcc ..
```

You can also use hipcc, which wraps nvcc:  
```sh
cmake -DCMAKE_CXX_COMPILER=hipcc ..
```

Or normal clang or g++, which will call nvcc for the hip code:  
```sh
cmake -DCMAKE_CXX_COMPILER=clang ..
```

## Compile with `make`:
```sh
export MAKEFLAGS=-j$(nproc)
make
```


# Exercises:

## Exercise 1:
Determine the error-handling behaviour for the HIP runtime by:
- failing an allocation on the device
- failing a kernel launch

## Exercise 2: Vector addition kernel

Compute the sum of two vectors of arbitrary length for general
thread-block sizes. Verify the result by comparing with a CPU-based
computation.

## Exercise 3: Vector addition kernel more ops per thread

Compute the sum of two vectors of arbitrary length for general
thread-block sizes by having each thread compute 4 terms.  Organize the
threads in two ways:
 - by having each thread treat 4 contiguous terms.  For example, thread 0
 computes terms {0, 1, 2, 3}, thread 1 computes terms {4, 5, 6, 7}, etc.
  - by having contiguous threads treat contiguous terms.  For example,
 thread 0 computes terms {0, N/4, 2N/4, 3N/4}, and thread 1 computes
 terms {1, N/4 + 1, 2N/4 + 1, 3N/4 + 1}, etc.

Verify the result by comparing with a CPU-based computation.

## Exercise 4:
Matrix addition kernel - 1 thread per element using 2D grid of 2D thread blocks

- Use a 2D thread organization to perform a matrix addition.  Verify the
result by comparing with a CPU-based computation.
