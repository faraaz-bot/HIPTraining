## Environment, configuration, and compilation:

# Set up your cmake environment to detect ROCm:
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake/

# Configure hipcc:
cmake -DCMAKE_CXX_COMPILER=hipcc ..

# Configure amd-clang:
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++  ..

# To build with Nvidia Target: Configure hipcc-nvcc
Upgrade to a later cmake (something like 3.24 or later; may need to install libssl-dev on ubuntu).
Grab the source files, and then build and install with:
cmake -DCMAKE_INSTALL_PREFIX=${HOME}/cmake .. && make && make install
export PATH=${HOME}/cmake/bin:${PATH}

export HIP_PLATFORM=nvidia
cmake -DCMAKE_CXX_COMPILER=hipcc ..

# Compile with make:
export MAKEFLAGS=-j$(nproc)
make


## Exercises:

# Exercise 1:
Construct an average moving filter of the provided signal.
1. Your average moving filter function should take in various values for 'k'
2. Your average moving filter should ignore data that it 'out of bounds'.
3. Compare the results betweeen the raw signal and the filtered signal using k = 5, 25, 50
4. Utilizie the python script plot.py to visualize your results.

# Exercise 2:
Compute the numerical integration of sqrt(1-x^2) from -1 to 1
1. Generate a divice function to apply sqrt(1-x^2)
2. Apply the Trapezoidal rule to calculate discrete areas 
3. Reduce the discrete areas to find the value of the discrete integral
4. Multiply the result by 2, what's the output value?
