# Latency and Occupancy HIP Training Exercises

These exercises use Boost program options to parse command-line
arguments.  On Ubunt, this is available in the
   libboost-program-options-dev
package.  You can pass arguments via bash using the following notation,
for example:
```sh
  ./ex1_latency --N $(( 2 ** 20 ))
```
which sets N to be 2 to the power of 20.

For exercise 2 HIP API calls, see

https://docs.amd.com/projects/HIP/en/docs-5.0.0/doxygen/html/group___occupancy.html


## Exercise 1: The empty kernel

Measure empty kernel launch latency via the empty kernel; see how this
is affected by the number of waves, threa-block size, and LDS use.


## Exercise 2: Occupancy computation and measurement

We will use the simple vector-add kernel to explore the functionality
of occupancy via kernel launches and the HIP API.

- Write a vector-addition kernel and verify the results on the host.

- Use the HIP API to get the kernel's occupancy given for a given
  LDS usage, problem size, and thread-block size.

- Use the HIP API to get the optimal thread-block and grid-block
  parameters for maximizing occupancy.

- Use the HIP API to determine the maximum allowable thread block
  size.  What happens when you ask for more than the maximum allowable
  threads per block?  (What should happen?)

- Set the launch bounds for the kernel.  What happens when you launch
  the kernel with a different number of threads per block?  (What
  should happen?)

- Compile using `hipcc --save-temps` and then look at .s file that has
  the gfx model: does the maximum occupancy agree with the API report?
  Does it use the expected number of scalar and vector registers?  You
  can just uncomment the command in CMakeLists.txt to enable this
  flag.
  
- Compile using `-Rpass-analysis=kernel-resource-usage` and observe
  the occupancy info at compile time!

## Exercise 3: vector addition, but we add extra LDS to reduce occupancy

Using the results from exercise 2, determine the performance
characteristics as a function of occupancy for the vector-add kernel.
It's a good idea to still validate the computation with a host
version, just in case.
