Welcome to the MFMA coding example!

build cmd:
hipcc simple_sgemm.cpp -o simple_gemm -DNDEBUG=1 

run cmd:
./simple_gemm

Exercise instructions:
1. Compile and run the code for the problem described in the code. Make sure to read all the documentation for motivation and explanation.

2. The previous example is designed for a BlockMNK = (16, 16, 4) instruction. Modify the example to run successfully with BlockMNK = (32, 32, 2). Ensure it passes validation.

3. The previous example was designed for data layoutsABC = (col_major, row_major, col_major). Re-write the sample to accept layoutsABC = (row_major, row_major, row_major). Do this without using LDS. Ensure it passes validation.

4. Consider modifying the sample to have each wave process more than one block of output, say BLOCKS_X x BLOCKS_Y. How would your __builtin choices change and how would this affect block loading, K-accumulation loop and mapping?

5. Consider modifying the thread-block to accomodate up to 4 waves (256 threads). How would your mapping change / what are the opportunities for data sharing and optimization? Can we use LDS somehow?

6. Consider modifying the sample for another datatype, say F16. How would this affect your approach to loading / storing and offset mapping?

7. How would you go about porting this code to RDNA cards? How would this affect your wave size and __builtin choices? How would this change load / store patterns?

8. Check out rocWMMA project at https://github.com/ROCmSoftwarePlatform/rocWMMA. This library wraps all the above details into a simple API that supports both MFMA and WMMA instructions!