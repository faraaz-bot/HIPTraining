Welcome to the MFMA coding example!

build cmd:
hipcc simple_hgemm.cpp -o simple_hgemm -DNDEBUG=1 

run cmd:
./simple_hgemm

Exercise instructions:
1. Compile and run the code for the problem described in the code. Make sure to read all the documentation for motivation and explanation.

2. The simple_hgemm example is designed for a BlockMNK = (16, 16, 16) instruction. Modify it to run successfully with BlockMNK = (32, 32, 8). Ensure it passes validation.

3. Modify the simple_hgemm example to support another datatype, say F32. For this case, use the instruction for BlockMNK = (16 x 16 x 4). Ensure it passes validation.

4. The simple_hgemm example was designed for data layoutsABC = (col_major, row_major, col_major). Re-write the sample to accept layoutsABC = (row_major, row_major, row_major). Do this without using LDS. Ensure it passes validation.

Food for thought:

1. Consider modifying the thread-block to accomodate up to 4 waves (256 threads). How would your mapping change / what are the opportunities for data sharing and optimization? Can we use LDS somehow?

    Possible Answer: The current mapping assumes that thread Id's run from 0 - 63. In a threadblock of 256, the thread Id's can be >= 64 < 256. Thread id's should mod 64 such that they fall within the range of 0 - 63 again. We also mustn't forget to update the global offsets for target C blocks with the number of waves in the x, y directions.
    Waves in the same workgroup can share loading inputs for A and B. Notice that waves processing blocks in the same grid row will use the same 'A' input data stepping through K.
    Conversely, waves processing blocks in the same grid column will use the same 'B' input data stepping through K. As long as we synchronize wave steps through the K dimension,
    waves can share the loading burden for A and B inputs and store them into LDS for all to use. The larger the macro tile of the entire workgroup, the more opportunity we have
    for data sharing.

2. Consider modifying the sample to have each wave process more than one block of output, say BLOCKS_X x BLOCKS_Y. How would your __builtin choices change and how would this affect block loading, K-accumulation loop and mapping?

    Possible answer: You could do this in a few different ways. You could choose to try and process the entire larger tile BLOCKS_X x BLOCKS_Y by choosing a larger MFMA instruction
    (e.g. choose 32x32 instead of 16x16) and modify the cbsz, blgp flags to process a the larger tile. OR, you might think about using discrete block sizes
    and iterating over BLOCKS_X x BLOCKS_Y in a specified order. OR you might choose instructions with multiple output blocks and accumulate them over shorter K iterations. There
    are many ways to do this, however the sizes of BLOCKS_X x BLOCKS_Y will dictate what is possible, given the limitations of each of the methods above. Block loading of course would have to be adjusted such that you align your inputs to specific blocks, and they must be mirrored in A and B. The effects on K will only change the number of iterations
    for each MFMA instruction. There are many tuning opportunities here to make sure that the memory xfer and mfma calculations are balanced in a way that we can hide latency effectively use resources. Moreover, mapping data in longer contiguous dimensions has great advantages in accelerating relatively slow loads from global memory.

3. How would you go about porting this code to RDNA cards? How would this affect your wave size and __builtin choices? How would this change load / store patterns?

    Possible answer: RDNA cards have slightly different layouts in A/B/C, however the principle is the same. The instructions are named WMMA instead of MFMA. The differences are best viewed using the matrix instruction layout calculator found here: https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator. The main gist is that A and B inputs must be mirrored in the upper / lower register halves. As a result, RDNA can only support 16x16 sized blocks (wave 32). The C layout is similar to CDNA for f32 compute type, however if the accum type is 16-bit, then the output is NOT packed. RDNA also supports both wave 32 / wave 64 versions of the instructions which will affect your design of your implementation. Notwithstanding, if you write code designed to be supported for multiple different architecture, including both CDNA and RDNA, you can expect to differentiate code paths based on architecture symbols set by the compiler. E.g. __gfx908__ or __gfx1100__

4. Check out rocWMMA project at https://github.com/ROCmSoftwarePlatform/rocWMMA. This library wraps all the above details into a simple API that supports both MFMA and WMMA instructions!