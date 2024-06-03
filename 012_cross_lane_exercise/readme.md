# Welcome to the Cross Lane coding example!

## build cmd:  
`hipcc cross_lane_example.cpp -o cross_lane_example -DNDEBUG=1 -O3 --save-temps`

## run cmd:  
`./cross_lane_example <0-3>`

## Exercise instructions:
1. Compile and run the code for the problem described in the code. Make sure to read all the documentation for motivation and explanation.

2. Read the goals section and implement LDS, swizzle, dpp and bpermute methods. Ensure they pass validation.

3. Observe the performance of each of the methods and cross-reference with your learnings from cross-lane operations training module.

4. Increase the number of trial runs to 1000. How does this affect your performance observations?

## Food for thought:

1. Consider changing the datatype to double (8 byte elements). Would your methods still work? Why?

    Possible Answer: Yes. Under the hood, the compiler will separate the upper and lower 32 bits of each element into two registers in the same lane. The _impl methods
    in the exercise account for this by performing the data movement for each b32 half. Following this operation, there is still the 64bit add of 1 which takes place and
    should be accounted correctly for successful validation. 

2. Consider changing the datatype to f16 (2 byte elements). Would your methods still work? Why? How would you adjust for this datatype?

    Possible answer: For this particular challenge where we need to swap even elements with odd elements, cross-lane operations are not suitable. The reason is that
    the data were are interested in (even/odd neighbours) are in the same lane! There are other `__builtin` choices such as `__builtin_amdgcn_perm` which can permute data
    within each lane that can perform the desired swapping, then you would follow up with the add of +1. This DOES NOT however preclude f16 data from being used with
    cross-lane operations, just the data locality for this particular challenge is not suitable. If the challenge was to swap groups of 2 x f16 elements with
    their neighbours then again the operation requires cross-lane data movement and we can use cross-lane ops.

3. How would you go about porting this code to RDNA cards? How would this affect your wave size and `__builtin` choices?

    Possible answer: RDNA cards have slightly different support in terms of capabilities of cross lane operations. All of the instructions are present in RDNA, however 
    RDNA register sizes are by default 32 elements wide. This limits some of the capabilities of each of the `__builtin`s - full descriptions are available in ISA
    documentation. Notwithstanding, if you write code designed to be supported for multiple different architecture, including both CDNA and RDNA, you can expect to differentiate code paths based on architecture symbols set by the compiler. E.g. `__gfx908__` or `__gfx1100__`.

4. Check out rocWMMA project at https://github.com/ROCm/rocWMMA. This library wraps all the above details into portable utilities that support CDNA and RDNA
    architectures!