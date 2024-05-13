# Floating Point Arithmetic Exercise

## Introduction

An exercise exploring the accuracy of floating point arithmetic using HIP kernels.

The sum of inverse squares (aka the Basel Problem) is shown to converge to $\frac{(\pi)^2}{6}$, i.e.: $$\sum_{x=1}^\infty \frac{1}{x^2} = 1 + \frac14 + \frac19 + \frac{1}{16} + ...$$
In this exercise we will use various methods to calculate this to the limits of single precision accuracy.

First, calculate this result as a double-precision value. We will use this as a reference when calculating the single-precision approximation. Next, we will create HIP kernels to calculate the summation in single-precision up to *n* elements. Use the following steps as a guideline:
- Use a single thread block with *t* threads
- Have each thread compute a partial sum of $\lceil \frac nt \rceil$ and store this into memory
- Reduce this array of partial sums to get the final approximation

## Exercise 1: Single Thread Approximation

Use a single thread on the device to compute the series. Use an increasing number of elements and to compare the results.
a) Plot the error vs. the number of elements used
b) Why does the error stop decreasing?
c) At what value of *n* does the error stop decreasing? Why?

## Exercise 2: Multithread Approximation

Modify your code to use multiple threads. Try out an increasing number of threads to see the behaviour.
a) How does the relative error compare to using a single thread? Why?

## Exercise 3: Reverse Summation Single Thread Approximation

Reverse the order of the summation, i.e.: $$\sum_{x=1}^{n-1} \frac{1}{(n-x)^2} = \frac{1}{n^2} + \frac{1}{(n-1)^2} + \frac{1}{(n-2)^2} + ... + 1$$

Repeat exercise 1 (single thread) using the reversed summation. Explain the new results compared to the original results.

## Exercise 4: Reverse Summation Multithread Approximation

Repeat exercise 2 (multithread) using the reversed summation. Explain the new results compared to the original results.

## Notes

You have been provided with template files **ex1_single_thread.cpp** and **ex2_multithread.cpp**. Feel free to use these as a baseline in the exercise, or start from scratch yourself.