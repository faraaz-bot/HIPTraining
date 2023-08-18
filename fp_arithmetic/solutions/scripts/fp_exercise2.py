#!/usr/bin/env python3

"""Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import subprocess
import numpy as np
import math

parser = argparse.ArgumentParser(description='Used with ex1_single_thread.cpp solution. Iteravely calls ' +
                                              'solution with an increasing number of elements (n) and plots '+
                                              'the relative error against n.')
parser.add_argument('-x', '--executable', type=str, help='Name of the solution executable. Must take in three arguments: ' +
                                                         '# elements, # threads, and y/n for csv output.')
parser.add_argument('-o', '--output_file', default='ex2_plot.png', type=str, help='Name of output file for plot.')
parser.add_argument('--start_num', type=int, default=1024, help='Start # of elements of series.')
parser.add_argument('--end_num', type=int, default=16777216, help='End # of elements of series.')
parser.add_argument('--increment', type=int, default=8192, help='# of elements to jump in each iteration.')
parser.add_argument('--thread_list', type=str, default='1,100,1000', help='A comma separated list of values denoting ' +
                                                                             'the number of threads to use.')

args = parser.parse_args()

if(args.start_num <= 0):
    print("--start_num must be > 0.")
    exit(-1)
if(args.end_num < args.start_num):
    print("--end_num must be >= start_num")
    exit(-1)
if(args.increment <= 0):
    print("--increment must be > 0")
    exit(-1)

thread_list = args.thread_list.split(',')
print(thread_list)
plt.figure()
plt.ylabel('log(relative error)')
plt.xlabel('log(# of elements)')
colors = 'bgrcmykw' # matplotlib colors
c_idx = 0

for t in thread_list:
    print("With t =", t)
    element_vals = []
    error_vals = []
    t = int(t)
    for i in range(args.start_num, args.end_num, args.increment):
        element_vals.append(math.log(i, 10))
        process = subprocess.Popen([args.executable, str(i), str(t), 'y'], stdout=subprocess.PIPE, text=True)
        out, err = process.communicate()
        lines = out.splitlines()
        titles = lines[0].split(',')
        idx = titles.index('relative_error')
        val = float(lines[1].split(',')[idx])
        error_vals.append(math.log(val, 10))

    
    plt.plot(element_vals, error_vals,
             color=colors[c_idx],
             label=str(t) + " thread(s)")
    c_idx += 1


plt.legend()
plt.savefig(args.output_file)
