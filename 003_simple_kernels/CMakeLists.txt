# #############################################################################
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# #############################################################################

cmake_minimum_required( VERSION 3.21 )

project( hip_simple_kernels LANGUAGES CXX HIP )

set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

if( $ENV{HIP_PLATFORM} MATCHES "nvidia" )
  message( STATUS "HIP_PLATFORM: nvidia" )
else()
  message( STATUS "HIP_PLATFORM: amd" )
endif()

find_package( hip REQUIRED CONFIG PATHS /opt/rocm/lib/cmake/hip/ )

set( example_list
  ex1_errors
  ex2_vadd
  ex3_vadd4by
  ex4_matadd )

foreach( example ${example_list} )
  add_executable( ${example}  ${example}.cpp )
  
  # Since the files don't end in .hip (or .cu), we need to tell the
  # compiler that it should look for GPU kernels.
  set_source_files_properties(${example}.cpp PROPERTIES LANGUAGE HIP)

  if( NOT CMAKE_CXX_COMPILER MATCHES ".*hipcc")
    target_link_libraries( ${example} hip::device hip::host )
  endif()
endforeach()
