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

cmake_minimum_required( VERSION 3.16 )


project( hip_simple_kernels LANGUAGES CXX )
set( CMAKE_CXX_STANDARD 14 )

find_package(HIP CONFIG)

enable_language(HIP)
set(CMAKE_HIP_LINKER_PREFERENCE_PROPAGATES 0)



if( NOT HIP_FOUND )
  find_package( hip REQUIRED )
endif()

set( example_list
  ex0_errors
  ex2_vaddblock
  ex4_vaddgrid
  ex1_vadd
  ex3_vadd4by
  ex5_matadd )

message("${CMAKE_CXX_COMPILER_ID}")
message("${CMAKE_HIP_COMPILER_ID}")

if( CMAKE_CXX_COMPILER MATCHES ".*hipcc")
  message("hipcc eh?")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "ROCMClang")
  message("w00t")
endif()

foreach( example ${example_list} )

  add_executable( ${example} ${example}.cpp )

  # $HIP_INCLUDE_PATH gets set in hipcc, not by ROCMClang itself.
    
  # If we are using hip-clang, we need to get the hip library.
  target_compile_definitions( ${example} PRIVATE __HIP_PLATFORM_AMD__ )
  target_link_libraries( ${example} hip::device hip::host )
  #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-arch=gfx90c:xnack-")
  
endforeach( )
