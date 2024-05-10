cmake_minimum_required(VERSION 3.10)

project( transpose LANGUAGES CXX )

add_executable( transpose transpose_main.cpp)
set_property(TARGET transpose PROPERTY CXX_STANDARD 17 )
