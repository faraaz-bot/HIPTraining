cmake_minimum_required(VERSION 3.10)

project( hipgraph LANGUAGES CXX )

add_executable( exercise1 exercise1.cpp)
add_executable( exercise2 exercise2.cpp)
set_property(TARGET exercise1 PROPERTY CXX_STANDARD 17 )
set_property(TARGET exercise2 PROPERTY CXX_STANDARD 17 )

add_executable( example_from_slides slides_demo/example_from_slides.cpp)
set_property(TARGET example_from_slides PROPERTY CXX_STANDARD 17 )
