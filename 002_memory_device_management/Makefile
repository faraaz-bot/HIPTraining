CXX=hipcc
FLAGS=-O3 -std=c++14
INC=-I/opt/rocm/include

dir_guard=@mkdir -p $(@D)

default: build/ex1 build/ex2 build/ex3 build/ex4

build/%: %.cpp
	$(dir_guard)
	$(CXX) $(FLAGS) -o $@ -c $< $(INC)
