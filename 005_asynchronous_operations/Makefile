CXX=hipcc
FLAGS=-O3 -std=c++14
INC=-I/opt/rocm/include
LIB=-lroctx64

dir_guard=@mkdir -p $(@D)

default: build/ex1 build/ex2

build/car1.o: car1.cpp car.hpp
	$(dir_guard)
	$(CXX) $(FLAGS) -o build/car1.o -c car1.cpp $(INC)

build/car2.o: car2.cpp car.hpp
	$(dir_guard)
	$(CXX) $(FLAGS) -o build/car2.o -c car2.cpp $(INC)

build/ex1.o: ex1.cpp
	$(dir_guard)
	$(CXX) $(FLAGS) -o build/ex1.o -c ex1.cpp $(INC)

build/ex1: build/ex1.o build/car1.o
	$(dir_guard)
	$(CXX) $(FLAGS) -o build/ex1 build/ex1.o build/car1.o $(LIB)

build/ex2.o: ex2.cpp
	$(dir_guard)
	$(CXX) $(FLAGS) -o build/ex2.o -c ex2.cpp $(INC)

build/ex2: build/ex2.o build/car2.o
	$(dir_guard)
	$(CXX) $(FLAGS) -o build/ex2 build/ex2.o build/car2.o $(LIB)

clean:
	@rm -rf build/*.o build/ex1 build/ex2
