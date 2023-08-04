CXX=hipcc
FLAGS=-O3 -std=c++14
INC=-I/opt/rocm/include
LIB=-lroctx64

default: ex1 ex2

car.o: car.cpp car.hpp
	$(CXX) $(FLAGS) -c car.cpp $(INC)

ex1.o: ex1.cpp
	$(CXX) $(FLAGS) -c ex1.cpp $(INC)

ex1: ex1.o car.o
	$(CXX) $(FLAGS) -o ex1 ex1.o car.o $(LIB)

ex2.o: ex2.cpp
	$(CXX) $(FLAGS) -c ex2.cpp $(INC)

ex2: car.o ex2.o car.o
	$(CXX) $(FLAGS) -o ex2 ex2.o car.o $(LIB)

clean:
	rm -rf *.o ex1 ex2
