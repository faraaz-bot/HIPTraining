CXX=/opt/rocm/hip/bin/hipcc

CXXFLAGS=-Ofast -I/opt/rocm/hip/include/ -march=native

ALL = ex0_errors ex1_vadd ex2_vaddblock ex3_vadd4by ex4_vaddgrid ex5_matadd

all: $(ALL)

$(TARGET): $(TARGET:.cpp)
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o $(ALL)
