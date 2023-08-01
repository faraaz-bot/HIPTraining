CXX=/opt/rocm/hip/bin/hipcc

CXXFLAGS=-Ofast -I/opt/rocm/hip/include/ -march=native

ALL = numerical_integration 

all: $(ALL)

$(TARGET): $(TARGET:.cpp)
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o $(ALL)
