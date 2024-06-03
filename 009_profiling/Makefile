CXX=hipcc

dir_guard=@mkdir -p $(@D)

default: build/vcopy

build/vcopy.cpp:
	$(dir_guard)
	wget https://github.com/ROCm/omniperf/raw/main/sample/vcopy.cpp -O $@

build/%: build/%.cpp
	$(dir_guard)
	$(CXX) -o $@ -c $<

clean:
	rm -rf ./build
