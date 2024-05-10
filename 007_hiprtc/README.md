The goal of this exercise is to convert a kernel built with hipcc to
one built with hipRTC.

This is a CMake project.  Configure and build it with:

```sh
  mkdir build
  cd build
  cmake -DCMAKE_CXX_COMPILER=hipcc ..
  make
```
  
Then run it with:

```sh
  ./transpose
```
  
The program will:

* Print an 11x20 array of floats to stdout

* Launch a kernel to transpose it on the GPU

* Print the 20x11 transposed array

Your goal is to build the kernel with hipRTC and launch it to do the
same work.

Some notes:

* `do_transpose` is the C++ function responsible for running the
  transpose on the GPU.  You can make your changes here.  You should
  not have a need to change `main`.

* The kernel is already in a separate file.  Your solution can read
  that file into a string if it's easier to do that.
  
* It's currently a function template.  You can specialize it for a
  single type and remove the template so long as it still does the
  same job.

* `hipModuleGetFunction` will fail if the kernel name you specify is
  wrong.  Beware of C++ name mangling, and give the kernel function C
  linkage if in doubt.

* Keep in mind that there is no type safety around kernel arguments
  when you launch the kernel!  Double check the type and size of each
  parameter to ensure you're constructing the argument buffer
  correctly.

