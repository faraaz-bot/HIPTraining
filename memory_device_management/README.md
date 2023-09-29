# Requirements

- Make sure you have ROCm installed in your environment. The typical location for the ROCm installation is `/opt/rocm`
- Make sure that the path to ROCm binaries (eg. `/opt/rocm/bin`) is added to your `PATH` variable
- Make sure you have GPUs visible in your environment. You can use `rocminfo` command and look for entries with "gfx" in the name.

# Compilation steps for exercises
* `hipcc <exercise.cpp> -o <exercise>`

# Run steps for exercises
* `./<exercise>`
