Prequisites
===========

To run the code, the following software must be available:
- Python 3.6.x or newer.
- The `virtualenv` module (installed with `pip install virtualenv`).
- A C++17-capable compiler (e.g., GCC 7.x or Clang 6.x).
- The Intel FPGA OpenCL SDK (tested with 18.1.1 and 19.1)

Setup
=====

Sourcing the script `setup_virtualenv.sh` will setup a virtualenv with all the
modules required to run StencilFlow, including the relevant version of DaCe:

```bash
source setup_virtualenv.sh
```

Running
=======

To run the end-to-end flow on an input JSON file, the executable
`bin/run_program.py` can be used. Example usage:

```bash
bin/run_program.py test/testing/jacobi3d_32x32x32_8itr_8vec.json emulation -compare-to-reference
```

This will compile the FPGA kernel for Intel's emulation flow, execute it, build
a reference CPU program, run both, and verify that the results match.

The generated program will be located in `.dacecache/<kernel name>`, with the
kernel source files themselves in:

```bash
.dacecache/<kernel name>/src/intel_fpga/device
```

Tests
=====

The repository ships with a number of tests that verify various aspects of
functionality. These can be run with:

```bash
test/test_stencil.py
```

It is a known issue that launching Intel FPGA kernels can sometimes fail
sporadically, seemingly due to file I/O issues. Running individual programs
should never fail.
