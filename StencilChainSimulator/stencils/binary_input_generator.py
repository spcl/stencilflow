import numpy as np

# general parameters
_N = 10*10*10
_GENERATOR = "incr"  # {const, random, incr}
_FILENAME = "inB"
_DATA_TYPE = "csv" # {csv, bin, dat}
# random
_LOWER_BOUND = 0.0
_UPPER_BOUND = 10.0
# constant
_CONSTANT = 1.0

# generate data
if _GENERATOR == "random":
    # random data
    data = np.random.uniform(low=_LOWER_BOUND, high=_UPPER_BOUND, size=_N)
elif _GENERATOR == "const":
    # constant value
    data = np.array([_CONSTANT]*_N)
elif _GENERATOR == "incr":
    # increasing value
    data = np.arange(0.0, _N, dtype="float64")
else:
    print("Generator {} not supported. Exit.".format(_GENERATOR))
    exit()

# write data to file
filename = _FILENAME + "." + _DATA_TYPE

if _DATA_TYPE == "csv":
    np.savetxt(filename, [data], delimiter=",")
elif _DATA_TYPE == "bin" or _DATA_TYPE == "dat":
    data.astype('float64').tofile(filename)
