import collections
import functools
import operator

# Value inserted when the output is junk and should not be used.
# This is useful for debugging, as it can be used to trace where a wrong read
# happened.
JUNK_VAL = -100000


def dim_to_abs_val(input, dimensions):
    """Compute scalar number from independent dimension unit."""
    vec = [
        functools.reduce(operator.mul, dimensions[i + 1:], 1)
        for i in range(len(dimensions))
    ]
    return functools.reduce(operator.add, map(operator.mul, input, vec), 0)


def make_iterators(dimensions, halo_sizes=None, parameters=None):
    def add_halo(i):
        if i == len(dimensions) - 1 and halo_sizes is not None:
            return " + " + str(-halo_sizes[0] + halo_sizes[1])
        else:
            return ""

    if parameters is None:
        return collections.OrderedDict([("i" + str(i),
                                         "0:" + str(d) + add_halo(i))
                                        for i, d in enumerate(dimensions)])
    else:
        return collections.OrderedDict([(parameters[i],
                                         "0:" + str(d) + add_halo(i))
                                        for i, d in enumerate(dimensions)])
