def stencil_memory_index(indices, dimensions):
    if len(indices) != len(dimensions):
        raise ValueError("Dimension mismatch")
    factor = 1
    res = 0
    for i, d in zip(reversed(indices), reversed(dimensions)):
        res += i * factor
        factor *= d
    return res

def stencil_distance(a, b, dimensions):
    return abs(stencil_memory_index(a) - stencil_memory_index(b))

