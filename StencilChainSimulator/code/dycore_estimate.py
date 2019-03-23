import operator
import helper
from kernel_chain_graph import KernelChainGraph
from functools import reduce
from typing import List


'''
    assumptions:
'''

'''
datatype: at the moment, we assume float (32bit) are precise enough
'''
_SIZEOF_DATATYPE: int = 4  # in bytes

'''
we assume that we can iterate over the smaller dimension by transposition of all data arrays, if not, change this to
[64, 1024, 1024] (would lead to a factor 16 of buffer space growth
'''
_DIMENSIONS: List[int] = [1024, 1024, 64]  # longitude x latitude x altitude

'''
estimate for critical path length (#cycles):

Stencil Chains:
    fastwaves:     [3, 1, 54]
    diffusion_min: [3, 0, 24]
    advection_min: [7, 4, 26]
    
-> use mean of the three: [4, 2, 35]
'''
_MEAN_CRITICAL_PATH_KERNEL: List[int] = [4, 2, 35]

'''
 FPGA clock frequency: 200Mhz
'''
_FPGA_CLOCK_FREQUENCY: int = 2e8

'''
 available FGPA fast memory (Stratix 10): 25MB
'''
_FPGA_FAST_MEMORY_SIZE: int = 25e6

'''
 available bandwidth to slow memory (Stratix 10): 86.4GB/s
'''
_FPGA_BANDWIDTH: int = 86.4e9

'''
 we assume that as soon as the pipeline is saturated, we can produce one result per cycle
'''
_CYCLES_PER_OUTPUT: int = 1


def do_estimate():
    """
    This function is meant to programmatically go through the current 'best-estimate' calculation of our model.
    :return: None
    """

    print("COSMO dynamical core buffer size estimate report:\n")

    # instantiate the dummy-dycore to get full analysis
    chain = KernelChainGraph("input/dycore_upper_half_3.json")

    '''
     assumption: since we implemented ~1/2 of the dycore in the dummy input file, we assume the critical path is 2x longer
    '''
    _DYCORE_CRITICAL_PATH_LENGTH = 2*chain.compute_critical_path()

    # compute total critical path
    critical_path_dim = [x*_DYCORE_CRITICAL_PATH_LENGTH for x in _MEAN_CRITICAL_PATH_KERNEL]
    print("total critical path length (dimensionless) = _MEAN_CRITICAL_PATH_KERNEL * _DYCORE_CRITICAL_PATH_LENGTH = "
          "{} * {} = {}".format(_MEAN_CRITICAL_PATH_KERNEL, _DYCORE_CRITICAL_PATH_LENGTH, critical_path_dim))

    critical_path_cyc = helper.dim_to_abs_val(critical_path_dim, _DIMENSIONS)
    print("total critical path length (cycles) = {} cycles\n".format(critical_path_cyc))

    # compute maximum possible communication volume
    run_time_cyc = critical_path_cyc + reduce(operator.mul, _DIMENSIONS)
    print("total run time (cycles) = latency + dimX*dimY*dimZ = {}".format(run_time_cyc))

    run_time_sec = run_time_cyc / _FPGA_CLOCK_FREQUENCY
    print("total run time (seconds) = total run time (cycles) / _FPGA_CLOCK_FREQUENCY = {}".format(run_time_sec))

    comm_vol = _FPGA_BANDWIDTH * run_time_sec
    print("maximum available communication volume (to slow memory) = _FPGA_BANDWIDTH * total run time (seconds) = {}"
          .format(comm_vol))

    # TODO: add function/plot for the optimization of buffer space till reaching the maximum available
    #  communication volume


if __name__ == "__main__":
    do_estimate()
