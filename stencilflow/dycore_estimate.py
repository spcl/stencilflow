#!/usr/bin/env python3
# encoding: utf-8

"""
BSD 3-Clause License

Copyright (c) 2018-2020, Andreas Kuster
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__author__ = "Andreas Kuster"
__copyright__ = "Copyright 2018-2020, StencilFlow"
__license__ = "BSD-3-Clause"

import functools
import operator
from functools import reduce
from typing import List

from stencilflow.log_level import LogLevel
from stencilflow.helper import helper
from stencilflow.kernel_chain_graph import KernelChainGraph

"""
    Intro:
        This is a buffer size and bandwidth estimate for the whole dynamical core of the COSMO weather model. With only
        a few actual kernel chains ported and having the shape of the whole (production) dynamical core, we try to
        estimate if it the buffer space and bandwidth requirements are in the order of resources we have available on a
        single or an cluster of at most 32 Intel Stratix 10 FPGAs.

    Assumptions:
        - data type: float (32bit IEEE 754)
        - iteration over the smaller two dimensions possible (e.g. 1024x1024x64 -> iterate over 64x1024)
        - estimate for critical path length (#cycles):
            Stencil Chains (automatic output):
                fastwaves:     [3, 1, 54]
                diffusion_min: [3, 0, 24]
                advection_min: [7, 4, 26]
                -> use the mean of the three: [4, 2, 35]
        - clock frequency: 200Mhz
        - fast memory (Stratix 10): 25MB
        - bandwidth to slow memory (Stratix 10): 86.4GB/s
        - we assume that as soon as the pipeline is saturated, we can produce one result per cycle

"""
# datatype: for the moment, we assume float (32bit) are precise enough
_SIZEOF_DATATYPE: int = 4  # in bytes

# we assume that we can iterate over the smaller dimension by transposition of all data arrays, if not, change this to
# [64, 1024, 1024] (would lead to an additional factor 16 of buffer space growth)
_DIMENSIONS: List[int] = [1024, 1024, 64]  # longitude x latitude x altitude

#  FPGA clock frequency: 200Mhz
_FPGA_CLOCK_FREQUENCY: int = int(2e8)

# available FGPA fast memory (Stratix 10): 25MB
_FPGA_FAST_MEMORY_SIZE: int = int(25e6)

# available bandwidth to slow memory (Stratix 10): 86.4GB/s
_FPGA_BANDWIDTH: int = int(86.4e9)

# we assume that as soon as the pipeline is saturated, we can produce one result per cycle
_CYCLES_PER_OUTPUT: int = 1


def do_estimate():
    """
    This function is meant to programmatically go through the current 'best-estimate' calculation of our model.
    :return: None
    """

    # estimate for critical path length (#cycles): mean of the tree stencils: fastwaves, diffusion, advection
    critical_paths: List[List[int]] = list(list())
    # instantiate fastwaves and add critical path
    critical_paths.append(KernelChainGraph(path="input/fastwaves.json",
                                           plot_graph=False,
                                           log_level=LogLevel.FULL.value).compute_critical_path_dim())
    # instantiate diffusion and add critical path
    critical_paths.append(KernelChainGraph(path="input/diffusion.json",
                                           plot_graph=False,
                                           log_level=LogLevel.FULL.value).compute_critical_path_dim())
    # instantiate advection and add critical path
    critical_paths.append(KernelChainGraph(path="input/advection.json",
                                           plot_graph=False,
                                           log_level=LogLevel.FULL.value).compute_critical_path_dim())
    # calculate mean of the three
    critical_path_sum = functools.reduce(lambda x, y: helper.list_add_cwise(x, y), critical_paths, [0] * 3)
    mean: List[int] = list(map(lambda x: x / len(critical_paths), critical_path_sum))
    _MEAN_CRITICAL_PATH_KERNEL: List[int] = mean
    print("Mean critical path length of the three stencils is: {}".format(mean))
    # print header
    print("###########################################################")
    print("COSMO dynamical core buffer size estimate report:\n")
    print("###########################################################")
    # instantiate the dummy-dycore (modified to fixed latency per kernel of 4*latency(addition) to get full analysis
    chain = KernelChainGraph("input/dycore_upper_half.json")
    # assumption: since we implemented ~1/2 of the dycore in the dummy input file, we assume the critical path is 2x
    # longer
    _DYCORE_CRITICAL_PATH_LENGTH = 2 * chain.compute_critical_path()
    # compute total critical path
    critical_path_dim = [x * _DYCORE_CRITICAL_PATH_LENGTH for x in _MEAN_CRITICAL_PATH_KERNEL]
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


if __name__ == "__main__":
    do_estimate()
