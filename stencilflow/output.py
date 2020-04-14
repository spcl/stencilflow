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

import os
import functools
import operator

from typing import List

from dace.dtypes import typeclass

import stencilflow.helper as helper
from stencilflow.base_node_class import BaseKernelNodeClass
from stencilflow.bounded_queue import BoundedQueue


class Output(BaseKernelNodeClass):
    """
        The Output class is a subclass of the BaseKernelNodeClass and represents an Ouput node in the KernelChainGraph.
        Its purpose is to store data coming from the pipeline/dataflow design.
    """
    def __init__(self,
                 name: str,
                 data_type: typeclass,
                 dimensions: List[int],
                 data_queue=None) -> None:
        """
        Initializes the Output class with given initialization parameters.
        :param name: name of the output node
        :param data_type: data type of the data feed into output
        :param dimensions: global problem dimensions
        :param data_queue: dummy
        """
        # init superclass with queue of size: global problem size
        super().__init__(name=name,
                         data_type=data_type,
                         data_queue=BoundedQueue(name="output",
                                                 maxsize=functools.reduce(
                                                     operator.mul, dimensions),
                                                 collection=[]))

    def reset_old_compute_state(self) -> None:
        """
        Reset compute-specific internal state (only for Kernel node).
        """
        pass  # nothing to do

    def try_read(self) -> None:
        """
        Read data from predecessor.
        """
        # check for single input
        assert len(self.inputs) == 1  # there should be only a single one
        for inp in self.inputs:
            # read data
            if self.inputs[inp]["delay_buffer"].try_peek_last() is not False and self.inputs[inp]["delay_buffer"] \
                    .try_peek_last() is not None:
                self.data_queue.enqueue(
                    self.inputs[inp]["delay_buffer"].dequeue())
                self.program_counter += 1
            elif self.inputs[inp]["delay_buffer"].try_peek_last() is not False:
                self.inputs[inp]["delay_buffer"].dequeue()  # remove bubble

    def try_write(self) -> None:
        """
        Feed data to all successor channels (for Input and Kernel nodes)
        """
        pass  # nothing to do

    def write_result_to_file(self, input_config_name: str) -> None:
        """
        Write internal queue with computation result to the file results/INPUT_CONFIG_NAME/SELF.NAME_simulation.dat
        :param input_config_name: the config name, used to determine the save path
        """
        # join the paths
        output_folder = os.path.join("results", input_config_name)
        # create (recursively) directories
        os.makedirs(output_folder, exist_ok=True)
        # store the data
        helper.save_array(
            self.data_queue.export_data(),
            "{}/{}_{}.dat".format(output_folder, self.name, 'simulation'))
