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

import numpy as np
from dace.dtypes import typeclass

from base_node_class import BaseKernelNodeClass
from bounded_queue import BoundedQueue


class Input(BaseKernelNodeClass):
    """
        The Input class is a subclass of the BaseKernelNodeClass and represents an Input node in the KernelChainGraph.
        Its purpose is to feed input array data into the pipeline/dataflow design.
    """

    def __init__(self,
                 name: str,
                 data_type: typeclass,
                 data_queue: BoundedQueue = None) -> None:
        """
        Initialize the Input node.
        :param name: node name
        :param data_type: data type of the data
        :param data_queue: BoundedQueue containing the input data
        """
        # initialize superclass
        super().__init__(name=name, data_queue=data_queue, data_type=data_type)
        # set internal fields
        self.queues = dict()
        self.dimension_size = data_queue.maxsize
        self.init = False  # flag for internal initialization (must be done later when all successors are added to the
        # graph)

    def init_queues(self):
        """
        Create individual queues for all successors in order to feed data individually into the channels.
        """
        # add a queue for each successor
        self.queues = dict()
        for successor in self.outputs:
            self.queues[successor] = BoundedQueue(name=successor,
                                                  maxsize=self.data_queue.maxsize,
                                                  collection=self.data_queue.export_data())
        self.init = True  # set init flag

    def reset_old_compute_state(self):
        """
        Reset compute-specific internal state (only for Kernel node).
        """
        pass  # nothing to do

    def try_read(self):
        """
        Read data from predecessor (only for Kernel and Output node).
        """
        pass  # nothing to do

    def try_write(self):
        """
        Feed data to all successor channels.
        :return:
        """
        # set up all individual data queues
        if not self.init:
            self.init_queues()
        # feed data into pipeline inputs (all kernels that feed from this input data array)
        for successor in self.outputs:
            if self.queues[successor].is_empty() and not self.outputs[successor]["delay_buffer"].is_full():  # no more
                # data to feed, add bubble
                self.outputs[successor]["delay_buffer"].enqueue(None)  # insert bubble
            elif self.outputs[successor]["delay_buffer"].is_full():  # channel full, skip
                pass
            else:  # feed data into channel
                data = self.queues[successor].dequeue()
                self.outputs[successor]["delay_buffer"].enqueue(data)
                self.program_counter = self.dimension_size - max([self.queues[x].size() for x in self.queues])

    def init_input_data(self, inputs):
        """
        Initialize internal queue i.e. read data from config or external file.
        :param inputs:
        :return:
        """
        # check if data is in the config or in a separate file
        if isinstance(inputs[self.name]["data"], list):  # inline
            self.data_queue.import_data(inputs[self.name]["data"])
        elif isinstance(inputs[self.name]["data"], str):  # external file
            coll = None
            if inputs[self.name]["data"].lower().endswith(('.dat', '.bin', '.data')):  # general binary data file
                coll = np.fromfile(inputs[self.name]["data"], inputs[self.name]["data_type"].type)
            if inputs[self.name]["data"].lower().endswith('.h5'):  # h5 file
                from h5py import File
                f = File(inputs[self.name]["data"], 'r')
                coll = np.array(list(f[list(f.keys())[0]]), dtype=inputs[self.name]["data_type"].type)  # read data
                # from first key
            elif inputs[self.name]["data"].lower().endswith('.csv'):  # csv file
                coll = list(np.genfromtxt(inputs[self.name]["data"], delimiter=',',
                                          dtype=inputs[self.name]["data_type"].type))
            # add data to queue
            self.data_queue.import_data(coll)
        else:
            raise Exception("Input data representation should either be implicit (list) or a path to a csv file.")
