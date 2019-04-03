import argparse
import operator
import functools
import networkx as nx
import helper
from kernel import Kernel
from bounded_queue import BoundedQueue
from base_node_class import BaseKernelNodeClass
from typing import List, Dict

class Optimizer:

    def __init__(self, kernels):
        self.kernels = kernels
        self.metric_data: List[(BoundedQueue, int)] = list() # [Queue, communication volume]


    def do_optimization(self, available_communication_volume):

        fast_memory_use = 0
        for item in self.metric_data:
            if not item[0].swap_out:
                fast_memory_use += item[0].max_size
        slow_memory_use = 0

        # optimize for maximum communication volume use (and as
        for item in self.metric_data:
            if available_communication_volume > item[1]:
               available_communication_volume -= item[1]
               fast_memory_use -= item[0].max_size
               slow_memory_use += item[0].max_size
               item[0].swap_out = True
            else:
                break


    def add_delay_buffers_to_metric(self):
        pass

    def add_internal_buffers_to_metric(self):
        pass


if __name__ == "__main__":
    opt = Optimizer()

    print()
