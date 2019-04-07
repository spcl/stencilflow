import operator
import functools
from enum import Enum
import helper
from kernel import Kernel
from bounded_queue import BoundedQueue
from functools import reduce
from typing import List, Dict

# assumption: float32
_SIZEOF_DATATYPE = 4

class Optimizer:

    '''
        optimization strategy:
            - initial state: all buffers are in fast memory, there is no communcation volume used for transfering data
            between slow and fast memory
            -

    '''

    def __init__(self, kernels: Dict[str, Kernel], dimensions: List[int]):
        self.dimensions: List[int] = dimensions
        self.kernels = kernels
        self.fast_memory_use: int = 0
        self.slow_memory_use: int = 0
        self.metric_data: List[(BoundedQueue, int)] = list()  # [Queue, communication volume]
        self.add_buffers_to_metric()

    def minimize_comm_vol(self, fast_memory_bound: int, slow_memory_bound: int):
        # optimize for minimal communication volume use / maximal fast memory use
        opt = self.max_metric()
        while self.fast_memory_use > fast_memory_bound:
            self.fast_memory_use -= opt[0].max_size * _SIZEOF_DATATYPE
            self.slow_memory_use += opt[0].max_size * _SIZEOF_DATATYPE
            opt[0].swap_out = True
            self.update_neighbours(opt)
            opt = self.max_metric()


    def minimize_fast_mem(self, communication_volume_bound: int):
        # optimize for minimal fast memory use / maximum communication volume use
        opt = self.max_metric()
        while opt[1] < communication_volume_bound:
            communication_volume_bound -= opt[1]
            self.fast_memory_use -= opt[0].max_size * _SIZEOF_DATATYPE
            self.slow_memory_use += opt[0].max_size * _SIZEOF_DATATYPE
            opt[0].swap_out = True
            self.update_neighbours(opt)
            opt = self.max_metric()


    def optimize_to_ratio(self, ratio: float): # ratio = #fast_mem / #comm_vol
        # optimize for the ratio of #fast_memory/communication_volume
        opt = self.max_metric()
        while self.ratio() > ratio:
            self.fast_memory_use -= opt[0].max_size * _SIZEOF_DATATYPE
            self.slow_memory_use += opt[0].max_size * _SIZEOF_DATATYPE
            opt[0].swap_out = True
            self.update_neighbours(opt)
            opt = self.max_metric()


    def ratio(self):
        return self.fast_memory_use / self.slow_memory_use


    def reset(self):
        # initially put all buffers into fast memory
        for item in self.metric_data:
            item[0].swap_out = False

        # count fast memory usage
        self.fast_memory_use = 0
        for item in self.metric_data:
            if not item[0].swap_out:
                self.fast_memory_use += item[0].max_size

        # set slow memory usage
        self.slow_memory_use = 0


    def max_metric(self):
        return max(self.metric_data, lambda x: x[0].max_size / x[1])


    def update_neighbours(self, buffers):
        """
            How to determine the necessary communication volume?
            (predecessor, successor):
            case (fast, fast): 2C
            case (fast, slow): C
            case (slow, fast): C
            case (slow, slow): 0

            where C:= communication volume to stream single data array in or out of fast memory (=SIZE(data array))

            Note;
            pred of delay buffer is always fast memory
        """

    def add_buffers_to_metric(self):
        for kernel in self.kernels:
            for buf in self.kernels[kernel].delay_buffer:

                # get delay buffer first
                del_buf = {
                    "queue": self.kernels[kernel].delay_buffer[buf],
                    "comm_vol": 2*self.single_comm_volume(),
                    "type": "delay",
                    "prev": None,
                    "next": None}
                self.fast_memory_use += del_buf["queue"].maxsize * _SIZEOF_DATATYPE
                self.metric_data.append(del_buf)

                # get internal buffers next
                prev = del_buf
                for entry in self.kernels[kernel].internal_buffer[buf]:
                    curr = {
                    "queue": entry,
                    "comm_vol": 2*self.single_comm_volume(),
                    "type": "internal",
                    "prev": prev,
                    "next": None}
                    prev["next"] = curr
                    self.fast_memory_use += curr["queue"].maxsize * _SIZEOF_DATATYPE
                    self.metric_data.append(curr)
                    prev = curr

    def single_comm_volume(self):
        return reduce(operator.mul, self.dimensions) * _SIZEOF_DATATYPE

if __name__ == "__main__":
    opt = Optimizer()
    print()
