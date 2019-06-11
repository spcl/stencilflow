import operator
from kernel import Kernel
from functools import reduce
from typing import List, Dict

# assumption: float32
_SIZEOF_DATATYPE = 4
_EPS = 1e-10

class Optimizer:

    """
        optimization strategy:
            - initial state: all buffers are in fast memory, there is no communication volume used for transferring data
            between slow and fast memory
            -
    """

    def __init__(self, kernels: Dict[str, Kernel], dimensions: List[int], verbose: bool = False):
        self.dimensions: List[int] = dimensions
        self.kernels = kernels
        self.fast_memory_use: int = 0
        self.slow_memory_use: int = 0
        self.metric_data: List[Dict] = list()
        self.add_buffers_to_metric()
        self.reset()
        self.verbose = verbose

    def reinit(self):
        self.fast_memory_use: int = 0
        self.slow_memory_use: int = 0
        self.metric_data: List[Dict] = list()
        self.add_buffers_to_metric()
        self.reset()

    def minimize_comm_vol(self, fast_memory_bound: int, slow_memory_bound: int):
        self.reinit()
        # optimize for minimal communication volume use / maximal fast memory use
        opt = self.max_metric()
        while not self.empty_list(self.metric_data) and self.fast_memory_use > fast_memory_bound:
            self.fast_memory_use -= opt["queue"].maxsize * _SIZEOF_DATATYPE
            self.slow_memory_use += opt["queue"].maxsize * _SIZEOF_DATATYPE
            opt["queue"].swap_out = True
            self.update_neighbours(opt)
            self.metric_data.remove(opt)
            opt = self.max_metric()


    def minimize_fast_mem(self, communication_volume_bound: int):
        self.reinit()
        # optimize for minimal fast memory use / maximum communication volume use
        opt = self.max_metric()
        while not self.empty_list(self.metric_data) and opt["comm_vol"] < communication_volume_bound:
            communication_volume_bound -= opt["comm_vol"]
            self.fast_memory_use -= opt["queue"].maxsize * _SIZEOF_DATATYPE
            self.slow_memory_use += opt["queue"].maxsize * _SIZEOF_DATATYPE
            opt["queue"].swap_out = True
            self.update_neighbours(opt)
            self.metric_data.remove(opt)
            opt = self.max_metric()


    def optimize_to_ratio(self, ratio: float): # ratio = #fast_mem / #comm_vol
        self.reinit()
        # optimize for the ratio of #fast_memory/communication_volume
        opt = self.max_metric()
        while not self.empty_list(self.metric_data) and self.ratio() > ratio:
            self.fast_memory_use -= opt["queue"].maxsize * _SIZEOF_DATATYPE
            self.slow_memory_use += opt["queue"].maxsize * _SIZEOF_DATATYPE
            opt["queue"].swap_out = True
            self.update_neighbours(opt)
            self.metric_data.remove(opt)
            opt = self.max_metric()

    @staticmethod
    def empty_list(lst):
        return len(lst) == 0

    def ratio(self):
        return self.fast_memory_use / (self.slow_memory_use + 1e-5)


    def reset(self):
        # initially put all buffers into fast memory
        for item in self.metric_data:
            item["queue"].swap_out = False

        # count fast memory usage
        self.fast_memory_use = 0
        for item in self.metric_data:
            if not item["queue"].swap_out:
                self.fast_memory_use += item["queue"].maxsize

        # set slow memory usage
        self.slow_memory_use = 0


    def max_metric(self):
        if self.empty_list(self.metric_data):
            return None
        else:
            return max(self.metric_data, key=lambda x: x["queue"].maxsize / x["comm_vol"])

    def update_neighbours(self, buffer):
        if buffer["prev"] is not None:
            self.update_comm_vol(buffer["prev"])
        if buffer["next"] is not None:
            self.update_comm_vol(buffer["next"])

    def update_comm_vol(self, buffer):
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
        if buffer["type"] == "delay":
            pre_fast = True
        elif buffer["prev"]["queue"].swap_out:
            pre_fast = False
        else:
            pre_fast = True

        if buffer["next"] is None:
            succ_fast = True
        elif buffer["next"]["queue"].swap_out:
            succ_fast = False
        else:
            succ_fast = True

        if pre_fast and succ_fast:
            buffer["comm_vol"] = 2*self.single_comm_volume()
        elif (pre_fast and not succ_fast) or (not pre_fast and succ_fast):
            buffer["comm_vol"] = 1*self.single_comm_volume()
        else:
            buffer["comm_vol"] = _EPS


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
