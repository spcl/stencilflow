import operator
import functools
import helper
from bounded_queue import BoundedQueue
from base_node_class import BaseKernelNodeClass
from typing import List
from dace.types import typeclass

class Output(BaseKernelNodeClass):

    def __init__(self, name: str, data_type: typeclass, dimensions: List[int], data_queue=None):
        super().__init__(name=name, data_type=data_type, data_queue=BoundedQueue(name="output",
                                                                                 maxsize=functools.reduce(operator.mul, dimensions),
                                                                                 collection=[]))

    def reset_old_compute_state(self):
        pass  # nothing to do

    def try_read(self):
        assert len(self.inputs) == 1  # there should be only a single one
        for inp in self.inputs:
            if self.inputs[inp]["delay_buffer"].try_peek_last() is not False and self.inputs[inp]["delay_buffer"].try_peek_last() is not None:
                self.data_queue.enqueue(self.inputs[inp]["delay_buffer"].dequeue())
                self.program_counter += 1
            elif self.inputs[inp]["delay_buffer"].try_peek_last() is not False:
                self.inputs[inp]["delay_buffer"].dequeue()  # remove bubble

    def try_write(self):
        pass  # nothing to do

    def write_result_to_file(self):
        print("final output: {}".format(self.data_queue.export_data()))
        helper.save_array(self.data_queue.export_data(), "{}_{}.dat".format(self.name, "out"))
