from bounded_queue import BoundedQueue
from base_node_class import BaseKernelNodeClass
from dace.types import typeclass

class Input(BaseKernelNodeClass):

    def __init__(self, name: str, data_type: typeclass, data_queue: BoundedQueue = None) -> None:
        super().__init__(name=name, data_queue=data_queue, data_type=data_type)

    def reset_old_compute_state(self):
        pass  # nothing to do

    def try_read(self):
        pass  # nothing to do

    def try_write(self):
        # feed data into pipeline inputs (all kernels that feed from this input data array)
        if self.data_queue.is_empty():
            for successor in self.outputs:
                self.outputs[successor]["delay_buffer"].enqueue(None)  # insert bubble
        else:
            data = self.data_queue.dequeue()
            for successor in self.outputs:
                self.outputs[successor]["delay_buffer"].enqueue(data)
                self.program_counter += 1

    def init_input_data(self, inputs):
        # TODO: make use of passed data_type = inputs[self.name]["data_type"]
        # check if data is in the file or in a separate file
        if isinstance(inputs[self.name]["data"], list):
            self.data_queue.import_data(inputs[self.name]["data"])

        elif isinstance(inputs[self.name]["data"], str):  # external file
            coll = None

            if inputs[self.name]["data"].lower().endswith(('.dat', '.bin', '.data')):  # general binary data
                from numpy import fromfile
                coll = fromfile(inputs[self.name]["data"], float)
            if inputs[self.name]["data"].lower().endswith('.h5'):
                from h5py import File
                f = File(inputs[self.name]["data"], 'r')
                coll = list(f[list(f.keys())[0]])  # read data from first key
            elif inputs[self.name]["data"].lower().endswith('.csv'):
                from numpy import genfromtxt
                coll = list(genfromtxt(inputs[self.name]["data"], delimiter=','))

            self.data_queue.import_data(coll)
        else:
            raise Exception("Input data representation should either be implicit (list) or a path to a csv file.")
