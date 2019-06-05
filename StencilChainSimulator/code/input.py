from bounded_queue import BoundedQueue
from base_node_class import BaseKernelNodeClass
from dace.types import typeclass

class Input(BaseKernelNodeClass):

    def __init__(self, name: str, data_type: typeclass, data_queue: BoundedQueue = None) -> None:
        super().__init__(name=name, data_queue=data_queue, data_type=data_type)
        self.init = False
        self.queues = dict()

    def init_queues(self):
        self.queues = dict()
        for successor in self.outputs:
            self.queues[successor] = BoundedQueue(name=successor,
                                                  maxsize=self.data_queue.maxsize,
                                                  collection=self.data_queue.export_data())
        self.init = True

    def reset_old_compute_state(self):
        pass  # nothing to do

    def try_read(self):
        pass  # nothing to do

    def try_write(self):
        # set up all individual data queues
        if not self.init:
            self.init_queues()

        # feed data into pipeline inputs (all kernels that feed from this input data array)
        for successor in self.outputs:
            if self.queues[successor].is_empty() and not self.outputs[successor]["delay_buffer"].is_full():
                self.outputs[successor]["delay_buffer"].enqueue(None)  # insert bubble
            elif self.outputs[successor]["delay_buffer"].is_full():
                pass
            else:
                data = self.queues[successor].dequeue()
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
