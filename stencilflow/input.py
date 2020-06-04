from dace.dtypes import typeclass

from stencilflow.base_node_class import BaseKernelNodeClass
from stencilflow.bounded_queue import BoundedQueue


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
            self.queues[successor] = BoundedQueue(
                name=successor,
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
            if self.queues[successor].is_empty() and not self.outputs[
                    successor]["delay_buffer"].is_full():  # no more
                # data to feed, add bubble
                self.outputs[successor]["delay_buffer"].enqueue(
                    None)  # insert bubble
            elif self.outputs[successor]["delay_buffer"].is_full(
            ):  # channel full, skip
                pass
            else:  # feed data into channel
                data = self.queues[successor].dequeue()
                self.outputs[successor]["delay_buffer"].enqueue(data)
                self.program_counter = self.dimension_size - max(
                    [self.queues[x].size() for x in self.queues])
