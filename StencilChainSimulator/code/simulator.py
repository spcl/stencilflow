
class Simulator:

    def __init__(self, input_nodes, kernel_nodes, output_nodes, dimensions) -> None:
        self.dimensions = dimensions
        self.input_nodes = input_nodes
        self.kernel_nodes = kernel_nodes
        self.output_nodes = output_nodes

    def step_execution(self):

        """
        # try to read all kernel inputs
        for kernel in self.kernel_nodes:
            try:
                self.kernel_nodes[kernel].reset_old_compute_state()
                self.kernel_nodes[kernel].try_read()
            except Exception as ex:
                self.diagnostics(ex)
        """
        # try to execute all kernels
        for kernel in self.kernel_nodes:
            try:
                self.kernel_nodes[kernel].try_execute()
            except Exception as ex:
                self.diagnostics(ex)

        # try to write all kernel outputs
        for kernel in self.output_nodes:
            try:
                kernel.try_write()
            except Exception as ex:
                self.diagnostics(ex)

        for kernel in self.kernel_nodes:
            try:
                kernel.try_write()
            except Exception as ex:
                self.diagnostics(ex)


    def diagnostics(self, exception):
        # gather info from all kernels
        for kernel in self.kernel_nodes:
            self.kernel_nodes[kernel].diagnostics()
        raise exception

"""
    Procedure: 
    (1) INPUT nodes: feed all outgoing queues with data
    (2) run all kernels (read->execute->write)
"""