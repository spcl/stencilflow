from kernel_chain_graph import KernelChainGraph


class Simulator:

    def __init__(self, path: str) -> None:
        self.chain_graph = KernelChainGraph(path)


    '''
    def step_execution(self):
        # try to read all kernel inputs
        for kernel in self.kernels:
            try:
                kernel.try_read()
            except Exception as ex:
                self.diagnostics(ex)
        # try to execute all kernels
        for kernel in self.kernels:
            try:
                kernel.try_execute()
            except Exception as ex:
                self.diagnostics(ex)
        # try to write all kernel outputs
        for kernel in self.kernels:
            try:
                kernel.try_write()
            except Exception as ex:
                self.diagnostics(ex)

    def diagnostics(self, exception):
        # gather info from all kernels
        for kernel in self.kernels:
            kernel.diagnostics()
        raise exception
    '''


"""
    Procedure: 
    (1) INPUT nodes: feed all outgoing queues with data
    (2) run all kernels (read->execute->write)
"""