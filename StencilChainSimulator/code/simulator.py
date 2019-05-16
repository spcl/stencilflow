
class Simulator:

    def __init__(self, input_nodes, input_config, kernel_nodes, output_nodes, dimensions) -> None:
        self.dimensions = dimensions
        self.input_nodes = input_nodes
        self.input_config = input_config
        self.kernel_nodes = kernel_nodes
        self.output_nodes = output_nodes

    def step_execution(self):

        # try to read all kernel inputs
        for kernel in self.kernel_nodes:
            try:
                self.kernel_nodes[kernel].reset_old_compute_state()
                self.kernel_nodes[kernel].try_read()
            except Exception as ex:
                self.diagnostics(ex)

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

    def initialize(self):
        # import data
        for input in self.input_nodes:
            self.input_nodes[input].init_input_data(self.input_config)


    def finalize(self):
        # save data
        for output in self.output_nodes:
            output.write_result_to_file()



    def simulate(self):

        self.initialize()

        # run simulation # TODO: how to detect that the simulation has been completed?
        # while(...):
        #   self.step_execution()
        self.step_execution()


        self.finalize()


    def diagnostics(self, exception):
        # gather info from all kernels
        # for kernel in self.kernel_nodes:
        #    self.kernel_nodes[kernel].diagnostics()
        raise exception

"""
    Procedure: 
    (1) INPUT nodes: feed all outgoing queues with data
    (2) run all kernels (read->execute->write)
"""