import functools
import operator

class Simulator:

    def __init__(self, input_nodes, input_config, kernel_nodes, output_nodes, dimensions) -> None:
        self.dimensions = dimensions
        self.input_nodes = input_nodes
        self.input_config = input_config
        self.kernel_nodes = kernel_nodes
        self.output_nodes = output_nodes

    def step_execution(self):
        """
            try to read all kernel inputs
        """
        for output in self.output_nodes:
            try:
                self.output_nodes[output].try_read()
            except Exception as ex:
                self.diagnostics(ex)
        for kernel in self.kernel_nodes:
            try:
                self.kernel_nodes[kernel].reset_old_compute_state()
                self.kernel_nodes[kernel].try_read()
            except Exception as ex:
                self.diagnostics(ex)

        """
            try to execute all kernels
        """
        for kernel in self.kernel_nodes:
            try:
                self.kernel_nodes[kernel].try_execute()
            except Exception as ex:
                self.diagnostics(ex)

        """
            try to write all kernel outputs
        """
        for input in self.input_nodes:
            try:
                self.input_nodes[input].try_write()
            except Exception as ex:
                self.diagnostics(ex)
        for kernel in self.kernel_nodes:
            try:
                self.kernel_nodes[kernel].try_write()
            except Exception as ex:
                self.diagnostics(ex)

    def initialize(self):
        # import data
        for input in self.input_nodes:
            self.input_nodes[input].init_input_data(self.input_config)

        #for input in self.input_nodes:
        #   import helper
        #    for i in range(functools.reduce(operator.mul, self.dimensions, 1)):
        #        self.input_nodes[input].try_write()

    def finalize(self):
        # save data
        for output in self.output_nodes:
            self.output_nodes[output].write_result_to_file()

    def get_result(self):
        # return all output data
        result_dict = dict()
        for output in self.output_nodes:
            result_dict[output] = self.output_nodes[output].data_queue.export_data()
        return result_dict

    def all_done(self) -> bool:
        total_elements = functools.reduce(operator.mul, self.dimensions)

        for input in self.input_nodes:
            if self.input_nodes[input].program_counter < total_elements:
                return False

        for kernel in self.kernel_nodes:
            if self.kernel_nodes[kernel].program_counter < total_elements:
                return False

        for output in self.output_nodes:
            if self.output_nodes[output].program_counter < total_elements:
                return False
        return True

    def simulate(self):

        self.initialize()

        # run simulation
        PC = 0
        while not self.all_done():
            self.step_execution()
            PC += 1
            '''
            for input in self.input_nodes:
                print("input:{}, PC: {}".format(input, self.input_nodes[input].program_counter))
            for kernel in self.kernel_nodes:
                print("kernel:{}, PC: {}".format(kernel, self.kernel_nodes[kernel].program_counter))
            for output in self.output_nodes:
                print("output:{}, PC: {}".format(output, self.output_nodes[output].program_counter))
            '''

        print("Simulation completed after {} cycles.".format(PC))

        self.finalize()

    def diagnostics(self, exception):
        # TODO: gather info from all kernels
        # for kernel in self.kernel_nodes:
        #    self.kernel_nodes[kernel].diagnostics()
        raise exception
