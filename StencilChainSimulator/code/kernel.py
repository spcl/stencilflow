import os
import ast
import json
import re
import networkx as nx
from StencilChainSimulator.code.helper import Helper
from StencilChainSimulator.code.compute_graph import ComputeGraph
from StencilChainSimulator.code.calculator import Calculator
from StencilChainSimulator.code.bounded_queue import BoundedQueue


class Kernel:

    def __init__(self, name, kernel_string):

        # store arguments
        self.name = name  # kernel name
        self.kernel_string = kernel_string  # raw kernel string input

        # read static parameters from config
        self.config = Helper.parse_config("kernel.config")
        self.calculator = Calculator()

        # analyse input
        self.graph = ComputeGraph()
        self.graph.generate_graph(kernel_string)
        self.graph.calculate_latency()
        self.graph.plot_graph()

        # init sim specific params
        self.var_map = None  # var_map[var_name] = var_value
        self.read_success = False
        self.exec_success = False
        self.result = None  # type: float
        self.inputs = None  # type: [(str, BoundedQueue), ... ] # [(name, queue), ...]
        self.outputs = None  # type: [(str, BoundedQueue), ... ] # [(name, queue), ...]

    '''  
        interface for FPGA-like execution (get called from the scheduler)
    
            - read: 
                    - saturation phase: read unconditionally
                    - execution phase: read all inputs iff they are available
            - execute:
                    - saturation phase: do nothing
                    - execution phase: if input read, execute stencil using the input
            - write:
                    - saturation phase: do nothing
                    - execution phase: write result from execution to output buffers 
                        --> if output buffer overflows: assumptions about size were wrong!
                        
            - return:
                    - True  iff successful
                    - False otherwise 
    '''

    def reset_old_state(self):
        self.var_map = dict()
        self.read_success = False
        self.exec_success = False
        self.result = None

    def try_read(self):

        # reset old state
        self.reset_old_state()

        # check if all inputs are available
        all_available = True
        for inp in self.inputs:
            if inp.isEmpty():
                all_available = False
                break

        # dequeue all of them into the variable map
        if all_available:
            for inp in self.inputs:
                # read inputs into var_map
                try:
                    self.var_map[inp[0]] = inp[2].dequeue()
                except Exception as ex:
                    self.diagnostics(ex)

        self.read_success = all_available
        return all_available

    def try_execute(self):

        # check if read has been successful
        if self.read_success:
            # execute calculation
            try:
                self.result = self.calculator.eval_expr(self.var_map, self.kernel_string)
            except Exception as ex:
                self.diagnostics(ex)

        self.exec_success = True
        return self.exec_success

    def try_write(self):

        # check if execution has been successful
        if self.exec_success:
            # write result to all output queues
            for outp in self.outputs:
                try:
                    outp.enqueue(self.result)
                except Exception as ex:
                    self.diagnostics(ex)

        return self.available

    '''
        interface for error overview reporting (gets called in case of an exception)

        - goal:
                - get an overview over the whole stencil chain state in case of an error
                    - maximal and current size of all buffers
                    - type of phase (saturation/execution)
                    - efficiency (#execution cycles / #total cycles)
    '''
    def diagnostics(self, ex):
        raise NotImplementedError()

    '''
        simple test kernel for debugging
    '''


if __name__ == "__main__":
    kernel = Kernel("ppgk", "res = wgtfac[i,j,k] * ppuv[i,j,k] + (1.0 - wgtfac[i,j,k]) * ppuv[i,j,k-1];")

