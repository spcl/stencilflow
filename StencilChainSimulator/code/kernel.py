import os
import ast
import json
import re
import networkx as nx
from StencilChainSimulator.code.helper import Helper


class Kernel:

    def __init__(self, name, kernel_string):

        # store arguments
        self.name = name  # kernel name
        self.kernel_string = kernel_string  # raw kernel string input

        # read static parameters from config
        self.config = Helper.parse_config("kernel.config")

        # analyse input
        self.parse_kernel_string()
        self.graph = nx.DiGraph()  # create empty directed graph

    ''' 
        analyse input kernel string 
    '''

    def parse_kernel_string(self):

        # remove "auto res =" from input string
        equation = re.sub("auto res = ", "", self.kernel_string)

        # TODO: kernel_string can be of format: "auto res = c + d(i,j,k); c = a(i,j,k) + b(i,j,k);"

        node = ast.parse(equation)
        
        print("dummy")

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

    def try_read(self):
        raise NotImplementedError()

    def try_execute(self):
        raise NotImplementedError()

    def try_write(self):
        raise NotImplementedError()

    '''
        interface for error overview reporting (gets called in case of an exception)

        - goal:
                - get an overview over the whole stencil chain state in case of an error
                    - maximal and current size of all buffers
                    - type of phase (saturation/execution)
                    - efficiency (#execution cycles / #total cycles)
    '''
    def diagnostics(self):
        raise NotImplementedError()

    '''
        simple test kernel for debugging
    '''


if __name__ == "__main__":
    kernel = Kernel("res1", "auto res = res1(i,j,k) + inC(i,j,k);")
