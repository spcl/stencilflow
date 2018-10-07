import os
import ast
import json

class Kernel:

    def __init__(self, name, kernel_string):

        # store arguments
        self.name = name  # kernel name
        self.kernel_string = kernel_string  # raw kernel string input

        # read static parameters from config
        self.config_path = "kernel.config"
        self.op_latency = None
        self.parse_config()

    ''' 
        import config 
    '''
    def parse_config(self):

        # check file exist and readable
        if not os.access(self.config_path, os.R_OK):
            raise Exception("Config does not exist or is not readable.")

        # open the file read-only
        file_handle = open(self.config_path, "r")

        # try to parse it
        config = json.loads(file_handle.read())  # type: dict

        # close the file handle
        file_handle.close()

        # fill in information
        self.op_latency = config["op_latency"]

    ''' 
        analyse input kernel string 
    '''

    def parse_inputs(self):
        return

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
