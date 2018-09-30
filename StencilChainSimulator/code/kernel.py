
class Kernel:

    def __init__(self, name):
        self.name = name


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
    '''
    def read(self):
        raise NotImplementedError()

    def execute(self):
        raise NotImplementedError()

    def write(self):
        raise NotImplementedError()

    '''
        interface for error overview reporting (gets called in case of an exception)

        - goal:
                - get an overview over the whole stencil chain state in case of an error
                    - current size of all buffers
                    - type of phase (saturation/execution)
                    - efficiency (#execution cycles / # total cycles)
    '''
    def diagnostics(self):
        raise NotImplementedError()