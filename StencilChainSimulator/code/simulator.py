
class Simulator:

    def __init__(self, kernels):
        self.kernels = kernels

    def stepexecution(self):
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
