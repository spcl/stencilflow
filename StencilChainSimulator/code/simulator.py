
class Simulator:

    def __init__(self, kernels):
        self.kernels = kernels

    def stepexecution(self):
        # read all kernel inputs
        for kernel in self.kernels:
            try:
                kernel.read()
            except Exception as ex:
                self.diagnostics(ex)
        # execute all executable kernels
        for kernel in self.kernels:
            if kernel.canexecute():
                try:
                    kernel.execute()
                except Exception as ex:
                    self.diagnostics(ex)
        # write all writable kernels
        for kernel in self.kernels:
            if kernel.canwrite():
                try:
                    kernel.write()
                except Exception as ex:
                    self.diagnostics(ex)

    def diagnostics(self, exception):
        # get info from all kernels
        for kernel in self.kernels:
            kernel.diagnostics()
