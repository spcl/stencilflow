from StencilChainSimulator.code.helper import Helper
from StencilChainSimulator.code.compute_graph import ComputeGraph
from StencilChainSimulator.code.calculator import Calculator
from StencilChainSimulator.code.bounded_queue import BoundedQueue


class Kernel:

    """
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

            - boundary condition:
                    - We know that the stencil boundary conditions in the COSMO model are functions of the local
                      neighbourhood (e.g. gradient, average, replicate value form n to n+1 (border replication),...)
                    - Idea: We implement the border replication strategy for all scenarios statically (this is enough
                      accuracy, since the latency would be most likely the same (the strategies mentioned above can be
                      implemented in parallel in hardware), and the amount of buffer space does not change as well.
                      Therefore this is a valid assumption and reduction of the problem complexity.

            - note:
                    - re-emptying of queue after reaching bound
                    - scenario:
                      suppose:  for i=1..N
                                    for j=1..M
                                        for k=1..P
                                            out[i,j,k] = in[i-1,j,k] + in[i,j,k] + in[i+1,j,k]
                                        end
                                    end
                                end
                      the internal buffer is of size: 2*P*N
                       j
                      /
                      --> i     in the case above we have to buffer 2 complete layers in i-j-direction till we can start
                      |         doing meaning full pipeline operations
                      k

                      if we reach the bound i == N, TODO: special handling?
    """

    def __init__(self, name, kernel_string, dimensions):

        # store arguments
        self.name = name  # kernel name
        self.kernel_string = kernel_string  # raw kernel string input
        self.dimensions = dimensions  # type: [int, int, int] # input array dimensions [dimX, dimY, dimZ]

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
        self.inputs = None  # type: [(str, BoundedQueue), ... ] # [(name, queue), ...] # 
        #  TODO: must be more flexible: e.g. multiple inputs access the same BoundedQueue: out = A[i,j,k] + A[i, j+1, k]
        self.outputs = None  # type: [(str, BoundedQueue), ... ] # [(name, queue), ...]

        # output delay queue: for simulation of calculation latency, fill it up with bubbles
        self.out_delay_queue = BoundedQueue("delay_output", self.graph.max_latency, [None]*(self.graph.max_latency-1))

    def reset_old_compute_state(self):
        self.var_map = dict()
        self.read_success = False
        self.exec_success = False
        self.result = None

    def convert_3d_to_1d(self, index):
        # convert [i, j, k] to flat 1D array index using the given dimensions [dimX, dimY, dimZ]
        # index = i*dimY*dimZ + j*dimZ + k = (i*dimY + j)*dimZ + k
        return (index[0]*self.dimensions[1] + index[1]*self.dimensions[2]) + index[2]

    def try_read(self):

        # reset old state
        self.reset_old_compute_state()

        # TODO: implement buffering for array access of the form: res = A[i,j,k] + A[i,j,k+1]




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
                # write result to latency-simulating buffer
                self.out_delay_queue.enqueue(self.result)
            except Exception as ex:
                self.diagnostics(ex)
        else:
            # write bubble to latency-simulating buffer
            self.out_delay_queue.enqueue(None)

        self.exec_success = True
        return self.exec_success

    def try_write(self):

        # check if data (not a bubble) is available
        data = self.out_delay_queue.dequeue()
        if data is not None:
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
    kernel = Kernel("ppgk", "res = wgtfac[i,j,k] * ppuv[i,j,k] + (1.0 - wgtfac[i,j,k]) * ppuv[i,j,k-1];", [10, 10, 10])
    print("Critical path latency: " + str(kernel.graph.max_latency))

