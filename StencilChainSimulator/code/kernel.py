import helper
from compute_graph import ComputeGraph
from calculator import Calculator
from bounded_queue import BoundedQueue
from base_node_class import BaseKernelNodeClass
from compute_graph import Name, Num, Binop, Call, Output, Subscript


class Kernel(BaseKernelNodeClass):

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

    def __init__(self, name, kernel_string, dimensions, plot_graph=False):

        super().__init__(name)

        # store arguments
        self.kernel_string = kernel_string  # raw kernel string input
        self.dimensions = dimensions  # type: [int, int, int] # input array dimensions [dimX, dimY, dimZ]

        # read static parameters from config
        self.config = helper.parse_json("kernel.config")
        self.calculator = Calculator()

        # analyse input
        self.graph = ComputeGraph()
        self.graph.generate_graph(kernel_string)
        self.graph.calculate_latency()
        self.graph.determine_inputs_outputs()
        self.graph.setup_internal_buffers()
        if plot_graph:
            self.graph.plot_graph(name + ".png")

        # init sim specific params
        self.var_map = None  # var_map[var_name] = var_value
        self.read_success = False
        self.exec_success = False
        self.result = None  # type: float
        self.inputs = dict()  # type: dict()  # self.inputs[name] = queue
        self.outputs = dict()  # type: dict()  # self.inputs[name] = queue

        # output delay queue: for simulation of calculation latency, fill it up with bubbles
        self.out_delay_queue = BoundedQueue("delay_output", self.graph.max_latency, [None]*(self.graph.max_latency-1))

        # setup internal buffer queues
        self.internal_buffer = dict()
        self.setup_internal_buffers()

    def iter_comp_tree(self, node):

        pred = list(self.graph.graph.pred[node])

        if isinstance(node, Binop):
            lhs = pred[1]
            rhs = pred[0]
            return "(" + self.iter_comp_tree(lhs) + " " + node.generate_op_sym() + " " + self.iter_comp_tree(rhs) + ")"
            # return self.iter_comp_tree(lhs) + " " + node.generate_op_sym() + " " + self.iter_comp_tree(rhs)
        elif isinstance(node, Call):
            return node.name + "(" + self.iter_comp_tree(pred[0]) + ")"
        elif isinstance(node, Name) or isinstance(node, Num):
            return str(node.name)
        elif isinstance(node, Subscript):
            dim_index = helper.list_subtract_cwise(node.index, self.graph.max_index[node.name])
            word_index = self.convert_3d_to_1d(dim_index)
            return node.name + "[" + str(word_index) + "]"

    def generate_relative_access_kernel_string(self):
        # format: 'res = vdc[index1] + vout[index2]'

        # find output node
        res = None
        for node in self.graph.graph.nodes:
            if isinstance(node, Output):
                res = node

        return "res = " + self.iter_comp_tree(list(self.graph.graph.pred[res])[0])

    def reset_old_compute_state(self):
        self.var_map = dict()
        self.read_success = False
        self.exec_success = False
        self.result = None

    def convert_3d_to_1d(self, index):
        # convert [i, j, k] to flat 1D array index using the given dimensions [dimX, dimY, dimZ]
        # index = i*dimY*dimZ + j*dimZ + k = (i*dimY + j)*dimZ + k
        if not index:
            return 0  # empty list
        return (index[0] * self.dimensions[1] * self.dimensions[2] +
                index[1] * self.dimensions[2]) + index[2]

    def setup_internal_buffers(self):

        for buf_name in self.graph.buffer_size:
            self.internal_buffer[buf_name] = BoundedQueue(name=buf_name,
                                                          maxsize=self.convert_3d_to_1d(self.graph.buffer_size[buf_name])+1)

    def buffer_position(self, access):
        return self.convert_3d_to_1d(self.graph.min_index[access.name]) - self.convert_3d_to_1d(access.index)

    def try_read(self):

        # TODO: test this method as soon as the kernel_chain_graph has linked the input queues with the output queues
        # of different kernels

        # reset old state
        self.reset_old_compute_state()

        # check if all inputs are available
        all_available = True
        for inp in self.graph.inputs:
            if isinstance(inp, Num):
                pass
            elif self.inputs[inp.name].peek(self.buffer_position(inp)) is None:  # check if array access location
                #  is filled with a bubble
                all_available = False
                break

        # get all values and put them into the variable map
        if all_available:
            for inp in self.inputs:
                # read inputs into var_map
                if isinstance(inp, Num):
                    self.var_map[inp.name] = float(inp.name)
                elif isinstance(inp, Name):
                    # get value from internal_buffer
                    try:
                        self.var_map[inp.name] = self.internal_buffer[inp.name].peek(self.buffer_position(inp))
                    except Exception as ex:
                        self.diagnostics(ex)

        self.read_success = all_available

        if self.read_success:
            # pop oldest element from all queues
            for queu in self.inputs:
                queu[1].dequeue()

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

    kernel1 = Kernel("ppgk", "res = wgtfac[i,j+1,k] * ppuv[i,j,k] + (1.0 - wgtfac[i,j,k]) * ppuv[i,j,k-1];", [10, 10, 10])
    print("Critical path latency: " + str(kernel1.graph.max_latency))

    kernel2 = Kernel("dummy", "res = (sin(a[i,j,k])-b[i,j,k]) * (a[i,j,k-1]+b[i,j-1,k-1])", [10, 10, 10])
    print("Kernel string conversion:")
    print(kernel2.kernel_string)
    print(kernel2.generate_relative_access_kernel_string())
